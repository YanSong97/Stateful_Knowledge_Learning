import asyncio
import os
import pdb
import re
import socket
import sys
import tempfile
from contextlib import asynccontextmanager
from typing import Any, Dict
from collections import defaultdict

import aiohttp
import fastapi
import numpy as np
import ray
import uvicorn
from datasets import load_dataset
from omegaconf import OmegaConf
from openai.types.chat.chat_completion import ChatCompletion
from starlette.requests import Request
from starlette.responses import JSONResponse
import random
from typing import Any, Dict, List, Union, Callable
import torch
from tensordict import TensorDict

from examples.test_lpm import action_lookup
from src.scheduler.naive_chat_scheduler import NaiveChatCompletionScheduler
from examples.async_rollout_utils import SeedGenerator
from verl.protocol import DataProto
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from src.envs import get_env
from src.envs.frozen_lake.gym_frozenlake.env import FrozenLakeEnv
from transformers.utils.chat_template_utils import render_jinja_template
import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask

from dataclasses import dataclass
from typing import Optional
import importlib

DEBUG=False

def verify_action(action, available_actions):
    action_lookup = {"left": 1, "down": 2, "right": 3, "up": 4}

    if action is None:
        return available_actions[int(np.random.choice(len(available_actions), 1)[0])], 0.

    action = action.lower().strip()
    if action in action_lookup:
        return action_lookup[action], 1.
    else:
        return available_actions[int(np.random.choice(len(available_actions), 1)[0])], 0.


class LPMChatCompletionScheduler(NaiveChatCompletionScheduler):
    """This is a chat completion scheduler that supports env rollout and internal planning
    here the scheduler does not repeat the batch data, but need to interleave
    """

    def __init__(self, config, model_path, server_addresses, env_config, **kwargs):
        super().__init__(config, model_path, server_addresses, **kwargs)
        self.env_config = env_config

        # prompt set
        module_path, class_name = env_config.prompt_set.rsplit(".", 1)
        module = importlib.import_module(module_path)
        prompt_set = getattr(module, class_name)

        # system prompt
        self.system_prompt = prompt_set["system_prompt"]

        # query instruction
        self.act_query_prompt = prompt_set["act_query_prompt"]
        self.plan_query_prompt = prompt_set['plan_query_prompt']
        self.plan_action_query_prompt = prompt_set['plan_action_query_prompt']

        # env termination feedback
        self.success_prompt = prompt_set["success_prompt"]
        self.fail_prompt = prompt_set["fail_prompt"]

        # plan feedback
        self.plan_success_prompt = prompt_set['plan_success_prompt']
        self.plan_fail_prompt = prompt_set['plan_fail_prompt']
        self.no_valid_decision_feedback = prompt_set['no_valid_decision_feedback']


        # model template set
        if env_config.template_set is not None:
            module_path, class_name = env_config.template_set.rsplit(".", 1)
            module = importlib.import_module(module_path)
            model_template = getattr(module, class_name)
            self.tokenizer.chat_template = model_template     #

        # action processor
        module_path, class_name = env_config.action_processor.rsplit(".", 1)
        module = importlib.import_module(module_path)
        self.extract_answer = getattr(module, class_name)

        self.move_tag = ["<move>", "</move>"]
        self.reset_tag = ["<reset>", "</reset>"]
        self.end_tag = ["<end>", "</end>"]
        self.answer_tag = ["<answer>", "</answer>"]

        # envs
        # self.batched_env = []
        # self.batched_state_query = []
        # self.batched_state = []
        self.available_actions_lst = [1, 2, 3, 4]

        self.conversation_prefix = "tool"
        self.plan_prefix = "plan"
        self.max_turns=self.config.multi_turn.max_turns     # for acting
        self.is_validate=None
        self.max_plan_traj_n = int(self.config.multi_turn.max_plan_traj_n)      # for planning
        self.max_plan_horizon = int(self.config.multi_turn.max_plan_horizon)    # for planning

        # seed generator
        train_seed = eval(str(self.config.multi_turn.envs.env_config.rollout.train_seed))
        val_seed = eval(str(self.config.multi_turn.envs.env_config.rollout.val_seed))
        self.train_seed_pool = SeedGenerator(start_range=train_seed[0], end_range=train_seed[1], shuffle=True)
        self.val_seed_pool = SeedGenerator(start_range=val_seed[0], end_range=val_seed[1], shuffle=True)

        self.each_response_max_length = self.config.each_response_length
        self.total_response_max_length = self.config.response_length
        self.total_plan_response_max_length = self.config.total_plan_response_max_length

        # reward settings
        self.act_action_format_penality = self.config.multi_turn.reward.act_action_format_penality


    def build_env(self, n: int, repeat: int, interleave: bool = True):
        assert interleave
        self.batched_env = []
        self.batched_state_query = []
        self.batched_state = []

        for _ in range(n):
            # retrive random seed
            if self.is_validate:
                _seed = next(self.val_seed_pool)
            else:
                _seed = next(self.train_seed_pool)

            repeat_env = []
            repeat_state = []
            repeat_state_query = []
            for i in range(repeat):
                _single_env = get_env(self.env_config)
                _state = _single_env.reset(_seed)
                _query = self.act_query_prompt.format(state=_state, available_move='1,2,3,4', turn_idx=1, turn_left=self.max_turns)
                repeat_env.append(_single_env)
                repeat_state.append(_state)
                repeat_state_query.append(_query)

            self.batched_env.extend(repeat_env)
            self.batched_state.extend(repeat_state)
            self.batched_state_query.extend(repeat_state_query)



    async def generate_sequences(self, batch: DataProto, **sampling_params) -> DataProto:
        """
        the batch has not been repeated
        Args:
            batch:
            **sampling_params:

        Returns:

        """
        kwargs = dict(
            n=1, #self.config.n,   keep at 1, and create n envs
            max_completion_tokens=self.each_response_max_length,     # each max response length
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            include_stop_str_in_output=True,
            # extra_body={
            #     "include_stop_str_in_output": True,
            #     "stop": ["</answer>"],
            # },
        )       # TODO: configurable

        do_sample = batch.meta_info.get("do_sample", True)
        is_validate = batch.meta_info.get("validate", False)
        if not do_sample or is_validate:
            kwargs["n"] = 1
            kwargs["temperature"] = 0
            self.is_validate = True
            repeat_num = 1
        else:
            self.is_validate = False
            repeat_num = self.config.multi_turn.envs.env_config.rollout.n       # number of parallel env

        kwargs.update(sampling_params)
        if DEBUG:
            print(f"[EnvChatCompletionScheduler] generate_sequences sampling params: {kwargs}")

        max_turns = self.max_turns

        def tool_one_step_env(env, action):
            next_state, reward, done, info = env.step(action)
            env_success = env.check_success()
            falling = env.check_falling()
            return next_state, reward, done, env_success, falling

        def tool_reset_env(env, target_state):
            return env.reset_state(target_state)

        async def dummy_fn(completions, callback_additional_info, exception):
            tmp_holder = callback_additional_info['tmp_holder']
            tmp_holder.append(completions)
            return

        async def linear_planning(completions: ChatCompletion, info: Dict[str, Any], exception: Exception):
            """
            planning internally with multiple tool call. Here agent can choose to plan, reset, or end planning
            The workflow to check the decision condition is:
            1. Consider initial planning case
            2. Receive completion, parse decision
            3. Handle decision:
                a. if end, parse final action and return
                b. if move, perform one-step simulation, get next state:
                    - if game terminate due to internal reason or reaching max planning horizon ,
                        get final feedback and append to conversation, ready for reset, jump tp 3.c
                    - if already_overlong, get final feedback, ready for summerization, jump to 4
                c. if reset, need to consider current planning budget:
                    - if planning budget is empty (reach maximum planning traj, maximum planning tokens), stop reset,
                        ready for summerization, jump to 4
                    - if budget is enough, reset simulation
            4. Handle summerization case:
                a. seed a summerization request, parse answers and return
            5. Submit planning request
            """
            (plan_conversations,
             simulated_env,
             available_tool_call,
             final_action,
             plan_traj,
             plan_info) = (
                info['plan_conversations'],
                info['simulated_env'],
                info['available_tool_call'],
                info['final_action'],
                info['plan_traj'],
                info['plan_info']
            )

            original_planning_stats_dict = plan_info['original_planning_stats_dict']
            original_planning_state = plan_info['original_planning_state']

            #======================= Initialize Planning Case ============================================#
            if plan_info['plan_idx']==0 and plan_info["plan_step"]==0:
                assert available_tool_call == ['MOVE', "END"]
                # get rollout prompts
                tool_call_prompt = [
                    # {
                    # "role": 'system',
                    # "content": self.system_prompt_for_plan
                # },
                    {
                        "role": self.plan_prefix,
                        "content": self.plan_query_prompt.format(
                            start_state = original_planning_state,
                            current_state = simulated_env.render(),
                            available_decision=",".join(available_tool_call),
                            turn_idx = 1,
                            step_idx = 1,
                            turn_left = self.max_plan_traj_n-1,
                            max_step = self.max_plan_horizon,
                            extra_info = "the simulation starts"
                        ),
                        "type": "plan",
                        "id": f"{plan_conversations[1]['id']}-plan_turn_1-plan_step_1"
                    }
                ]
                plan_conversations.extend(tool_call_prompt)
                plan_traj['state'].append(simulated_env.render())

                plan_info["plan_idx"] += 1
                plan_info["plan_step"] += 1

                # start planning
                await self.submit_chat_completions(
                    callback=linear_planning,
                    callback_additional_info={
                        "plan_conversations": plan_conversations,
                        "simulated_env": simulated_env,
                        "available_tool_call": available_tool_call,
                        "final_action": final_action,
                        "plan_traj": plan_traj,
                        "plan_info": plan_info
                    },
                    model=self.model_name,
                    messages=plan_conversations,
                    stop = [self.move_tag[-1], self.reset_tag[-1], self.end_tag[-1]],        # stop at each decision
                    **kwargs,
                )
                return

            #====================== Process Assistant Planning Response =============================#
            role, content = completions.choices[0].message.role, completions.choices[0].message.content
            plan_conversations.append(
                {"role": role,
                 "content": content,
                 "type": "plan",
                 "id": f"{plan_conversations[1]['id']}-plan_turn_{plan_info['plan_idx']}-plan_step_{plan_info['plan_step']}"
                 }
            )
            plan_info['plan_token_usage'] += completions.usage.completion_tokens
            already_overlong = (plan_info['plan_token_usage'] >
                                self.total_plan_response_max_length)
            single_call_overlong = int(completions.usage.completion_tokens == self.each_response_max_length)
            plan_info["overlong"] = already_overlong
            plan_info['single_call_overlong'] = single_call_overlong

            if DEBUG:
                print(f"\nPlanning Turn {plan_info['plan_idx']}, steps {plan_info['plan_step']}, planning content = {content}")

            #====================== Parse Decision ====================================================#
            ready_summerize = False
            real_action = None
            plan_action = None
            reset_plan = None

            move_matches = re.findall(r"{tag_1}(.*?){tag_2}".format(tag_1=self.move_tag[0], tag_2=self.move_tag[1]), content, re.DOTALL)
            reset_matches = re.findall(r"{tag_1}(.*?){tag_2}".format(tag_1=self.reset_tag[0], tag_2=self.reset_tag[1]), content, re.DOTALL)
            end_matches = re.findall(r"{tag_1}(.*?){tag_2}".format(tag_1=self.end_tag[0], tag_2=self.end_tag[1]),content, re.DOTALL)
            answer_matches = re.findall(r"{tag_1}(.*?){tag_2}".format(tag_1=self.answer_tag[0], tag_2=self.answer_tag[1]), content, re.DOTALL)
            if answer_matches:
                # Case 1: if end tag detected, return final decision
                # get final planning decision answer
                real_action = answer_matches[0]
                final_action.append(real_action)
                plan_traj['action'].append(f"Decision: {real_action}")
                return

            if end_matches:
                plan_action = None
                reset_plan = None
                ready_summerize = True
                plan_traj['plan_valid_decision'].append(1)
                plan_traj['action'].append("end")

            else:
                if move_matches:        # if moving decision detected
                    plan_action, valid_action = self.extract_answer(content, available_actions=[1,2,3,4])
                    plan_traj['plan_valid_action'].append(valid_action)

                if reset_matches:       # if reset decision detected
                    if plan_info['plan_idx'] == self.max_plan_traj_n:
                        # when no more planning traj budget left, jump to summerization phase directly
                        reset_plan = False
                        ready_summerize = True
                    else:
                        reset_plan = True

                if not move_matches and not reset_matches:      # if no decision is detected
                    plan_conversations.append(
                        {"role": self.plan_prefix,
                         "content": "No valid action detected. End current planning turn.",
                         "type": "plan",
                         "id": f"{plan_conversations[1]['id']}_plan_turn_{plan_info['plan_idx']}-plan_step_{plan_info['plan_step']}"
                         }
                    )
                    plan_traj['plan_valid_decision'].append(0)
                    if plan_info['plan_idx'] == self.max_plan_traj_n:        # summerize when no planning budget left
                        reset_plan = False
                        ready_summerize = True
                    else:
                        reset_plan = True
                else:
                    plan_traj['plan_valid_decision'].append(1)

                if already_overlong:
                    # overlong overwrites all condition
                    ready_summerize = True


            #==================== Process Decision ===================================================#
            plan_traj['assistant_reward'].append(0)       # reward value placeholder, for easy post-processing credit assignment

            if plan_action is not None:     # env rollout
                next_state, reward, done, env_success, falling = tool_one_step_env(simulated_env, plan_action)
                plan_traj['state'].append(next_state)
                plan_traj['action'].append(plan_action)
                plan_traj['reward'].append(reward)
                # if the game terminate, append the final feedback to conversation and ready for reset
                # Check Planning Termination
                if plan_info['plan_step'] == self.max_plan_horizon:
                    if done and falling:        # if the game happen to finish
                        fail_reason = "because falling into holes"
                    else:
                        fail_reason = "because reaching maximum planning steps ahead"
                        done = True

                else:
                    if already_overlong:
                        fail_reason = "because overlong"
                        assert ready_summerize
                        done = True
                    elif done and falling:
                        fail_reason = "because falling into hole"
                    else:
                        fail_reason = ""

                plan_info['plan_step'] += 1
                plan_traj['done'].append(done)

                if done:
                    distance2goal = simulated_env.compute_distance2goal()
                    # check success if the game has terminated
                    end_info = self.success_prompt.format(state=next_state) if env_success else self.fail_prompt.format(
                        state=next_state, fail_reason=fail_reason, extra_info=f"You are still {distance2goal} blocks away from the goal.")
                    plan_traj['end_info'].append(end_info)
                    plan_conversations.append(
                        {"role": self.plan_prefix,
                         "content": end_info,  #.format(state=next_state)
                         "type": "plan",
                         "id": f"{plan_conversations[1]['id']}-plan_turn_{plan_info['plan_idx']}-plan_step_{plan_info['plan_step']}",
                         }
                    )  # final env return

                    # ready for reset
                    reset_plan = True
                    if plan_info['plan_idx'] == self.max_plan_traj_n:        # but when running out of planning budget, summerize
                        ready_summerize = True
                else:
                    available_tool_call=['MOVE', "RESET", "END"]
                    plan_conversations.append(
                        {
                            "role": self.plan_prefix,
                            "content": self.plan_query_prompt.format(
                                start_state=original_planning_state,
                                current_state=next_state,
                                available_decision=",".join(available_tool_call),
                                turn_idx=plan_info['plan_idx'],
                                step_idx=plan_info['plan_step'],
                                turn_left=self.max_plan_traj_n-plan_info['plan_idx'],
                                max_step=self.max_plan_horizon-plan_info['plan_step']+1,
                                extra_info="the simulation continues"
                            ),
                            "type": "plan",
                            "id": f"{plan_conversations[1]['id']}-plan_turn_{plan_info['plan_idx']}-plan_step_{plan_info['plan_step']}",
                        }
                    )

            # process summerization first, before processing reset
            if ready_summerize:
                plan_traj['action'].append('summerize')
                # finish planning, ask for final answer summerization
                final_answer = []
                plan_conversations.append(
                    {
                        "role": self.plan_prefix,
                        "content": self.plan_action_query_prompt.format(state=original_planning_state),
                        "type": "plan",
                        "id": f"{plan_conversations[1]['id']}-plan_turn_{plan_info['plan_idx']}-plan_step_{plan_info['plan_step']}",
                    }
                )

                await self.submit_chat_completions(
                    callback=dummy_fn,
                    callback_additional_info={
                        "tmp_holder": final_answer,
                    },
                    model=self.model_name,
                    messages=plan_conversations,
                    stop=[self.answer_tag[-1]],
                    **kwargs,
                )
                role, content = final_answer[0].choices[0].message.role, final_answer[0].choices[0].message.content
                answer_matches = re.findall(r"{tag_1}(.*?){tag_2}".format(tag_1=self.answer_tag[0], tag_2=self.answer_tag[1]), content, re.DOTALL)
                if answer_matches:
                    final_action.append(answer_matches[0])
                else:
                    final_action.append(None)

                single_call_overlong = int(final_answer[0].usage.completion_tokens == self.each_response_max_length)
                plan_traj['single_call_overlong'].append(single_call_overlong)

                if DEBUG:
                    print(f"Final answer = {content}")
                plan_conversations.extend([
                    {
                        "role": role,
                        "content": content,
                        "type": "plan",
                        "id": f"{plan_conversations[1]['id']}-plan_turn_{plan_info['plan_idx']}-plan_step_{plan_info['plan_step']}",
                     },
                ]
                )
                # reward placeholder is not added here. The true reward in external environment is given later

                plan_traj['action'].append(f"Decision: {final_action[0]}")
                plan_info['plan_token_usage'] += final_answer[0].usage.completion_tokens
                return


            # process reset
            if reset_plan:     # still has planning budget and choose to replan
                plan_info['plan_idx'] += 1
                plan_info['plan_step'] = 1
                # if plan_idx < self.max_plan_traj_n-1:       # we still have not reach maximum planning traj
                simulated_env = tool_reset_env(simulated_env, original_planning_stats_dict)
                available_tool_call = ["MOVE", "END"]
                plan_conversations.append(
                    {
                        "role": self.plan_prefix,
                        "content": self.plan_query_prompt.format(
                            start_state=original_planning_state,
                            current_state=simulated_env.render(),
                            available_decision=",".join(available_tool_call),
                            turn_idx=plan_info['plan_idx'],
                            step_idx=plan_info['plan_step'],
                            turn_left=self.max_plan_traj_n-plan_info['plan_idx'],
                            max_step=self.max_plan_horizon,
                            extra_info="the simulation has been restarted to original game state"
                        ),
                        "type": "plan",
                        "id": f"plan_turn_{plan_info['plan_idx']}-plan_step_{plan_info['plan_step']}",
                    }
                )


            # =========================== Submit chat completions for further planning ================================#
            extra_headers = {"x-request-id": completions.id}
            await self.submit_chat_completions(
                callback=linear_planning,
                callback_additional_info={
                    "plan_conversations": plan_conversations,
                    "simulated_env": simulated_env,
                    "available_tool_call": available_tool_call,
                    "final_action": final_action,
                    "plan_traj": plan_traj,
                    "plan_info": plan_info,
                },
                model=self.model_name,
                messages=plan_conversations,
                extra_headers=extra_headers,
                stop=[self.move_tag[-1], self.end_tag[-1], self.reset_tag[-1], self.answer_tag[-1]], #["</answer>", "</move>", "</reset>"],
                **kwargs,
            )


        async def external_acting(info: Dict[str, Any]):
            """
            acting in external environment
            """

            batch_conversations, batch_traj, batch_token_n, batch_index, turn, batch_plan_conversations = (
                info["batch_conversations"],
                info["batch_traj"],
                info["batch_token_n"],
                info["batch_index"],
                info["turn"],
                info["batch_plan_conversations"],
            )
            # =================== Starting Planning =================================================#
            # if completions is None:
            original_planning_stats_dict = self.batched_env[batch_index].get_key_stats()
            new_env = get_env(self.env_config).reset_state(original_planning_stats_dict)

            batch_plan_conversations[batch_index].append([])
            plan_conversations = batch_plan_conversations[batch_index][-1] #defaultdict(list)
            # add prompts in real-world
            plan_conversations.extend(batch_conversations[batch_index])     # external conversation as prefix

            plan_traj = defaultdict(list)
            final_answer_lst = []
            plan_info = {"plan_step": 0, "plan_idx": 0, "plan_token_usage": 0,
                         "original_planning_stats_dict": original_planning_stats_dict,
                         "original_planning_state": new_env.render(), "overlong": False}
            await linear_planning(
                completions = None,
                info = {
                    "plan_conversations": plan_conversations,
                    "simulated_env": new_env,
                    "available_tool_call": ['MOVE', "END"],
                    "final_action": final_answer_lst,
                    "plan_traj": plan_traj,
                    "plan_info": plan_info,
                },
                exception=None
            )
            _final_answer = final_answer_lst[-1]
            if DEBUG:
                print(f"FINAL action = {_final_answer}")
            batch_traj[batch_index]['reward'].extend(plan_traj['assistant_reward'])
            batch_traj[batch_index]['plan'].append(plan_traj)
            batch_traj[batch_index]['plan_completion'].append(plan_info['plan_token_usage'])
            batch_traj[batch_index]['plan_overlong'].append(int(plan_info['overlong']))
            batch_traj[batch_index]['plan_turns'].append(plan_info['plan_idx'])

            batch_conversations[batch_index].append(
                {"role": self.conversation_prefix,      # TODO[yan]: tool of assistant
                 # "content": plan_conversations[-1]['content'] + f"\nPlanned action: {_final_answer}",
                 "content": f"\nPlanned action: {_final_answer}",
                 "type": "act",
                 "id": f"batch_{batch_index}-step_{turn}"
                 }
            )

            # process final decision
            planned_action, action_is_valid = verify_action(_final_answer, available_actions=self.available_actions_lst)

            # # ==================== assistant response ==============================================#
            # role, content = completions.choices[0].message.role, completions.choices[0].message.content
            # batch_conversations[batch_index].append({"role": role, "content": content})
            # token_usage = completions.usage.completion_tokens
            # batch_token_n[batch_index] += token_usage
            # print(f"\nActing content={content}")

            #=========================== Env Step Fn ====================================================#
            # extracted_action, action_is_valid = extract_answer(content, available_actions=[1,2,3,4])
            batch_traj[batch_index]['action'].append(planned_action)
            batch_traj[batch_index]['action_is_valid'].append(action_is_valid)

            next_state, reward, done, info = self.batched_env[batch_index].step(planned_action)
            turn += 1

            if not action_is_valid:     # penalize non-valid action choice
                reward += self.act_action_format_penality
            if DEBUG:
                print(f"[turn={turn}] Env stepping {next_state}")
            batch_traj[batch_index]['state'].append(next_state)
            batch_traj[batch_index]['reward'].append(reward)
            batch_traj[batch_index]['terminate'].append(done)

            #=========================== Check Termination ============================================#
            if done:
                # check success
                env_success = self.batched_env[batch_index].check_success()
                end_info = self.success_prompt.format(state=next_state) if env_success else self.fail_prompt.format(state=next_state, fail_reason="because failling into holes")
                # if batch_token_n[batch_index] >= self.total_response_max_length and not env_success:
                #     end_info = fail_prompt.format(state=next_state, fail_reason="because reaching maximum response length")

                batch_conversations[batch_index].append(
                    {"role": self.conversation_prefix, "content": end_info.format(state=next_state),
                     "type": "act", "id": f"batch_{batch_index}-step_{turn}"}
                )     # final env return
                batch_traj[batch_index]['success'].append(float(env_success))
                if DEBUG:
                    print(f"[turn={turn}] Env terminate, done!")
                return

            if turn > max_turns:
                batch_conversations[batch_index].append(
                    {"role": self.conversation_prefix,
                     "content": self.fail_prompt.format(state=next_state, fail_reason="because reaching maximum turns"),
                     "type": "act",
                     "id": f"batch_{batch_index}-step_{turn}"
                     }
                )     # final env return
                batch_traj[batch_index]['success'].append(0.)
                if DEBUG:
                    print(f"turn={turn}] Reach max turns {max_turns}, done!")
                return



            batch_conversations[batch_index].append(
                {"role": self.conversation_prefix,
                 "content": self.act_query_prompt.format(state=next_state, available_move='1,2,3,4', turn_idx=turn, turn_left=max_turns-turn+1),
                 "type": "act",
                 "id": f"batch_{batch_index}-step_{turn}"
                 }
            )
            if DEBUG:
                print(f"[turn={turn}] Env continue ...")


            #=========================== Submit chat completions ========================================#
            await external_acting(
                info = {
                    "batch_conversations": batch_conversations,
                    "batch_traj": batch_traj,
                    "batch_token_n": batch_token_n,
                    "batch_index": batch_index,
                    "turn": turn,
                    "batch_plan_conversations": batch_plan_conversations,
                },
            )


        # repeat batch to parallelize envs
        gen_batch = batch.repeat(repeat_times=repeat_num, interleave=True)
        tasks, batch_conversations, batch_traj = [], [None] * len(gen_batch), [defaultdict(list) for _ in range(len(gen_batch))]
        batch_token_n = [0 for _ in range(len(gen_batch))]
        batch_plan_conversations = [[] for _ in range(len(gen_batch))]


        # build local batched env
        self.build_env(n=len(batch), repeat=repeat_num, interleave=True)        # for each original samples, repeat

        # replace raw prompt with sys_prompt + initial_state query TODO[yan]: can we do it outside the scheduler?
        new_raw_prompt = []
        for batch_index, conversation in enumerate(gen_batch.non_tensor_batch["raw_prompt"]):
            sys_p = {"role": "system", "content": self.system_prompt, "type": 'system'}
            conversation[0]['content'] += self.batched_state_query[batch_index]
            conversation[0]['type'] = 'act'
            conversation[0]['id'] = f"batch_{batch_index}-step_1"

            new_raw_prompt.append([sys_p, conversation[0]])
        gen_batch.non_tensor_batch["processed_prompt"] = np.array(new_raw_prompt)       # TODO[yan]: no need to replace raw_prompt ?


        for batch_index, conversation in enumerate(gen_batch.non_tensor_batch["processed_prompt"]):
            # raw_prompt: [{"role": "user", "content": ""}, ["role": "assistant", "content"], ...]
            batch_conversations[batch_index] = conversation.tolist()        # need to contain complete prompt and query
            batch_traj[batch_index]['state'].append(self.batched_state[batch_index])

            tasks.append(
                asyncio.create_task(
                    external_acting(
                        info={
                            "batch_conversations": batch_conversations,
                            "batch_traj": batch_traj,
                            "batch_token_n": batch_token_n,
                            "batch_index": batch_index,
                            "turn": 1,
                            "batch_plan_conversations": batch_plan_conversations,
                        },
                    )
                )
            )

        await asyncio.gather(*tasks)
        if DEBUG:
            print("[EnvChatChatCompletionScheduler] generate_sequences done")
            print(f"\nTraj = {batch_traj}\n")

        # _postprocess assumes n>=1
        batch_conversations = [[conversation] for conversation in batch_conversations]
        return self._postprocess(gen_batch, batch_conversations, batch_plan_conversations, repeat_num, batch_traj)


    def _postprocess(self,
                     batch: DataProto,
                     batch_conversations: List[List[Dict[str, str]]],
                     batch_plan_conversations: List[List[Dict[int, List]]],
                     n: int,
                     traj) -> DataProto:
        # NOTE: consistent with batch version of generate_sequences in vllm_rollout_spmd.py
        # prompts: left pad
        # responses: right pad
        # input_ids: prompt + response
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        non_tensor_batch = batch.non_tensor_batch if hasattr(batch, "non_tensor_batch") else {}

        #======================== Processing Prompt ===============================#
        # prompts: [prompt] from input dataset, identical to self.tokenizer.apply_chat_template(batch_conversations[0][:2],  add_generation_prompt=True, tokenize=False)
        raw_prompts = [self.tokenizer.apply_chat_template(prompt, add_generation_prompt=False, tokenize=False) for prompt in batch.non_tensor_batch["processed_prompt"]]
        model_inputs = self.tokenizer(raw_prompts, return_tensors='pt', add_special_tokens=False, padding=True, padding_side='left')
        prompt_input_ids = model_inputs.pop("input_ids")
        prompt_attention_mask = model_inputs.pop("attention_mask")

        pad_prompt_input_ids, pad_prompt_attention_mask = verl_F.postprocess_data(
            input_ids = prompt_input_ids,
            attention_mask = prompt_attention_mask,
            max_length = self.config.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.config.truncation
        )
        pad_prompt_position_ids = compute_position_id_with_mask(pad_prompt_attention_mask)

        assert len(batch_conversations) == len(raw_prompts)
        batch_conversations = [conversation for conversations in batch_conversations for conversation in conversations]     # [batch_size*repeat, episode_length, 2, ]
        batch_plan_act_conversations = []

        #=========================== Processing Response =============================#
        for idx in range(len(batch_conversations)):
            plan_act_conversation = []
            plan_act_conversation.append(batch_conversations[idx][0])   # sys
            state_index = np.arange(1, len(batch_conversations[idx])-2, 2)
            for sid in state_index:
                assert batch_conversations[idx][sid]['type'] == 'act'
                plan_act_conversation.append(batch_conversations[idx][sid])                 # state query
                pure_planning_traj = [i for i in batch_plan_conversations[idx][sid//2] if i['type'] == 'plan']
                plan_act_conversation.extend(pure_planning_traj)                        # planning traj

                assert batch_conversations[idx][sid+1]['type'] == 'act'
                plan_act_conversation.append(batch_conversations[idx][sid+1])           # planned action
            plan_act_conversation.append(batch_conversations[idx][-1])                  # final feedback

            batch_plan_act_conversations.append(plan_act_conversation)


        response_input_ids = []
        response_attention_mask = []
        response_loss_mask = []
        response_reward = []
        raw_env_reward = [i["reward"] for i in traj]
        for idx, conversation in enumerate(batch_plan_act_conversations):
            assert len([i for i in conversation if i['role']=='assistant']) == len(raw_env_reward[idx]), \
                    print(f"There are {len([i for i in conversation if i['role']=='assistant'])} assistant response but {len(raw_env_reward[idx])} reward values.")
            # if len([i for i in conversation if i['role']=='assistant']) != len(raw_env_reward[idx]):
            #     import pdb
            #     pdb.set_trace()

            merge_conversation = self.merge_conversation(conversation)

            ret = self.customize_chat_template(conversation=merge_conversation,
                                               add_generation_prompt=False,
                                               return_dict=True,
                                               tokenize=True,
                                               return_assistant_tokens_mask=True,
                                               env_reward_lst=raw_env_reward[idx],
                                               )
            _raw_prompt_length = prompt_attention_mask[idx].sum()
            if DEBUG:
                print(f"raw prompt length = {_raw_prompt_length}, response length = {len(ret['input_ids'])}")

            # check first few tokens align with raw prompts
            assert (prompt_input_ids[idx][-_raw_prompt_length:] - torch.tensor(ret['input_ids'][:_raw_prompt_length])).sum().item()==0, print(f"prompt inputs does not align")
            assert len(ret['input_ids'][_raw_prompt_length:]) <= self.config.response_length, print(f"response input shape = {len(ret['input_ids'][_raw_prompt_length:])} has exceed max response length {self.config.response_length}")

            response_input_ids.append(ret['input_ids'][_raw_prompt_length:])       # throw away raw prompt
            response_attention_mask.append(ret['attention_mask'][_raw_prompt_length:])
            response_loss_mask.append(ret['assistant_masks'][_raw_prompt_length:])
            response_reward.append(ret['assistant_reward'][_raw_prompt_length:])

        # assert response_input_ids.size(1) <= self.config.response_length, print(f"response input shape = {response_input_ids.shape} has exceed max response length {self.config.response_length}")
        # concate
        pad_response_input_ids = pad_2d_list_to_length(response_input_ids,
                                                   self.tokenizer.pad_token_id,
                                                   max_length=self.config.response_length)

        pad_response_attention_mask = pad_2d_list_to_length(response_attention_mask, 0, max_length=self.config.response_length)
        pad_response_loss_mask = pad_2d_list_to_length(response_loss_mask, 0, max_length=self.config.response_length)
        pad_response_reward = pad_2d_list_to_length(response_reward, 0, max_length=self.config.response_length)
        delta_position_id = torch.arange(1, pad_response_input_ids.size(1) + 1)
        delta_position_id = delta_position_id.unsqueeze(0).expand(pad_response_input_ids.size(0), -1)
        pad_response_position_ids = pad_prompt_position_ids[..., -1:] + delta_position_id

        completion_input_ids = torch.cat([pad_prompt_input_ids, pad_response_input_ids], dim=-1)
        completion_attention_mask = torch.cat([pad_prompt_attention_mask, pad_response_attention_mask], dim=-1)
        completion_position_ids = torch.cat([pad_prompt_position_ids, pad_response_position_ids], dim=-1)

        if not self.config.multi_turn.post_process.assistant_loss_mask:
            pad_response_loss_mask = completion_attention_mask

        batch = TensorDict(
            {
                "prompts": prompt_input_ids,
                "responses": pad_response_input_ids,
                "input_ids": completion_input_ids,
                "attention_mask": completion_attention_mask,
                "position_ids": completion_position_ids,
                "reward": pad_response_reward,
                'loss_mask': pad_response_loss_mask,
                # "terminate_mask": terminate_mask,
            },
            batch_size=len(pad_response_input_ids),
        )

        # record success rate
        success_rate = [i['success'] for i in traj]
        action_valid_rate = [[np.mean(i['action_is_valid'])] for i in traj]
        total_plan_completion = [sum(i['plan_completion']) for i in traj]
        plan_overlong_rate = [np.mean(i['plan_overlong']) for i in traj]
        plan_turns = [np.mean(i['plan_turns']) for i in traj]

        # overlong
        completion_overlong_rate = []
        for i in traj:
            each_plan = [j for j in i['plan']]
            each_overlong_rate = np.concatenate([k['single_call_overlong'] for k in each_plan])
            completion_overlong_rate.append(np.mean(each_overlong_rate))


        non_tensor_batch['success_rate'] = np.array(success_rate)
        non_tensor_batch['action_valid_rate'] = np.stack(action_valid_rate)
        non_tensor_batch['total_plan_completion'] = np.array(total_plan_completion)
        non_tensor_batch['plan_overlong_rate'] = np.array(plan_overlong_rate)
        non_tensor_batch['plan_turns'] = np.array(plan_turns)
        non_tensor_batch['completion_overlong_rate'] = np.array(completion_overlong_rate)

        ret = DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

        ret.check_consistency()

        return ret

    def merge_conversation(self, conversation_lst):

        new_conversation = [
            dict(conversation_lst[0]),  # create new dict copies
            dict(conversation_lst[1])
        ]

        for i in range(2, len(conversation_lst)):
            current_msg = conversation_lst[i]
            if new_conversation[-1]['role'] == current_msg['role']:
                assert current_msg['role'] != 'assistant'
                # Merge with last message by creating a new merged dict
                merged_chat = {
                    'role': current_msg['role'],
                    'content': f"{new_conversation[-1]['content']}\n{current_msg['content']}"
                }
                new_conversation[-1] = merged_chat  # replace last message
            else:
                new_conversation.append(dict(current_msg))  # append a copy

        return new_conversation

    def customize_chat_template(self,
                                conversation: Union[List[Dict[str, str]], List[List[Dict[str, str]]]],
                                tools: Optional[List[Union[Dict, Callable]]] = None,
                                documents: Optional[List[Dict[str, str]]] = None,
                                chat_template: Optional[str] = None,
                                add_generation_prompt: bool = False,
                                continue_final_message: bool = False,
                                tokenize: bool = True,
                                padding = False,
                                truncation: bool = False,
                                max_length: Optional[int] = None,
                                return_tensors = None,
                                return_dict: bool = False,
                                tokenizer_kwargs = None,
                                return_assistant_tokens_mask: bool = False,
                                env_reward_lst: List = None,
                                **kwargs,):
        if return_dict and not tokenize:
            raise ValueError(
                "`return_dict=True` is incompatible with `tokenize=False`, because there is no dict "
                "of tokenizer outputs to return."
            )

        if return_assistant_tokens_mask and not return_dict:
            raise ValueError("`return_assistant_tokens_mask=True` is incompatible with `return_dict=False`")

        if tokenizer_kwargs is None:
            tokenizer_kwargs = {}

        chat_template = self.tokenizer.get_chat_template(chat_template, tools)
        if isinstance(conversation, (list, tuple)) and (
            isinstance(conversation[0], (list, tuple)) or hasattr(conversation[0], "messages")
        ):
            conversations = conversation
            is_batched = True
        else:
            conversations = [conversation]
            is_batched = False

        assert not continue_final_message
        template_kwargs = {**self.tokenizer.special_tokens_map, **kwargs}  # kwargs overwrite special tokens if both are present
        rendered_chat, generation_indices = render_jinja_template(
            conversations=conversations,
            tools=tools,
            documents=documents,
            chat_template=chat_template,
            return_assistant_tokens_mask=return_assistant_tokens_mask,
            continue_final_message=continue_final_message,
            add_generation_prompt=add_generation_prompt,
            **template_kwargs,
        )
        if not is_batched:
            rendered_chat = rendered_chat[0]

        if tokenize:
            out = self.tokenizer(
                rendered_chat,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                add_special_tokens=False,
                return_tensors=return_tensors,
                **tokenizer_kwargs,
            )
            if return_dict:
                if return_assistant_tokens_mask:
                    assistant_masks = []
                    assistant_reward_lst = []
                    turn_reward_idx = 0
                    if is_batched or return_tensors:
                        input_ids = out["input_ids"]
                    else:
                        input_ids = [out["input_ids"]]
                    for i in range(len(input_ids)):
                        if self.config.multi_turn.post_process.assistant_loss_mask:
                            current_mask = [0] * len(input_ids[i])
                        else:
                            current_mask = [1] * len(input_ids[i])

                        if self.config.multi_turn.post_process.process_reward:
                            current_reward = [0] * len(input_ids[i])
                        else:
                            current_reward = [0] * len(input_ids[i])
                            current_reward[-1] = sum(env_reward_lst)

                        for assistant_start_char, assistant_end_char in generation_indices[i]:
                            start_token = out.char_to_token(i, assistant_start_char)
                            end_token = out.char_to_token(i, assistant_end_char - 1)
                            if start_token is None:
                                # start_token is out of bounds maybe due to truncation.
                                break

                            if self.config.multi_turn.post_process.assistant_loss_mask:
                                for token_id in range(start_token + 3, end_token - 1 if end_token else len(input_ids[i])):
                                    current_mask[token_id] = 1
                                    # here we only consider the eos, leaving '\n' out for stable training, as well as initial three tokens [bos] assistant \n
                                    # Grad NaN issue: https://github.com/0russwest0/Agent-R1/issues/30

                            # final position
                            if self.config.multi_turn.post_process.process_reward:
                                final_index = end_token - 2 if end_token else len(input_ids[i])-1
                                current_reward[final_index] = env_reward_lst[turn_reward_idx]
                                turn_reward_idx += 1

                        assistant_masks.append(current_mask)
                        assistant_reward_lst.append(current_reward)

                    if not is_batched and not return_tensors:
                        assistant_masks = assistant_masks[0]
                        assistant_reward_lst = assistant_reward_lst[0]

                    out["assistant_masks"] = assistant_masks
                    out["assistant_reward"] = assistant_reward_lst

                    # check all credits is assigned
                    if len(env_reward_lst) != turn_reward_idx:
                        import pdb
                        pdb.set_trace()
                    assert len(env_reward_lst) == turn_reward_idx, print(f"env reward list number {len(env_reward_lst)} does not align with reward turn counts {turn_reward_idx}")
                    # assert (np.array(assistant_masks)*np.array(assistant_reward_lst)).sum().item() == sum(env_reward_lst), print(f"masked reward does not align with raw env reward")
                    assert abs((np.array(assistant_masks)*np.array(assistant_reward_lst)).sum().item()-sum(env_reward_lst)) < 1e-5, print(f"masked reward does not align with raw env reward")

                    if return_tensors:
                        out.convert_to_tensors(tensor_type=return_tensors)

                return out
            else:
                return out["input_ids"]
        else:
            return rendered_chat