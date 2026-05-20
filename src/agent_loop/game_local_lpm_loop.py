import time
import numpy as np
import asyncio
import json
import logging
import os
import re
import random
from typing import Any
from uuid import uuid4
import importlib
from tqdm import tqdm
import torch

class SerializableError(Exception):
    """A serializable exception that can be passed across process boundaries"""
    def __init__(self, message: str, original_exception_type: str = None):
        super().__init__(message)
        self.original_exception_type = original_exception_type

from omegaconf import OmegaConf
from transformers import AutoTokenizer
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register, _DummyConfig, AsyncLLMServerManager
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

from tasks.classic_games.chess.prompt import OSS_SIMPLEST_PLAN_PROMPT_SET as OSS_FEN_PLAN_ACT_PROMPT_SET


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))



def get_async_env(env_config):
    if env_config.env_name=="ChessPuzzles":
        if env_config.game_mode == "puzzles":
            from tasks.classic_games.chess.chess_puzzle_env import AsyncChessPuzzleEnv_Wrapper
            return AsyncChessPuzzleEnv_Wrapper(env_config=env_config)
        elif env_config.game_mode == "full":
            from tasks.classic_games.chess.chess_env import AsyncChessEnv_Wrapper
            return AsyncChessEnv_Wrapper(env_config=env_config)
        else:
            raise ValueError(f"Game mode {env_config.game_mode} is not supported.")
    elif env_config.env_name=="FrozenLake":
        from tasks.classic_games.frozen_lake.frozenlake import AsyncFrozenLakeEnv_Wrapper
        return AsyncFrozenLakeEnv_Wrapper(env_config=env_config)
    else:
        raise ValueError(f"Environment {env_config.env_name} is not supported.")




@register("game_local_lpm_agent")
class GameLocalLPM_AgentLoop(AgentLoopBase):
    """Agent loop with custom SearchClient for search API execution"""

    def __init__(self, trainer_config, server_manager, tokenizer, **kwargs):
        """Initialize the agent loop instance."""
        super().__init__(trainer_config, server_manager, tokenizer, **kwargs)
        # Initialize instance variable for tracking planned action validity
        self.planned_action_valid_records = []

    @classmethod
    def init_class(cls, config, tokenizer, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level SearchAgentLoop initialization")

        # Initialize tokenizer and config
        cls.tokenizer = tokenizer

        # Game/environment configuration and tags
        env_config = config.actor_rollout_ref.rollout.multi_turn.envs.env_config
        cls.env_config = env_config
        cls.max_steps = env_config.max_steps

        cls.max_plan_traj_n = env_config.rollout.max_plan_traj_n
        cls.max_plan_horizon = env_config.rollout.max_plan_horizon

        # action processor
        module_path, class_name = env_config.action_processor.rsplit(".", 1)
        module = importlib.import_module(module_path)
        cls.extract_answer = staticmethod(getattr(module, class_name))

        cls.each_response_length = config.actor_rollout_ref.rollout.each_response_length
        cls.summary_response_length = config.actor_rollout_ref.rollout.multi_turn.summary_response_length

        # prompt set
        prompt_set_path = env_config.prompt_set
        module_path, class_name = prompt_set_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        cls.prompt_set = getattr(module, class_name)

        cls.conversation_prefix = env_config.rollout.conversation_prefix


        # Query/answer/action detection patterns (configurable)
        action_start_tag = '<answer>'
        action_end_tag =  '</answer>'
        plan_action_start_tag = '<move>'
        plan_action_end_tag = '</move>'
        reset_start_tag = "<reset>"
        reset_end_tag = "</reset>"
        summary_start_tag = "<end>"
        summary_end_tag = "</end>"
        answer_start_tag = '<answer>'
        answer_end_tag = '</answer>'

        cls.special_token_list = ["<think>", "</think>", "<move>", "</move>", "<|im_start|>", "<|im_end|>", "+", "#"]

        cls.action_pattern = re.compile(f'{re.escape(action_start_tag)}(.*?){re.escape(action_end_tag)}', re.DOTALL)
        cls.plan_action_pattern = re.compile(f'{re.escape(plan_action_start_tag)}(.*?){re.escape(plan_action_end_tag)}', re.DOTALL)
        cls.reset_pattern = re.compile(f'{re.escape(reset_start_tag)}(.*?){re.escape(reset_end_tag)}', re.DOTALL)
        cls.summary_pattern = re.compile(f'{re.escape(summary_start_tag)}(.*?){re.escape(summary_end_tag)}', re.DOTALL)
        cls.answer_pattern = re.compile(f'{re.escape(answer_start_tag)}(.*?){re.escape(answer_end_tag)}', re.DOTALL)

        # planning config
        cls.early_stop_answer = config.actor_rollout_ref.rollout.multi_turn.get("early_stop_answer", False)   # whether to early stop when answer is detected during planning

        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        try:
            cls.system_prompt = tokenizer.apply_chat_template([{}], add_generation_prompt=False, tokenize=True)
        except Exception as e:
            cls.system_prompt = tokenizer.apply_chat_template([{"role": "system"}], add_generation_prompt=False, tokenize=True)


        cls.tool_masking = config.actor_rollout_ref.rollout.multi_turn.post_process.tool_masking

        cls.intermediate_instruction = config.actor_rollout_ref.rollout.multi_turn.get("intermediate_instruction", False)
        cls.random_move_when_invalid_plan = config.actor_rollout_ref.rollout.multi_turn.get("random_move_when_invalid_plan", False) # whether to switch to random move when invalid plan move is detected
        cls.random_move_when_invalid_act = config.actor_rollout_ref.rollout.multi_turn.random_move_when_invalid_act  # whether to switch to random act when invalid act is detected

        cls.independent_summary = config.actor_rollout_ref.rollout.multi_turn.get("independent_summary", False)  # if true, do summary independently,
        cls.include_simulation_in_summary = config.actor_rollout_ref.rollout.multi_turn.include_simulation_in_summary  # if true, include simulation history in summary
        cls.predefined_instruction = cls.prompt_set['system_prompt']
        cls.predefined_query = cls.prompt_set['plan_query_prompt']
        cls.act_action_format_penality = float(config.actor_rollout_ref.rollout.multi_turn.envs.env_config.act_action_format_penality)
        cls.plan_action_format_penality = float(config.actor_rollout_ref.rollout.multi_turn.envs.env_config.plan_action_format_penality)
        cls.plan_action_reward_scale = float(config.actor_rollout_ref.rollout.multi_turn.envs.env_config.plan_action_reward_scale)
        cls.act_reward_scale = float(config.actor_rollout_ref.rollout.multi_turn.envs.env_config.act_reward_scale)

        cls.use_stockfish_move_matching_reward = config.actor_rollout_ref.rollout.multi_turn.envs.env_config.use_stockfish_move_matching_reward
        cls.stockfish_move_matching_reward = float(config.actor_rollout_ref.rollout.multi_turn.envs.env_config.stockfish_move_matching_reward)
        cls.stockfish_move_topk = int(config.actor_rollout_ref.rollout.multi_turn.envs.env_config.stockfish_move_topk)

        cls.opponent_mode = config.actor_rollout_ref.rollout.multi_turn.envs.env_config.opponent_mode
        if cls.opponent_mode is not None:
            assert cls.opponent_mode in ["random", "fixed", "stockfish"]

        cls.final_reward_readjustment = config.actor_rollout_ref.rollout.multi_turn.final_reward_readjustment

        cls.process_sft_dataset = config.actor_rollout_ref.rollout.multi_turn.post_process.sft.process_sft_dataset
        if cls.process_sft_dataset:
            # sft prompt set
            prompt_set_path = config.actor_rollout_ref.rollout.multi_turn.post_process.sft.sft_prompt_set
            module_path, class_name = prompt_set_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            cls.sft_prompt_set = getattr(module, class_name)
        else:
            cls.sft_prompt_set = None

        cls.sft_dataset_type = config.actor_rollout_ref.rollout.multi_turn.post_process.sft.sft_dataset_type

        cls._env_initialized = False

        cls.ablation = config.actor_rollout_ref.rollout.multi_turn.get("ablation", None)
        cls.ablation_stockfish_move = config.actor_rollout_ref.rollout.multi_turn.get("ablation_stockfish_move", False)

    async def build_env(self, seed):
        # Initialize a single environment instance for this agent loop

        self.env = await get_async_env(self.env_config)

        init_state = await self.env.reset(specified_game_idx=seed)

        self.available_actions = getattr(self.env, 'legal_moves_list', [])  # all the available actions
        self._env_initialized = True

        _query = self.prompt_set['act_query_prompt'].format(
            state=init_state,
            available_move=self.env.legal_moves_string,
            turn_idx=0,
            turn_left=self.max_steps)

        return _query

    @rollout_trace_op
    async def run(self, messages: list[dict[str, Any]], sampling_params: dict[str, Any], seed: int) -> AgentLoopOutput:
        if self.ablation == "summary":
            return  await self.run_ablation_summary(messages, sampling_params, seed)
        else:
            # Default to sequential interaction with environment
            return await self.run_lpm(messages, sampling_params, seed)

    @rollout_trace_op
    async def run_lpm(self, messages: list[dict[str, Any]], sampling_params: dict[str, Any], seed: int) -> AgentLoopOutput:
        metrics = {}
        conversation_id = uuid4().hex
        worker_start_time = time.time()
        sampling_params['max_tokens'] = self.each_response_length

        system_prompt = self.prompt_set['system_prompt']

        # reset game and add user content
        initial_query = await self.build_env(seed)

        instruction_message = [
            {"role": "system", "content": system_prompt},
        ]
        turn_message = [[
            {"role": self.conversation_prefix, "content": initial_query}
        ]]

        # Initial prompt encoding
        prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                [instruction_message[0], turn_message[0][0]], add_generation_prompt=False, tokenize=True
            ),
        )
        response_mask = []
        reward_lst = []
        state_summary_pair = []   # (state, available_actions, summary), for SFT Loss

        user_turns, assistant_turns = 0, 0
        early_stop_flag = 0     # if not stop at agent response

        step = 0
        env_success=False
        env_end = False
        env_early_stop = False
        turn_valid_records = []
        plan_early_stop_lst = []
        plan_num_turns_lst = []

        while True:

            # start planning
            original_planning_stats_dict = self.env.get_key_stats()
            simulator = await get_async_env(self.env_config)
            _ = await simulator.reset_state(original_planning_stats_dict)

            available_moves = simulator.legal_moves_string

            current_plan_message = [instruction_message[0], turn_message[-1][0]]    # the system prompt and the current state query

            (
                planned_action, 
                updated_plan_message, 
                plan_ids,
                plan_response_mask,
                planning_metrics,
                metrics,
                plan_reward_lst,
                summary_text,
                plan_early_stop,
                num_plan_turns
                ) = await self.run_independent_planning(simulator, current_plan_message, sampling_params, metrics)

            prompt_ids += plan_ids
            response_mask += plan_response_mask
            # reward_lst += ([0.] * len(plan_response_mask))
            assert len(plan_reward_lst) == len(plan_response_mask), print(f"mismatch in plan_reward_lst and plan_response_mask: {len(plan_reward_lst)} != {len(plan_response_mask)}")
            reward_lst += plan_reward_lst
            plan_early_stop_lst.append(int(plan_early_stop))
            plan_num_turns_lst.append(num_plan_turns)

            turn_message[-1].extend(updated_plan_message)
            turn_valid_records.extend(planning_metrics['planning_valid_action'])

            state_summary_pair.append(
                {"state": original_planning_stats_dict['FEN'], 
                "available_moves": available_moves,
                "summary": summary_text
                }
                )

            # process the planned action
            planned_action, planned_action_valid, action_feedback = self._extract_action_from_assistant(planned_action)
            
            # Record the planned action validity for tracking
            self.planned_action_valid_records.append(planned_action_valid)

            # handle None action before env step
            if planned_action is None:     # this mean no valid action and we dont support auto random move
                done = True
                env_early_stop = True
                end_info = action_feedback + "\n" + "The game ends. You lose the game because no valid action is detected."
                env_message = [{"role": self.conversation_prefix, "content": end_info}]
                turn_message.append(env_message)

                final_env_feedback_ids = await self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.apply_chat_template(
                        env_message, add_generation_prompt=False, tokenize=True
                    )
                )
                final_env_feedback_ids = final_env_feedback_ids[len(self.system_prompt):]

                invalid_action_reward = -self.act_action_format_penality
                invalid_action_reward -= self.act_reward_scale     # also penalize for losing
                prompt_ids += final_env_feedback_ids
                response_mask += [0] * len(final_env_feedback_ids)

                if self.final_reward_readjustment:
                    reward_lst[-1] += invalid_action_reward
                    reward_lst += [0.] * len(final_env_feedback_ids)
                else:
                    reward_lst += [0.] * len(final_env_feedback_ids)
                    reward_lst[-1] += invalid_action_reward

                break


            move_matching_flag = False
            if self.use_stockfish_move_matching_reward:
                if planned_action_valid:
                    stockfish_proposed_move_list = await self.env.analyse_position(num_moves=self.stockfish_move_topk)
                    if planned_action in stockfish_proposed_move_list:
                        move_matching_flag = True

            # Step the environment with the extracted action
            with simple_timer("env_step", metrics):
                # if self.opponent_mode == "stockfish":
                next_state, reward, env_done, env_info = await self.env.step(planned_action)
                # else:
                #     next_state, reward, done, env_info = self.env.step(planned_action)

                reward = reward * self.act_reward_scale
            
            if not planned_action_valid:
                reward -= self.act_action_format_penality
            
            if self.use_stockfish_move_matching_reward and move_matching_flag:
                reward += self.stockfish_move_matching_reward

            # reward_lst[-1] += reward

            step += 1
            if env_done:
                env_end = True
                done = True
            else:
                if step > self.max_steps:
                    done = True
                else:
                    done = env_done

            if done:
                game_status = self.env.check_results()
                env_success = (game_status == "win")
                env_lose = (game_status == "lose")
                env_tie = (game_status == "tie")
                if env_success:
                    end_info = self.prompt_set['success_prompt'].format(state=next_state)
                elif env_lose:
                    end_info = self.prompt_set['fail_prompt'].format(state=next_state, fail_reason="")
                elif env_tie:
                    end_info = self.prompt_set['tie_prompt'].format(state=next_state)       # TODO: should we add a tie reason? since it can be maximum steps that cause stop

                env_message = [{"role": self.conversation_prefix, "content": end_info}]
                turn_message.append(env_message)

                # process the final env feedback
                final_env_feedback_ids = await self.loop.run_in_executor(
                        None,
                        lambda: self.tokenizer.apply_chat_template(
                            env_message, add_generation_prompt=False, tokenize=True
                        )
                    )
                final_env_feedback_ids = final_env_feedback_ids[len(self.system_prompt):]
                prompt_ids += final_env_feedback_ids
                response_mask += [0] * len(final_env_feedback_ids)
                if self.final_reward_readjustment:  # first add reward to the end token of action response, then add padding
                    reward_lst[-1] += reward
                    reward_lst += [0.] * len(final_env_feedback_ids)
                else:
                    # first add padding, then add reward to the end token of env feedback
                    reward_lst += [0.] * len(final_env_feedback_ids)
                    reward_lst[-1] += reward
                
                break

            
            reward_lst[-1] += reward
            # Update internal state and available actions
            self.state = next_state
            self.available_actions = getattr(self.env, 'legal_moves_list', [])      # all the available actions

            # Build feedback message to the agent
            env_feedback = self._format_env_feedback(next_state, self.env.legal_moves_string, step, env_info)
            env_message = [{"role": "user", "content": action_feedback + "\n" + env_feedback}]

            env_feedback_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    env_message, add_generation_prompt=False, tokenize=True
                )
            )
            env_feedback_ids = env_feedback_ids[len(self.system_prompt):]
            prompt_ids += env_feedback_ids
            response_mask += [0] * len(env_feedback_ids)
            reward_lst += [0.] * len(env_feedback_ids)

            turn_message.append(env_message)

            user_turns += 1

        response_ids = prompt_ids[-len(response_mask):]
        prompt_ids = prompt_ids[:len(prompt_ids) - len(response_mask)]

        metrics['others'] = {
            "env_early_stop": int(env_early_stop),
            "plan_early_stop": np.mean(plan_early_stop_lst),
            "plan_num_turns": np.mean(plan_num_turns_lst),
            "env_end": int(env_end),
            "overlong": int(len(response_ids) > self.response_length),
            "total_length": sum(response_mask),
            "success_rate": env_success,
            "action_valid_rate": np.mean(self.planned_action_valid_records) if len(self.planned_action_valid_records) > 0 else 0.0,
            "planning_valid_action_rate": np.mean(turn_valid_records) if len(turn_valid_records) > 0 else 0.0,
        }
        worker_end_time = time.time()
        worker_duration = worker_end_time - worker_start_time
        metrics['worker_run_time'] = worker_duration

        # process state_summary_pair for SFT Loss
        if self.process_sft_dataset: # and env_success:   # only keep the successful state_summary_pair
            if self.sft_dataset_type == "multiturn":
                sft_dataset = await self._process_multiturn_sft_dataset(instruction_message, turn_message, env_success)
            elif self.sft_dataset_type == "singleturn":
                sft_dataset = await self._process_sft_datasetV2(state_summary_pair, env_success)
        else:
            sft_dataset=[]

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[:self.response_length],
            response_mask=response_mask[:self.response_length],
            reward_lst=reward_lst[:self.response_length],
            num_turns=step + 1,
            metrics=metrics,
            sft_dataset=sft_dataset,
        )
        return output


    async def run_independent_planning(self, simulator, state_message, sampling_params, metrics):
        """
        run independent planning for the given simulator and state message
        need to return:
        1. a final planned action
        2. a updated plan message
        3. planning metrics
        """

        plan_step = 1
        plan_turn = 1
        plan_request_id = uuid4().hex
        
        # Initialize planning metrics tracking
        # This tracks planning behavior across different states and turns
        # - planning_turns_per_state: number of planning turns for each state
        # - planning_steps_per_state: number of planning steps for each state  
        # - All averages, std devs, and medians are calculated from these lists
        planning_metrics = {
            'total_planning_turns': 0,
            'total_planning_steps': 0,
            'planning_turns_per_state': [],
            'planning_steps_per_state': [],
            'planning_turn_details': [],  # Store detailed info for each planning turn
            'max_planning_steps_in_turn': 0,
            'min_planning_steps_in_turn': float('inf'),
            'early_terminations': 0,
            'planning_efficiency': 0.0,  # Will be calculated as steps per successful turn
            'avg_planning_turns_per_state': 0.0,
            'avg_planning_steps_per_state': 0.0,
            'avg_planning_steps_per_turn': 0.0,
            'early_termination_rate': 0.0,
            'std_planning_steps_per_state': 0.0,
            'std_planning_turns_per_state': 0.0,
            'median_planning_steps_per_state': 0.0,
            'median_planning_turns_per_state': 0.0,
            'planning_valid_action': [],
        }

        root_planning_state = simulator.observe()
        root_planning_stats_dict = simulator.get_key_stats()

        # independent planning prompt
        planning_instruction = self.predefined_instruction
        planning_user_prompt = self.predefined_query.format(
            turn_idx=1,
            step_idx=1,
            state=simulator.observe(),
            available_move=simulator.legal_moves_string,
            turn_left=self.max_plan_traj_n - 1,
            max_step=self.max_plan_horizon,
            extra_info="the simulation starts", # if there is a extra_info placeholder

        )

        plan_message = [{"role": "system", "content": planning_instruction}, {"role":"user", "content": planning_user_prompt}]

        # need to remove the system prompt ids as this is already included in the outer loop
        _query_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    [{"role":"user", "content": planning_user_prompt}], add_generation_prompt=False, tokenize=True
                ),
            )
        _query_ids = _query_ids[len(self.system_prompt):]
        plan_response_mask = [0] * len(_query_ids)
        plan_reward_lst = [0.] * len(_query_ids)

        plan_available_actions = simulator.legal_moves_list     # all the available planning actions
        simulation_history = [[f"State: {simulator.observe()}"]]

        plan_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    plan_message, add_generation_prompt=True, tokenize=True
                ),
            )

        plan_early_stop = False
        idx = 0
        while True:
            idx += 1
            if len(plan_message) > 2:
                last_user_ids = await self.loop.run_in_executor(
                        None,
                        lambda: self.tokenizer.apply_chat_template(
                            [plan_message[-1]], add_generation_prompt=True, tokenize=True
                        )
                    )
                last_user_ids = last_user_ids[len(self.system_prompt):]
                plan_ids += last_user_ids
                plan_response_mask += [0] * len(last_user_ids)
                plan_reward_lst += [0.] * len(last_user_ids)

            with simple_timer("generate_sequences", metrics):
                plan_step_request_id = f"{plan_request_id}_{plan_step}"
                sampling_params['max_tokens'] = self.each_response_length
                response_ids = await self.server_manager.generate(
                    request_id=plan_step_request_id, prompt_ids=plan_ids, sampling_params=sampling_params
                )
                plan_ids += response_ids
                plan_response_mask += [1] * len(response_ids)
                plan_reward_lst += [0.] * len(response_ids)

            # ====================== Parse Decision ====================================================#
            response_text = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(response_ids, skip_special_tokens=True)
            )
            plan_message.append({"role": "assistant", "content": response_text})

            (plan_action_text, 
             plan_action_valid, 
             feedback, 
             (jump_to_continue, jump_to_reset, jump_to_summarize, jump_to_end)
             ) = self._extract_plan_from_assistant(
                response_text, 
                plan_available_actions,
                current_planning_turn=plan_turn,
                current_planning_step=plan_step,
                )

            if jump_to_end: # early stop answer
                plan_early_stop = True
                # Record metrics for early termination
                planning_metrics['early_terminations'] += 1
                planning_metrics['planning_turn_details'].append({
                    'turn': plan_turn,
                    'steps': plan_step,
                    'state': root_planning_state,
                    'early_termination': True
                })
                planning_metrics['planning_turns_per_state'].append(plan_turn)
                planning_metrics['planning_steps_per_state'].append(plan_step)
                planning_metrics['total_planning_steps'] += plan_step
                planning_metrics['max_planning_steps_in_turn'] = max(planning_metrics['max_planning_steps_in_turn'], plan_step)
                planning_metrics['min_planning_steps_in_turn'] = min(planning_metrics['min_planning_steps_in_turn'], plan_step)
                
                plan_ids = plan_ids[-len(plan_response_mask):]
                
                return plan_action_text, plan_message[1:], plan_ids, plan_response_mask, planning_metrics, metrics, plan_reward_lst, response_text, plan_early_stop, idx

            # append env feedback if there is any
            if len(feedback) > 0:
                plan_message.append({"role": "user", "content": feedback})

            if jump_to_summarize:   # break the planning loop and jump to summarize phase
                # Record metrics for the final planning turn
                planning_metrics['planning_turn_details'].append({
                    'turn': plan_turn,
                    'steps': plan_step,
                    'state': root_planning_state
                })
                planning_metrics['planning_turns_per_state'].append(plan_turn)
                planning_metrics['planning_steps_per_state'].append(plan_step)
                planning_metrics['total_planning_steps'] += plan_step
                planning_metrics['max_planning_steps_in_turn'] = max(planning_metrics['max_planning_steps_in_turn'], plan_step)
                planning_metrics['min_planning_steps_in_turn'] = min(planning_metrics['min_planning_steps_in_turn'], plan_step)
                break

            if plan_action_text is not None:
                if plan_action_valid is not None:
                    planning_metrics['planning_valid_action'].append(int(plan_action_valid))
                # env step if action is detected
                # if self.opponent_mode == "stockfish":
                next_state, reward, done, env_info = await simulator.step(plan_action_text)
                # else:   
                #     next_state, reward, done, env_info = simulator.step(plan_action_text)

                plan_step += 1
                plan_reward_lst[-1] += (reward * self.plan_action_reward_scale - self.plan_action_format_penality)

                simulation_history[-1].append(f"Move: {plan_action_text}")
                if not done:
                    simulation_history[-1].append(f"Opponent's move: {env_info['oppo_move']}")
                simulation_history[-1].append(f"Reward: {reward}")
                simulation_history[-1].append(f"State: {next_state}")
                simulation_history[-1].append(f"Game Terminated: {done}")

                # check planning termination
                if done or (plan_step > self.max_plan_horizon):
                    # process termination feedback
                    if done:
                        game_status = simulator.check_results()
                        simulation_history[-1].append(f"Result: {game_status}")
                        env_success = (game_status == "win")
                        env_lose = (game_status == "lose")
                        env_tie = (game_status == "tie")
                        if env_success:
                            end_info = self.prompt_set['plan_success_prompt'].format(state=next_state)
                        elif env_lose:
                            end_info = self.prompt_set['plan_fail_prompt'].format(state=next_state, fail_reason="because you lose.", extra_info="")
                        elif env_tie:
                            end_info = self.prompt_set['plan_tie_prompt'].format(state=next_state)

                    else:   # reaching max planning budget
                        end_info = f"The opponent played the move {env_info['oppo_move']}.\n" + self.prompt_set['plan_fail_prompt'].format(
                            state=next_state, fail_reason="because reaching maximum planning steps ahead", extra_info="")

                    if plan_message[-1]['role'] == 'user':
                        plan_message[-1]['content'] += f"\n{end_info}"
                    else:
                        plan_message.append({"role": "user", "content": end_info})

                    if plan_turn == self.max_plan_traj_n:
                        jump_to_summarize = True
                        # when ready to break the planning loop, append user query to plan_ids
                        env_feedback_ids = await self.loop.run_in_executor(
                            None,
                            lambda: self.tokenizer.apply_chat_template(
                                [plan_message[-1]], add_generation_prompt=False, tokenize=True
                            )
                        )
                        env_feedback_ids = env_feedback_ids[len(self.system_prompt):]
                        plan_ids += env_feedback_ids
                        plan_response_mask += [0] * len(env_feedback_ids)
                        plan_reward_lst += [0.] * len(env_feedback_ids)

                        break
                    else:
                        jump_to_reset = True
                else:
                    env_feedback = self._format_plan_env_feedback(next_state, simulator.legal_moves_string, plan_turn, plan_step, env_info)
                    if plan_message[-1]['role'] == 'user':
                        plan_message[-1]['content'] += f"\n{env_feedback}"
                    else:
                        plan_message.append({"role": "user", "content": env_feedback})

            if jump_to_reset:     # reset the planning state
                # Record metrics for the completed planning turn
                planning_metrics['planning_turn_details'].append({
                    'turn': plan_turn,
                    'steps': plan_step,
                    'state': root_planning_state
                })
                planning_metrics['planning_turns_per_state'].append(plan_turn + 1)  # +1 because turn is 0-indexed
                planning_metrics['planning_steps_per_state'].append(plan_step)
                planning_metrics['total_planning_steps'] += plan_step
                planning_metrics['max_planning_steps_in_turn'] = max(planning_metrics['max_planning_steps_in_turn'], plan_step)
                planning_metrics['min_planning_steps_in_turn'] = min(planning_metrics['min_planning_steps_in_turn'], plan_step)
                
                simulator.reset_state(root_planning_stats_dict)
                plan_turn += 1
                plan_step = 1

                new_plan_prompt = self.predefined_query.format(
                    turn_idx=plan_turn,
                    step_idx=plan_step,
                    state=simulator.observe(),
                    available_move=simulator.legal_moves_string,
                    turn_left=self.max_plan_traj_n - plan_turn,
                    max_step=self.max_plan_horizon - plan_step + 1,
                    extra_info="the simulation has been restarted to original game state"
                )
                if plan_message[-1]['role'] == 'user':
                    plan_message[-1]['content'] += f"\n{new_plan_prompt}"
                else:
                    plan_message.append({"role": "user", "content": new_plan_prompt})   # might have overlap with previous env feedback in plan_ids

                simulation_history.append([f"State: {simulator.observe()}"])

            # update available actions
            plan_available_actions = simulator.legal_moves_list     # all the available planning actions

        # ready to summarize
        assert jump_to_summarize, print(f"jump_to_summarize = {jump_to_summarize}") # should be true

        if not self.independent_summary:
            if self.include_simulation_in_summary:
                history_string = ""
                for i, lst in enumerate(simulation_history):
                    _traj = ", ".join(lst)
                    _traj = f"{i+1}: " + _traj
                    history_string += _traj + "\n"
                plan_summarize_prompt = self.prompt_set['independent_summary_prompt'].format(
                    state=root_planning_state,
                    history=history_string
                )
            else:
                plan_summarize_prompt = self.prompt_set['plan_action_query_prompt'].format(
                    state=root_planning_state
                )

            plan_message.append({"role": "user", "content": plan_summarize_prompt})


            _plan_summarize_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": plan_summarize_prompt}], add_generation_prompt=True, tokenize=True
                ),
            )
            _plan_summarize_ids = _plan_summarize_ids[len(self.system_prompt):]
            plan_ids += _plan_summarize_ids
            plan_response_mask += [0] * len(_plan_summarize_ids)
            plan_reward_lst += [0.] * len(_plan_summarize_ids)
            
            with simple_timer("generate_sequences", metrics):
                sampling_params['max_tokens'] = self.summary_response_length
                response_ids = await self.server_manager.generate(
                    request_id=uuid4().hex, prompt_ids=plan_ids, sampling_params=sampling_params
                )
                plan_ids += response_ids
                plan_response_mask += [1] * len(response_ids)
                plan_reward_lst += [0.] * len(response_ids)

            summary_text = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.decode(response_ids, skip_special_tokens=True)
            )
            
        else:   # independent summary
            # process the simulation history
            history_string = ""
            for i, lst in enumerate(simulation_history):
                _traj = ", ".join(lst)
                _traj = f"{i+1}: " + _traj
                history_string += _traj + "\n"
            plan_summarize_prompt = self.prompt_set['independent_summary_prompt'].format(
                state=root_planning_state,
                history=history_string
            )
            plan_message.append({"role": "user", "content": plan_summarize_prompt})

            # _input_message = [raw_message[0], {"role": "user", "content": plan_summarize_prompt}]
            _input_message = [{"role": "system", "content": self.prompt_set['independent_summary_instruction']}, {"role": "user", "content": plan_summarize_prompt}]
            _plan_summarize_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    _input_message, add_generation_prompt=True, tokenize=True
                ),
            )

            with simple_timer("generate_sequences", metrics):
                sampling_params['max_tokens'] = self.summary_response_length
                response_ids = await self.server_manager.generate(
                    request_id=uuid4().hex, prompt_ids=_plan_summarize_ids, sampling_params=sampling_params
                )
            summary_query_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": plan_summarize_prompt}], add_generation_prompt=True, tokenize=True
                ),
            )
            summary_query_ids = summary_query_ids[len(self.system_prompt):]

            plan_ids += summary_query_ids
            plan_response_mask += [0] * len(summary_query_ids)
            plan_reward_lst += [0.] * len(summary_query_ids)
            plan_ids += response_ids
            plan_response_mask += [1] * len(response_ids)
            plan_reward_lst += [0.] * len(response_ids)

            summary_text = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.decode(response_ids, skip_special_tokens=True)
            )


        response_text = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(response_ids, skip_special_tokens=True)
        )
        plan_message.append({"role": "assistant", "content": response_text})

        (final_action, final_action_valid) = self._extract_summarize_from_assistant(response_text)

        # Calculate final planning metrics
        planning_metrics['total_planning_turns'] = len(planning_metrics['planning_turn_details'])
        planning_metrics['avg_planning_steps_per_turn'] = (
            planning_metrics['total_planning_steps'] / planning_metrics['total_planning_turns'] 
            if planning_metrics['total_planning_turns'] > 0 else 0
        )
        planning_metrics['avg_planning_steps_per_state'] = (
            np.mean(planning_metrics['planning_steps_per_state']) 
            if planning_metrics['planning_steps_per_state'] else 0
        )
        planning_metrics['avg_planning_turns_per_state'] = (
            np.mean(planning_metrics['planning_turns_per_state']) 
            if planning_metrics['planning_turns_per_state'] else 0
        )
        # Calculate standard deviations and medians
        planning_metrics['std_planning_steps_per_state'] = (
            np.std(planning_metrics['planning_steps_per_state']) 
            if planning_metrics['planning_steps_per_state'] else 0
        )
        planning_metrics['std_planning_turns_per_state'] = (
            np.std(planning_metrics['planning_turns_per_state']) 
            if planning_metrics['planning_turns_per_state'] else 0
        )
        planning_metrics['median_planning_steps_per_state'] = (
            np.median(planning_metrics['planning_steps_per_state']) 
            if planning_metrics['planning_steps_per_state'] else 0
        )
        planning_metrics['median_planning_turns_per_state'] = (
            np.median(planning_metrics['planning_turns_per_state']) 
            if planning_metrics['planning_turns_per_state'] else 0
        )
        # Handle edge case for min_planning_steps_in_turn
        if planning_metrics['min_planning_steps_in_turn'] == float('inf'):
            planning_metrics['min_planning_steps_in_turn'] = 0
            
        # Calculate planning efficiency (steps per successful turn, excluding early terminations)
        successful_turns = planning_metrics['total_planning_turns'] - planning_metrics['early_terminations']
        planning_metrics['planning_efficiency'] = (
            planning_metrics['total_planning_steps'] / successful_turns 
            if successful_turns > 0 else 0
        )
        
        # Add early termination rate
        planning_metrics['early_termination_rate'] = (
            planning_metrics['early_terminations'] / planning_metrics['total_planning_turns'] 
            if planning_metrics['total_planning_turns'] > 0 else 0
        )

        plan_ids = plan_ids[-len(plan_response_mask):]

        return final_action, plan_message[1:], plan_ids, plan_response_mask, planning_metrics, metrics, plan_reward_lst, summary_text, plan_early_stop, idx


    def _extract_plan_from_assistant(self, response: str, plan_available_actions: list, current_planning_turn, current_planning_step):
        """
        extract plan action tag and answer tag, then handle early stop case, 
        if answer tag is detected, return final action and jump to end
        if plan action tag is not detected, end current planning turn and decide to reset of summrize, depeding on current planning turn
        if plan action tag is detected, return plan action and jump to continue
        """

        jump_to_continue = False
        jump_to_reset = False
        jump_to_summarize = False
        jump_to_end = False

        plan_match = self.plan_action_pattern.search(response)      # ready to plan
        answer_match = self.answer_pattern.search(response)        # ready to answer
        reset_match = self.reset_pattern.search(response)           # ready to reset
        summary_match = self.summary_pattern.search(response)       # ready to summarize

        if answer_match and self.early_stop_answer:   # have detected answer tag, and we accept early stop answer, then jump to final answer
            jump_to_end = True
            final_action = answer_match.group(1).strip()
            # return the first action at root planning state
            return final_action, None, "", (jump_to_continue, jump_to_reset, jump_to_summarize, jump_to_end)

        if reset_match:  # have detected reset tag, then jump to reset
            if current_planning_turn == self.max_plan_traj_n:  # if no more planning traj budget left, jump to summerization phase directly
                jump_to_summarize = True
            else:
                jump_to_reset = True
            return None, None, "", (jump_to_continue, jump_to_reset, jump_to_summarize, jump_to_end)
        
        if summary_match:
            jump_to_summarize = True
            return None, None, "", (jump_to_continue, jump_to_reset, jump_to_summarize, jump_to_end)

        # when not answer, reset or end, then check plan move
        if not plan_match:
            if self.random_move_when_invalid_plan:
                random_move = random.choice(plan_available_actions)
                feedback = f"No valid plan move detected. Switch to random move: {random_move}"
                jump_to_continue=True
                return random_move, 0., feedback, (jump_to_continue, jump_to_reset, jump_to_summarize, jump_to_end)
            else:
                # not valid plan move is detected
                feedback = "You did not output a move. End current planning turn."
                if current_planning_turn == self.max_plan_traj_n:
                    jump_to_summarize = True
                else:
                    jump_to_reset = True
                return None, None, feedback, (jump_to_continue, jump_to_reset, jump_to_summarize, jump_to_end)
        else:
            feedback = "Planned move format detected."
            plan_action = plan_match.group(1).strip()
            jump_to_continue=True

            lower_letter_plan_available_actions = [i.lower() for i in plan_available_actions]
            for special_token in self.special_token_list:
                plan_action = plan_action.replace(special_token, "").strip()

            if plan_action.lower() in lower_letter_plan_available_actions:
                position_idx = lower_letter_plan_available_actions.index(plan_action.lower())
                feedback += f" The move is valid: {plan_available_actions[position_idx]}"
                return plan_available_actions[position_idx], 1.0, feedback, (jump_to_continue, jump_to_reset, jump_to_summarize, jump_to_end)
            else:
                random_move = random.choice(plan_available_actions)
                feedback += f" The move is invalid, switch to random move: {random_move}"
                return random_move, 0., feedback, (jump_to_continue, jump_to_reset, jump_to_summarize, jump_to_end)

    def _extract_summarize_from_assistant(self, response_text: str):
        """Extract summarize text from assistant messages using <summarize> tags."""

        answer_match = self.answer_pattern.search(response_text)        # ready to answer
        if not answer_match:
            return None, 0.
        else:
            return answer_match.group(1).strip(), 1.0

    def _extract_action_from_assistant(self, final_planned_action: str):
        """Extract action text from assistant messages using <action> tags."""
        # Collect assistant contents from the latest generation
        # print(f"\nllm response = {tmp_template}\n")
        if final_planned_action is None:
            if self.random_move_when_invalid_act:
                random_action = random.choice(self.available_actions)
                return random_action, 0., f"No output action detected. Swith to random move: {random_action}"
            else:
                return None, 0., "No output action detected. End current turn."

        match = self.action_pattern.search(final_planned_action)
        if not match:
            # random_action = random.choice(self.available_actions)
            # return random_action, 0., f"No output action detected. Swith to random move: {random_action}"
            action_content = final_planned_action   # check if action tag is detected
        else:
            action_content = match.group(1).strip()

        lower_letter_available_actions = [i.lower() for i in self.available_actions]

        for special_token in self.special_token_list:
            action_content = action_content.replace(special_token, "").strip()

        if action_content.lower() in lower_letter_available_actions:
            position_idx = lower_letter_available_actions.index(action_content.lower())

            return self.available_actions[position_idx], 1.0, f"Action {self.available_actions[position_idx]} is valid."
        else:
            if self.random_move_when_invalid_act:    # switch to random act when invalid act is detected
                random_action = random.choice(self.available_actions)
                return random_action, 0., f"Action {action_content.lower()} is not valid. Switch to random move: {random_action}"
            else:
                return None, 0., f"Action {action_content.lower()} is invalid. End current turn."

    def _format_env_feedback(self, state: Any, available_actions: list, step_idx, env_info: dict) -> str:
        # try:
        # actions_text = ", ".join(list(available_actions)[:100])
        # except Exception:

        if "oppo_move" in env_info and "act_query_prompt_with_info" in self.prompt_set:
            next_query = self.prompt_set['act_query_prompt_with_info'].format(
                state=state,
                available_move=available_actions,
                turn_idx=step_idx,
                turn_left=self.max_steps-step_idx,
                oppo_move=env_info['oppo_move'])
        else:
            next_query = self.prompt_set['act_query_prompt'].format(
                state=state,
                available_move=available_actions,
                turn_idx=step_idx,
                turn_left=self.max_steps-step_idx)

        return next_query


    def _format_plan_env_feedback(self, state: Any, available_actions: list, plan_turn, plan_step, env_info: dict) -> str:

        if "oppo_move" in env_info and "plan_query_prompt_with_info" in self.prompt_set:
            next_query = self.prompt_set["plan_query_prompt_with_info"].format(
                turn_idx=plan_turn,
                step_idx=plan_step,
                state=state,
                available_move=available_actions,
                turn_left=self.max_plan_traj_n - plan_turn,
                max_step=self.max_plan_horizon - plan_step + 1,
                oppo_move=env_info['oppo_move'],
                extra_info="the simulation continues",
            )
        else:
            next_query = self.prompt_set["plan_query_prompt"].format(
                turn_idx=plan_turn,
                step_idx=plan_step,
                state=state,
                available_move=available_actions,
                turn_left=self.max_plan_traj_n - plan_turn,
                max_step=self.max_plan_horizon - plan_step + 1,
                extra_info="the simulation continues",
            )

        return next_query

    async def _process_sft_dataset(self, state_summary_pair: list[dict]) -> list[dict]:
        """
        Process the state_summary_pair for SFT Loss
        """
        sft_dataset = []

        for state_summary in state_summary_pair:
            _sft_prompt = [
                {"role": "system", "content": self.sft_prompt_set['system_prompt']},
                {"role": "user", "content": self.sft_prompt_set['query_prompt'].format(
                    state=state_summary['state'],
                    available_move=state_summary['available_moves']
                )},
            ]
            _sft_prompt_chat_str = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    _sft_prompt, add_generation_prompt=False, tokenize=False
                )
            )
            _sft_response_chat_str = state_summary['summary'] + self.tokenizer.eos_token

            # tokenize
            prompt_ids_output = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer(_sft_prompt_chat_str, return_tensors="pt", add_special_tokens=False)
            )
            prompt_ids = prompt_ids_output["input_ids"][0]
            prompt_attention_mask = prompt_ids_output["attention_mask"][0]

            response_ids_output = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer(_sft_response_chat_str, return_tensors="pt", add_special_tokens=False)
            )
            response_ids = response_ids_output["input_ids"][0]
            response_attention_mask = response_ids_output["attention_mask"][0]

            sft_dataset.append({
                "prompt_ids": prompt_ids,
                "prompt_attention_mask": prompt_attention_mask,
                "response_ids": response_ids,
                "response_attention_mask": response_attention_mask,
            })

        return sft_dataset

    async def _process_sft_datasetV2(self, state_summary_pair: list[dict], env_success: bool) -> list[dict]:
        """
        Process the state_summary_pair for SFT Loss
        """
        sft_dataset = []
        if not env_success:
            return sft_dataset

        for state_summary in state_summary_pair:
            _sft_prompt = [
                {"role": "user", "content": self.sft_prompt_set['query_prompt'].format(
                    state=state_summary['state'],
                    available_move=state_summary['available_moves']
                )},
            ]
            prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    _sft_prompt, add_generation_prompt=True, tokenize=True, return_tensors="pt",
                )
            )
            prompt_ids = prompt_ids[0][len(self.system_prompt):]
            prompt_attention_mask = torch.ones_like(prompt_ids, dtype=prompt_ids.dtype)

            clean_thinking_summary = state_summary['summary']
            _reasoning_pattern = re.compile(f'{re.escape("<think>")}(.*?){re.escape("</think>")}', re.DOTALL)
            
            # Extract content from <think>...</think> tags
            redacted_reasoning_match = _reasoning_pattern.search(clean_thinking_summary)
            if redacted_reasoning_match:
                extracted_reasoning = redacted_reasoning_match.group(1).strip()
                # Use extracted reasoning as the response
                _sft_response_chat_str = extracted_reasoning
            else:
                continue
                # raise ValueError(f"No <think>...</think> tags found in the summary: {clean_thinking_summary}")
            
            # also remove the <answer>...</answer> tag if there is any
            answer_pattern = re.compile(f'{re.escape("<answer>")}(.*?){re.escape("</answer>")}', re.DOTALL)
            _sft_response_chat_str = answer_pattern.sub('', _sft_response_chat_str).strip()


            # tokenize
            # prompt_ids_output = await self.loop.run_in_executor(
            #     None,
            #     lambda: self.tokenizer(_sft_prompt_chat_str, return_tensors="pt", add_special_tokens=False)
            # )
            # prompt_ids = prompt_ids_output["input_ids"][0]
            # prompt_attention_mask = prompt_ids_output["attention_mask"][0]

            response_ids_output = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer(_sft_response_chat_str, return_tensors="pt", add_special_tokens=False)
            )
            response_ids = response_ids_output["input_ids"][0]
            response_attention_mask = response_ids_output["attention_mask"][0]

            sft_dataset.append({
                "prompt_ids": prompt_ids,
                "prompt_attention_mask": prompt_attention_mask,
                "response_ids": response_ids,
                "response_attention_mask": response_attention_mask,
            })

        return sft_dataset


    async def _process_multiturn_sft_dataset(self, instruction_message, plan_message, env_success):
        """
        instruction_message: [{"role": "system", "content": system_prompt}]
        plan_message: 
        [[{"role": "user", "content": planning_query}, {"role": "assistant", "content": planning_response}, ...],[...], [final_outcome] ]
        
        if env_success, then record the query, summary pair, otherwise, record the full traj
        """

        plan_summary_message = []
        for i in range(len(plan_message)-1):
            if env_success:
                plan_summary_message.append(plan_message[i][0]) # adding the simulator state
                plan_summary_message.append(plan_message[i][-1])    # adding the summary
            else:
                plan_summary_message.extend(plan_message[i]) # adding the simulator state

        final_message = instruction_message + plan_summary_message + plan_message[-1]

        # First, get the full conversation tokens
        full_tokens = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                final_message, add_generation_prompt=False, tokenize=True, return_tensors="pt",
            ),
        )

        concat_tokens = []
        concat_loss_mask = []
        concat_attention_mask = []
        i = 0
        while i < len(final_message):
            cur_message = final_message[i]
            if cur_message["role"] == "assistant":
                tokens, loss_mask, attention_mask = await self._process_message_tokens(
                    final_message, i, i + 1, is_assistant=True
                )
                concat_tokens.extend(tokens)
                concat_loss_mask.extend(loss_mask)
                concat_attention_mask.extend(attention_mask)
                i += 1
            elif cur_message["role"] in ["user", "system"]:
                if cur_message["role"] == "system" and i != 0:
                    raise ValueError("System message should be the first message")
                tokens, loss_mask, attention_mask = await self._process_message_tokens(
                    final_message, i, i + 1, is_assistant=False
                )
                concat_tokens.extend(tokens)
                concat_loss_mask.extend(loss_mask)
                concat_attention_mask.extend(attention_mask)
                i += 1
            else:
                raise ValueError(f"Unknown role: {cur_message['role']}")

        input_ids, loss_mask, attention_mask = self._validate_and_convert_tokens(
            full_tokens[0], concat_tokens, concat_loss_mask, concat_attention_mask
        )

        sft_dataset = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "loss_mask": loss_mask,
            }

        return [sft_dataset]


    async def _process_message_tokens(self, messages: list[dict], start_idx: int, end_idx: int, is_assistant: bool = False) -> tuple[list[int], list[int], list[int]]:

        if start_idx > 0:
            prev_applied_text = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                messages[:start_idx], add_generation_prompt=False, tokenize=False
                )
            )
            if is_assistant:
                prev_applied_text_w_generation_prompt = await self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.apply_chat_template(
                    messages[:start_idx], add_generation_prompt=True, tokenize=False
                    )
                )
        else:
            prev_applied_text = ""

        cur_applied_text = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
            messages[:end_idx], add_generation_prompt=False, tokenize=False
            )
        )

        if is_assistant:
            generation_prompt_text = prev_applied_text_w_generation_prompt[len(prev_applied_text) :]
            generation_prompt_tokens = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.encode(
                generation_prompt_text, add_special_tokens=False
            ))
            _message_tokens = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.encode(
                cur_applied_text[len(prev_applied_text_w_generation_prompt) :], add_special_tokens=False
                )
            )
            message_tokens = generation_prompt_tokens + _message_tokens
            loss_mask = [0] * (len(generation_prompt_tokens)) + [1] * (len(message_tokens) - len(generation_prompt_tokens))
        else:
            message_tokens = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.encode(
                cur_applied_text[len(prev_applied_text) :], add_special_tokens=False
                )
            )
            loss_mask = [0] * len(message_tokens)

        attention_mask = [1] * len(message_tokens)

        return message_tokens, loss_mask, attention_mask

    def _validate_and_convert_tokens(self, full_tokens: torch.Tensor, concat_tokens: list[int], concat_loss_mask: list[int], concat_attention_mask: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Validate tokenization and convert to tensors.
        """
        full_tokens_list = full_tokens.tolist()
        if len(concat_tokens) != len(full_tokens_list) or not all(
            a == b for a, b in zip(concat_tokens, full_tokens_list, strict=True)
        ):
            # raise ValueError("Token mismatch detected!")
            # print(f"Token mismatch detected! {len(concat_tokens)} != {len(full_tokens_list)}")
            return (
                torch.tensor(concat_tokens, dtype=torch.long),
                torch.tensor(concat_loss_mask, dtype=torch.long),
                torch.tensor(concat_attention_mask, dtype=torch.long),
            )
        return (
            full_tokens,
            torch.tensor(concat_loss_mask, dtype=torch.long),
            torch.tensor(concat_attention_mask, dtype=torch.long),
        )

    # ablation exp
    @rollout_trace_op
    async def run_ablation_summary(self, messages: list[dict[str, Any]], sampling_params: dict[str, Any], seed: int) -> AgentLoopOutput:
        """
        Ablation experiment for summary phase.
        The simulation/branching will not be considered as model completion, we will only include planning history and do independent summary. And we update summary to see the effect of summary.
        """

        metrics = {}
        worker_start_time = time.time()
        sampling_params['max_tokens'] = self.each_response_length

        system_prompt = self.prompt_set['system_prompt']

        # reset game and add user content
        initial_query = self.build_env(seed)

        instruction_message = [
            {"role": "system", "content": system_prompt},
        ]
        turn_message = [[
            {"role": self.conversation_prefix, "content": initial_query}
        ]]

        # Initial prompt encoding
        prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                [instruction_message[0], turn_message[0][0]], add_generation_prompt=False, tokenize=True
            ),
        )
        response_mask = []
        reward_lst = []

        user_turns, assistant_turns = 0, 0

        step = 0
        env_success=False
        env_end = False
        turn_valid_records = []
        state_summary_pair = []
        plan_ever_win_lst = []

        while True:

            # start planning
            original_planning_stats_dict = self.env.get_key_stats()
            # if self.opponent_mode == "stockfish":
            simulator = get_async_env(self.env_config)
            # else:
            #     simulator = get_env(self.env_config)
                
            simulator.reset_state(original_planning_stats_dict)

            available_moves = simulator.legal_moves_string

            current_plan_message = [instruction_message[0], turn_message[-1][0]]    # the system prompt and the current state query

            (
                planned_action, 
                updated_plan_message, 
                plan_ids,
                plan_response_mask,
                planning_metrics,
                metrics,
                summary_text,
                ever_win
                ) = await self.run_independent_summary(simulator, current_plan_message, sampling_params, metrics)

            prompt_ids += plan_ids
            response_mask += plan_response_mask
            reward_lst += ([0.] * len(plan_response_mask))

            turn_message[-1].extend(updated_plan_message)
            turn_valid_records.extend(planning_metrics['planning_valid_action'])
            state_summary_pair.append(
                {"state": original_planning_stats_dict['FEN'], 
                "available_moves": available_moves,
                "summary": summary_text
                }
                )
            plan_ever_win_lst.append(ever_win)

            # process the planned action
            planned_action, planned_action_valid, action_feedback = self._extract_action_from_assistant(planned_action)
            
            # Record the planned action validity for tracking
            self.planned_action_valid_records.append(planned_action_valid)
            # print(f"Recorded planned_action_valid = {planned_action_valid} (total records: {len(self.planned_action_valid_records)})")

            if planned_action is None:
                done = True
                end_info = action_feedback + "\n" + "The game ends. You lose the game because no valid action is detected."
                env_message = [{"role": self.conversation_prefix, "content": end_info}]
                turn_message.append(env_message)

                final_env_feedback_ids = await self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.apply_chat_template(
                        env_message, add_generation_prompt=False, tokenize=True
                    )
                )
                final_env_feedback_ids = final_env_feedback_ids[len(self.system_prompt):]
                prompt_ids += final_env_feedback_ids
                response_mask += [0] * len(final_env_feedback_ids)
                invalid_reward = -self.act_action_format_penality
                invalid_reward -= self.act_reward_scale     # also penalize for losing
                if self.final_reward_readjustment:
                    reward_lst[-1] += invalid_reward
                    reward_lst += [0.] * len(final_env_feedback_ids)
                else:
                    reward_lst += [0.] * len(final_env_feedback_ids)
                    reward_lst[-1] += invalid_reward
                break

            move_matching_flag = False
            if self.use_stockfish_move_matching_reward:
                if planned_action_valid:
                    proposed_move_list = await self.env.analyse_position(num_moves=1)
                    if planned_action in proposed_move_list:
                        move_matching_flag = True

            # Step the environment with the extracted action
            with simple_timer("env_step", metrics):
                # if self.opponent_mode == "stockfish":
                next_state, reward, env_done, env_info = await self.env.step(planned_action)
                # else:
                #     next_state, reward, done, env_info = self.env.step(planned_action)

                reward = reward * self.act_reward_scale

            if not planned_action_valid:
                reward -= self.act_action_format_penality
            
            if self.use_stockfish_move_matching_reward and move_matching_flag:
                reward += self.stockfish_move_matching_reward

            # reward_lst[-1] += reward

            step += 1
            if env_done:
                env_end = True
                done = True
            else:
                if step > self.max_steps:
                    done = True
                else:
                    done = env_done

            if done:
                game_status = self.env.check_results()
                env_success = (game_status == "win")
                env_lose = (game_status == "lose")
                env_tie = (game_status == "tie")
                if env_success:
                    end_info = self.prompt_set['success_prompt'].format(state=next_state)
                elif env_lose:
                    end_info = self.prompt_set['fail_prompt'].format(state=next_state, fail_reason="because you lose.")
                elif env_tie:
                    end_info = self.prompt_set['tie_prompt'].format(state=next_state)

                env_message = [{"role": self.conversation_prefix, "content": end_info}]
                final_env_feedback_ids = await self.loop.run_in_executor(
                        None,
                        lambda: self.tokenizer.apply_chat_template(
                            env_message, add_generation_prompt=False, tokenize=True
                        )
                    )
                final_env_feedback_ids = final_env_feedback_ids[len(self.system_prompt):]
                prompt_ids += final_env_feedback_ids
                response_mask += [0] * len(final_env_feedback_ids)
                if self.final_reward_readjustment:
                    reward_lst[-1] += reward
                    reward_lst += [0.] * len(final_env_feedback_ids)
                else:
                    reward_lst += [0.] * len(final_env_feedback_ids)
                    reward_lst[-1] += reward
                break

            
            reward_lst[-1] += reward    # action reward

            # Update internal state and available actions
            self.state = next_state
            self.available_actions = getattr(self.env, 'legal_moves_list', [])      # all the available actions

            # Build feedback message to the agent
            env_feedback = self._format_env_feedback(next_state, self.env.legal_moves_string, step, env_info)
            env_message = [{"role": "user", "content": action_feedback + "\n" + env_feedback}]

            env_feedback_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    env_message, add_generation_prompt=False, tokenize=True
                )
            )
            env_feedback_ids = env_feedback_ids[len(self.system_prompt):]
            prompt_ids += env_feedback_ids
            response_mask += [0] * len(env_feedback_ids)
            reward_lst += [0.] * len(env_feedback_ids)

            turn_message.append(env_message)

            user_turns += 1

        response_ids = prompt_ids[-len(response_mask):]
        prompt_ids = prompt_ids[:len(prompt_ids) - len(response_mask)]

        metrics['others'] = {
            "plan_ever_win_rate": int(sum(plan_ever_win_lst) > 0),
            "overlong": int(len(response_ids) > self.response_length),
            "env_end": int(env_end),
            "total_length": sum(response_mask),
            "success_rate": env_success,
            "action_valid_rate": np.mean(self.planned_action_valid_records) if len(self.planned_action_valid_records) > 0 else 0.0,
            "planning_valid_action_rate": np.mean(turn_valid_records) if len(turn_valid_records) > 0 else 0.0,
        }
        worker_end_time = time.time()
        worker_duration = worker_end_time - worker_start_time
        metrics['worker_run_time'] = worker_duration

        if self.process_sft_dataset: # and env_success:   # only keep the successful state_summary_pair
            if self.sft_dataset_type == "multiturn":
                raise NotImplementedError("Multiturn SFT dataset is not supported for ablation summary")
                sft_dataset = await self._process_multiturn_sft_dataset(instruction_message, turn_message, env_success)
            elif self.sft_dataset_type == "singleturn":
                sft_dataset = await self._process_sft_datasetV2(state_summary_pair, env_success)
        else:
            sft_dataset=[]

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[:self.response_length],
            response_mask=response_mask[:self.response_length],
            reward_lst=reward_lst[:self.response_length],
            num_turns=step + 1,
            metrics=metrics,
            sft_dataset=sft_dataset,
        )
        return output

    async def run_independent_summary(self, simulator, state_message, sampling_params, metrics):
        """
        Run independent planning for the given simulator and state message.
        """
        
        plan_step = 1
        plan_turn = 1
        plan_request_id = uuid4().hex
        
        # Initialize planning metrics tracking
        # This tracks planning behavior across different states and turns
        # - planning_turns_per_state: number of planning turns for each state
        # - planning_steps_per_state: number of planning steps for each state  
        # - All averages, std devs, and medians are calculated from these lists
        planning_metrics = {
            'planning_valid_action': [],
        }

        root_planning_state = simulator.observe()
        root_planning_stats_dict = simulator.get_key_stats()

        # independent planning prompt
        planning_instruction = self.predefined_instruction
        planning_user_prompt = self.predefined_query.format(
            turn_idx=1,
            step_idx=1,
            state=simulator.observe(),
            available_move=simulator.legal_moves_string,
            turn_left=self.max_plan_traj_n - 1,
            max_step=self.max_plan_horizon,
            extra_info="the simulation starts",
        )

        plan_message = [{"role": "system", "content": planning_instruction}, {"role":"user", "content": planning_user_prompt}]

        # need to remove the system prompt ids as this is already included in the outer loop
        # _query_ids = await self.loop.run_in_executor(
        #         None,
        #         lambda: self.tokenizer.apply_chat_template(
        #             [{"role":"user", "content": planning_user_prompt}], add_generation_prompt=False, tokenize=True
        #         ),
        #     )
        # _query_ids = _query_ids[len(self.system_prompt):]
        # plan_response_mask = [0] * len(_query_ids)

        plan_available_actions = simulator.legal_moves_list     # all the available planning actions
        simulation_history = [[f"State: {simulator.observe()}"]]

        plan_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    plan_message, add_generation_prompt=True, tokenize=True
                ),
            )

        visited_state_action_dict = {}
        idx = 0
        ever_win = 0
        while True:
            idx += 1
            if self.ablation_stockfish_move:
                # here we want to enumerate the move, and at least it will contain one optimal move for each state (UCB)
                if simulator.observe() not in visited_state_action_dict:
                    visited_state_action_dict[simulator.observe()] = {}
                    # set the action set in this state
                    _optimal_move = await simulator.analyse_position(num_moves=1)
                    _other_move = simulator.legal_moves_list[:self.stockfish_move_topk - 1]
                    _possible_action_set = _optimal_move + _other_move
                    visited_state_action_dict[simulator.observe()] = dict(zip(_possible_action_set, [0] * len(_possible_action_set)))   # 0 means action not chosen yet

                # the state has been visited before
                _possible_action_set = visited_state_action_dict[simulator.observe()]
                # chose the action has not been chosen yet first
                _not_chosen_action_set = [action for action, chosen in _possible_action_set.items() if not chosen]
                if len(_not_chosen_action_set) > 0:
                    _chosen_action = random.choice(_not_chosen_action_set)   # chose a random action from the not chosen action set
                else:
                    _chosen_action = random.choice(list(_possible_action_set.keys()))  # if all the actions have been chosen, chose a random action from the possible action set

                _possible_action_set[_chosen_action] = 1
                visited_state_action_dict[simulator.observe()] = _possible_action_set

                response_text = f"<move>{_chosen_action}</move>"

                # at lease one optimal move
                # optimal_move_list = await simulator.analyse_position(num_moves=1)
                # other_moves = simulator.legal_moves_list[:self.stockfish_move_topk - 1]
                # proposed_move_list = optimal_move_list + other_moves

                # random_stockfish_move = random.choice(proposed_move_list)
                # response_text = f"<move>{random_stockfish_move}</move>"
            else:
                if len(plan_message) > 2:
                    last_user_ids = await self.loop.run_in_executor(
                            None,
                            lambda: self.tokenizer.apply_chat_template(
                                [plan_message[-1]], add_generation_prompt=True, tokenize=True
                            )
                        )
                    last_user_ids = last_user_ids[len(self.system_prompt):]
                    plan_ids += last_user_ids

                with simple_timer("generate_sequences", metrics):
                    plan_step_request_id = f"{plan_request_id}_{plan_step}"
                    response_ids = await self.server_manager.generate(
                        request_id=plan_step_request_id, prompt_ids=plan_ids, sampling_params=sampling_params
                    )
                    plan_ids += response_ids

                # ====================== Parse Decision ====================================================#
                response_text = await self.loop.run_in_executor(
                    None, lambda: self.tokenizer.decode(response_ids, skip_special_tokens=True)
                )


            plan_message.append({"role": "assistant", "content": response_text})

            (plan_action_text, 
             plan_action_valid, 
             feedback, 
             (jump_to_continue, jump_to_reset, jump_to_summarize, jump_to_end)
             ) = self._extract_plan_from_assistant(
                response_text, 
                plan_available_actions,
                current_planning_turn=plan_turn,
                current_planning_step=plan_step,
                )

            if jump_to_end: # early stop answer
                raise NotImplementedError("Early stop answer is not supported for ablation summary")


            # append env feedback if there is any
            if len(feedback) > 0:
                plan_message.append({"role": "user", "content": feedback})

            if jump_to_summarize:   # break the planning loop and jump to summarize phase
                break

            if plan_action_text is not None:
                if plan_action_valid is not None:
                    planning_metrics['planning_valid_action'].append(int(plan_action_valid))
                # env step if action is detected
                # if self.opponent_mode == "stockfish":
                next_state, reward, done, env_info = await simulator.step(plan_action_text)
                # else:   
                #     next_state, reward, done, env_info = simulator.step(plan_action_text)

                plan_step += 1

                simulation_history[-1].append(f"Move: {plan_action_text}")
                if not done:
                    simulation_history[-1].append(f"Opponent's move: {env_info['oppo_move']}")
                simulation_history[-1].append(f"Reward: {reward}")
                simulation_history[-1].append(f"State: {next_state}")
                simulation_history[-1].append(f"Game Terminated: {done}")

                # check planning termination
                if done or (plan_step > self.max_plan_horizon):
                    # process termination feedback
                    if done:
                        game_status = simulator.check_results()
                        env_success = (game_status == "win")
                        env_lose = (game_status == "lose")
                        env_tie = (game_status == "tie")
                        # env_success = simulator.check_success()
                        simulation_history[-1].append(f"Result: {game_status}")
                        if env_success:
                            end_info = self.prompt_set['plan_success_prompt'].format(state=next_state, extra_info="")
                            ever_win = 1
                        elif env_lose:
                            end_info = self.prompt_set['plan_fail_prompt'].format(state=next_state, fail_reason="because you lose.", extra_info="")
                        elif env_tie:
                            end_info = self.prompt_set['plan_tie_prompt'].format(state=next_state, extra_info="")
                        else:
                            raise ValueError(f"Unknown game status: {game_status}")

                    else:
                        end_info = self.prompt_set['plan_fail_prompt'].format(
                            state=next_state, fail_reason="because reaching maximum planning steps ahead", extra_info="")

                    if plan_message[-1]['role'] == 'user':
                        plan_message[-1]['content'] += f"\n{end_info.format(state=next_state)}"
                    else:
                        plan_message.append({"role": "user", "content": end_info.format(state=next_state)})

                    if plan_turn == self.max_plan_traj_n:
                        jump_to_summarize = True
                        # when ready to break the planning loop, append user query to plan_ids
                        env_feedback_ids = await self.loop.run_in_executor(
                            None,
                            lambda: self.tokenizer.apply_chat_template(
                                [plan_message[-1]], add_generation_prompt=False, tokenize=True
                            )
                        )
                        env_feedback_ids = env_feedback_ids[len(self.system_prompt):]
                        plan_ids += env_feedback_ids

                        break
                    else:
                        jump_to_reset = True
                else:
                    env_feedback = self._format_plan_env_feedback(next_state, simulator.legal_moves_string, plan_turn, plan_step, env_info)
                    if plan_message[-1]['role'] == 'user':
                        plan_message[-1]['content'] += f"\n{env_feedback}"
                    else:
                        plan_message.append({"role": "user", "content": env_feedback})

            if jump_to_reset:     # reset the planning state
                
                simulator.reset_state(root_planning_stats_dict)
                plan_turn += 1
                plan_step = 1

                new_plan_prompt = self.predefined_query.format(
                    turn_idx=plan_turn,
                    step_idx=plan_step,
                    state=simulator.observe(),
                    available_move=simulator.legal_moves_string,
                    turn_left=self.max_plan_traj_n - plan_turn,
                    max_step=self.max_plan_horizon - plan_step + 1,
                    extra_info="the simulation has been restarted to original game state",
                )
                if plan_message[-1]['role'] == 'user':
                    plan_message[-1]['content'] += f"\n{new_plan_prompt}"
                else:
                    plan_message.append({"role": "user", "content": new_plan_prompt})   # might have overlap with previous env feedback in plan_ids

                simulation_history.append([f"State: {simulator.observe()}"])

            # update available actions
            plan_available_actions = simulator.legal_moves_list     # all the available planning actions

        # ready to summarize
        assert jump_to_summarize, print(f"jump_to_summarize = {jump_to_summarize}") # should be true

        # independent summary
        # process the simulation history
        history_string = ""
        for i, lst in enumerate(simulation_history):
            _traj = ", ".join(lst)
            _traj = f"Traj {i+1}: " + _traj
            history_string += _traj + "\n"
        plan_summarize_prompt = self.prompt_set['independent_summary_prompt'].format(
            state=root_planning_state,
            history=history_string
        )
        plan_message.append({"role": "user", "content": plan_summarize_prompt})

        # _input_message = [raw_message[0], {"role": "user", "content": plan_summarize_prompt}]
        # _input_message = [{"role": "system", "content": INDEPENDENT_SUMMARY_INSTRUCTION}, {"role": "user", "content": plan_summarize_prompt}]
        _input_message = [
            {"role": "system", "content": planning_instruction}, 
            {"role": "user", "content": plan_summarize_prompt},
        ]
        
        plan_summarize_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                _input_message, add_generation_prompt=True, tokenize=True
            ),
        )

        # need to remove the system prompt ids
        _summary_query_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                [{"role": "user", "content": plan_summarize_prompt}], add_generation_prompt=True, tokenize=True
            ),
        )
        _summary_query_ids = _summary_query_ids[len(self.system_prompt):]
        plan_response_mask = [0] * len(_summary_query_ids)

        with simple_timer("generate_sequences", metrics):
            response_ids = await self.server_manager.generate(
                request_id=uuid4().hex, prompt_ids=plan_summarize_ids, sampling_params=sampling_params
            )

        plan_summarize_ids += response_ids
        plan_response_mask += [1] * len(response_ids)


        response_text = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(response_ids, skip_special_tokens=True)
        )
        plan_message.append({"role": "assistant", "content": response_text})

        (final_action, final_action_valid) = self._extract_summarize_from_assistant(response_text)

        plan_ids = plan_summarize_ids[-len(plan_response_mask):]      # remove the system prompt ids

        summary_text = response_text

        return final_action, plan_message[1:], plan_ids, plan_response_mask, planning_metrics, metrics, summary_text, ever_win