import time
import numpy as np
import asyncio
import json
import logging
import os
import re
import random
import copy
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


def get_env(env_config):
    from tasks.classic_games.chess.chess_puzzle_env import ChessPuzzleEnv_Wrapper

    return ChessPuzzleEnv_Wrapper(env_config=env_config)

async def get_async_env(env_config):
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
    elif env_config.env_name=="ScienceWorld":

        from tasks.classic_games.scienceworld.scienceworld_env_client import AsyncScienceWorldEnvClient
        env = await AsyncScienceWorldEnvClient.connect(env_config=env_config, timeout=300)
        return env
        # from tasks.classic_games.scienceworld.scienceworld_env_client import ScienceWorldEnvClient
        # created_env = ScienceWorldEnvClient(env_config=env_config, timeout=300)
        # print(f"Create env id: {created_env.env_id}")
        # return created_env
    else:
        raise ValueError(f"Environment {env_config.env_name} is not supported.")






@register("game_value_lpm_agent")
class GameValueLPM_AgentLoop(AgentLoopBase):
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
        # answer_start_tag = '<answer>'
        cls.answer_start_tag = config.actor_rollout_ref.rollout.multi_turn.get("answer_tag", "<answer>")
        cls.answer_end_tag = cls.answer_start_tag[0] + "/" + cls.answer_start_tag[1:]

        cls.special_token_list = ["<think>", "</think>", "<move>", "</move>", "<|im_start|>", "<|im_end|>", "+", "#"]

        cls.action_pattern = re.compile(f'{re.escape(action_start_tag)}(.*?){re.escape(action_end_tag)}', re.DOTALL)
        cls.plan_action_pattern = re.compile(f'{re.escape(plan_action_start_tag)}(.*?){re.escape(plan_action_end_tag)}', re.DOTALL)
        cls.reset_pattern = re.compile(f'{re.escape(reset_start_tag)}(.*?){re.escape(reset_end_tag)}', re.DOTALL)
        cls.summary_pattern = re.compile(f'{re.escape(summary_start_tag)}(.*?){re.escape(summary_end_tag)}', re.DOTALL)
        cls.answer_pattern = re.compile(f'{re.escape(cls.answer_start_tag)}(.*?){re.escape(cls.answer_end_tag)}', re.DOTALL)

        # planning config
        cls.plan_mode = config.actor_rollout_ref.rollout.multi_turn.plan_mode   # "sequential" or "parallel"
        cls.early_stop_answer = config.actor_rollout_ref.rollout.multi_turn.get("early_stop_answer", False)   # whether to early stop when answer is detected during planning

        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        try:
            cls.system_prompt = tokenizer.apply_chat_template([{}], add_generation_prompt=False, tokenize=True)
        except Exception as e:
            cls.system_prompt = tokenizer.apply_chat_template([{"role": "system"}], add_generation_prompt=False, tokenize=True)


        cls.tool_masking = config.actor_rollout_ref.rollout.multi_turn.post_process.tool_masking

        cls.intermediate_instruction = config.actor_rollout_ref.rollout.multi_turn.get("intermediate_instruction", False)
        cls.random_move_when_invalid_plan_expansion = config.actor_rollout_ref.rollout.multi_turn.get("random_move_when_invalid_plan_expansion", False) # whether to switch to random move when invalid plan move is detected at the first turn (expansion)
        cls.random_move_when_invalid_plan_simulation = config.actor_rollout_ref.rollout.multi_turn.get("random_move_when_invalid_plan_simulation", False) # whether to switch to random move when invalid plan move is detected at the second turn (simulation)
        cls.random_move_when_invalid_act = config.actor_rollout_ref.rollout.multi_turn.random_move_when_invalid_act  # whether to switch to random act when invalid act is detected
        cls.include_random_expansion_in_response = config.actor_rollout_ref.rollout.multi_turn.include_random_expansion_in_response
        cls.include_random_simulation_in_response = config.actor_rollout_ref.rollout.multi_turn.include_random_simulation_in_response
        cls.include_random_act_in_response = config.actor_rollout_ref.rollout.multi_turn.include_random_act_in_response

        cls.use_remote_inference_when_invalid_move = config.actor_rollout_ref.rollout.multi_turn.use_remote_inference_when_invalid_move
        if cls.use_remote_inference_when_invalid_move:
            # initialize remote inference server
            cls.remote_sys_prompt = """\
You are a helpful assistant. You will be given a chess board state and an understanding of the situation. \
You will need to provide one or more moves purely based on the understanding only. \
Do not use any of your own knowledge or information.
"""
            cls.remote_query_prompt = """\
State: {state}
Available moves: {available_moves}

State Understanding: {state_understanding}
"""
            cls.remote_inference_prompt_type = config.actor_rollout_ref.rollout.multi_turn.remote_inference_prompt_type
            from src.workers.rollout.vllm_rollout.mcp_search_client import DirectModelAPIClient
            remote_inference_urls = config.actor_rollout_ref.rollout.multi_turn.get("remote_inference_urls", None)
            if not remote_inference_urls:
                env_urls = os.environ.get("REMOTE_INFERENCE_URLS")
                remote_inference_urls = [url.strip() for url in env_urls.split(",") if url.strip()] if env_urls else None
            if not remote_inference_urls:
                raise ValueError(
                    "remote_inference_urls is required when use_remote_inference_when_invalid_move is enabled. "
                    "Set actor_rollout_ref.rollout.multi_turn.remote_inference_urls or REMOTE_INFERENCE_URLS."
                )
            cls.model_api_client = DirectModelAPIClient(
                server_url_pool=remote_inference_urls,
                query_model=config.actor_rollout_ref.rollout.multi_turn.get("remote_inference_model", "rl_ckpt"),
                max_output_tokens=config.actor_rollout_ref.rollout.multi_turn.get("remote_inference_max_output_tokens", 100),
                # sys_prompt=cls.remote_sys_prompt
                )


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

        # exploration
        cls.exploration_plan = config.actor_rollout_ref.rollout.multi_turn.exploration_plan
        cls.full_exploration_in_simulation = config.actor_rollout_ref.rollout.multi_turn.full_exploration_in_simulation
        cls.exploration_act = config.actor_rollout_ref.rollout.multi_turn.exploration_act
        cls.exploration_mode = config.actor_rollout_ref.rollout.multi_turn.exploration_mode
        if cls.exploration_mode is not None:
            assert cls.exploration_mode in ["expert", "random"]
        cls.include_turn2_simulation_in_response = config.actor_rollout_ref.rollout.multi_turn.include_turn2_simulation_in_response

        cls.process_sft_dataset = config.actor_rollout_ref.rollout.multi_turn.post_process.sft.process_sft_dataset
        if cls.process_sft_dataset:
            # sft prompt set
            prompt_set_path = config.actor_rollout_ref.rollout.multi_turn.post_process.sft.sft_prompt_set
            module_path, class_name = prompt_set_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            cls.sft_prompt_set = getattr(module, class_name)

            cls.use_value_table = config.actor_rollout_ref.rollout.multi_turn.post_process.sft.use_value_table
            if cls.use_value_table:
                cls.value_table_url = config.actor_rollout_ref.rollout.multi_turn.post_process.sft.value_table_url
                cls.value_table_retrieve_strategy = config.actor_rollout_ref.rollout.multi_turn.post_process.sft.value_table_retrieve_strategy
                cls.value_table_max_values_per_state = config.actor_rollout_ref.rollout.multi_turn.post_process.sft.value_table_max_values_per_state
                from tasks.classic_games.chess.value_table_server import ValueTable
                cls.value_table = ValueTable(max_values_per_state=cls.value_table_max_values_per_state)
                cls.value_table_exploration_p = config.actor_rollout_ref.rollout.multi_turn.post_process.sft.value_table_exploration_p
        else:
            cls.sft_prompt_set = None
            cls.use_value_table = False

        cls.sft_dataset_type = config.actor_rollout_ref.rollout.multi_turn.post_process.sft.sft_dataset_type
        cls.distillation_mode = config.actor_rollout_ref.rollout.multi_turn.post_process.distillation_mode

        cls._env_initialized = False

        cls.ablation = config.actor_rollout_ref.rollout.multi_turn.get("ablation", None)
        cls.ablation_stockfish_move = config.actor_rollout_ref.rollout.multi_turn.get("ablation_stockfish_move", False)

        cls.max_window_size = config.actor_rollout_ref.rollout.multi_turn.get("max_window_size", 1)
        cls.validate_action = config.actor_rollout_ref.rollout.multi_turn.get("validate_action", True)

        if cls.distillation_mode in ["supervised-sd", "op-sd"]:
            cls.distillation_enable = True      # when True, record expansion and summary understanding in token-in-token-out process
        else:
            cls.distillation_enable = False


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
            turn_left=self.max_steps,
            task_description=self.env.task_goal if hasattr(self.env, 'task_goal') else "")

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

        if sampling_params['temperature'] == 0.:   # temperature is 0 demonstrate evaluation, so no exploration
            self.current_exploration_plan = 0.
            self.current_exploration_act = 0.
        else:
            self.current_exploration_plan = self.exploration_plan
            self.current_exploration_act = self.exploration_act

        system_prompt = self.prompt_set['system_prompt']

        # reset game and add user content
        initial_query = await self.build_env(seed)

        instruction_message = [
            {"role": "system", "content": system_prompt},
        ]
        turn_message = [[
            {"role": self.conversation_prefix, "content": initial_query}
        ]]
        clean_turn_message = [[
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
        distill_mask = []
        state_summary_pair = []   # (state, available_actions, summary), for SFT Loss

        if self.distillation_enable:
            distill_prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    [instruction_message[0], turn_message[0][0]], add_generation_prompt=False, tokenize=True
                ),
            )    # student ids

            distill_teacher_prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    [instruction_message[0], turn_message[0][0]], add_generation_prompt=False, tokenize=True
                ),
            )    # teacher ids

            distill_response_mask = []          # student response mask
            distill_teacher_response_mask = []    # teacher response mask

            # if supervised-sd, the teacher response is the summary text, the student response is the summary text replacing root state inference
            # if op-sd, the teacher response is the root state inference replacing summary text, the student response is the root state inference

        user_turns, assistant_turns = 0, 0
        early_stop_flag = 0     # if not stop at agent response

        step = 0
        env_success=False
        env_end = False
        env_early_stop = False
        turn_valid_records = []
        plan_early_stop_lst = []
        plan_num_turns_lst = []
        plan_ever_win_lst = []

        while True:

            # start planning
            original_planning_stats_dict = self.env.get_key_stats()   

            simulator = await get_async_env(self.env_config)
            _ = await simulator.reset_state(original_planning_stats_dict)
            available_moves = simulator.legal_moves_string

            if self.max_window_size == 1:
                current_plan_message = [instruction_message[0], turn_message[-1][0]]    # the system prompt and the current state query
            else:
                current_plan_message = [{"role": "system", "content": system_prompt}]
                for m in clean_turn_message[-self.max_window_size:]:
                    for k in m:
                        current_plan_message.append(copy.copy(k))

            if self.plan_mode == "sequential":
                (
                    planned_action, 
                    updated_plan_message, 
                    plan_ids,
                    plan_response_mask,
                    planning_metrics,
                    metrics,
                    plan_reward_lst,
                    summary_text,
                    summary_decision_text,
                    plan_early_stop,
                    num_plan_turns,
                    num_diverse_move,
                    ever_win
                    ) = await self.run_independent_planning(simulator, current_plan_message, sampling_params, metrics)
            elif self.plan_mode == "parallel":
                (
                    planned_action, 
                    planned_action_valid,
                    action_feedback,
                    updated_plan_message, 
                    plan_ids,
                    plan_response_mask,
                    planning_metrics,
                    metrics,
                    plan_reward_lst,
                    summary_text,
                    summary_decision_text,
                    plan_early_stop,
                    num_plan_turns,
                    num_diverse_move,
                    ever_win,
                    full_distill_plan_ids,
                    full_distill_plan_mask,
                    full_distill_plan_teacher_ids,
                    full_distill_plan_teacher_response_mask,
                ) = await self.run_parallel_planning(simulator, current_plan_message, sampling_params, metrics)
            elif self.plan_mode == 'policy_centric_parallel':
                (
                    planned_action,
                    planned_action_valid,
                    action_feedback,
                    updated_plan_message, 
                    plan_ids,
                    plan_response_mask,
                    planning_metrics,
                    metrics,
                    plan_reward_lst,
                    summary_text,
                    summary_decision_text,
                    plan_early_stop,
                    num_plan_turns,
                    num_diverse_move,
                    ever_win
                ) = await self.run_policy_centric_parallel_planning(simulator, current_plan_message, sampling_params, metrics)
            else:
                raise ValueError(f"Invalid plan mode: {self.plan_mode}")

            prompt_ids += plan_ids
            response_mask += plan_response_mask
            # reward_lst += ([0.] * len(plan_response_mask))
            assert len(plan_reward_lst) == len(plan_response_mask), print(f"mismatch in plan_reward_lst and plan_response_mask: {len(plan_reward_lst)} != {len(plan_response_mask)}")
            reward_lst += plan_reward_lst
            plan_early_stop_lst.append(int(plan_early_stop))
            plan_num_turns_lst.append(num_plan_turns)
            plan_ever_win_lst.append(ever_win)
            if self.distillation_enable:
                if self.distillation_mode == "supervised-sd":
                    distill_prompt_ids.extend(full_distill_plan_ids)
                    distill_response_mask.extend(full_distill_plan_mask)
                    distill_teacher_response_mask += full_distill_plan_teacher_response_mask
                elif self.distillation_mode == "op-sd":
                    distill_teacher_prompt_ids.extend(full_distill_plan_teacher_ids)
                    distill_teacher_response_mask += full_distill_plan_teacher_response_mask
                    distill_response_mask.extend(full_distill_plan_mask)

            turn_message[-1].extend(updated_plan_message)
            turn_valid_records.extend(planning_metrics['planning_valid_action'])

            if self.process_sft_dataset:
                state_summary_pair.append(
                    {"state": original_planning_stats_dict['FEN'], 
                    "available_moves": available_moves,
                    "summary": summary_text,
                    "decision": summary_decision_text,
                    }
                    )

            # process the planned action
            # planned_action, planned_action_valid, action_feedback = self._extract_action_from_assistant(planned_action)
            
            # Record the planned action validity for tracking
            self.planned_action_valid_records.append(planned_action_valid)

            # handle None action before env step
            if planned_action is None:     # this mean no valid action and we dont support auto random move
                done = True
                env_early_stop = True
                end_info = action_feedback + "\n" + "The game ends. You lose the game because no valid action is detected."
                env_message = [{"role": self.conversation_prefix, "content": end_info}]
                turn_message.append(env_message)
                clean_turn_message.append(env_message)

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
                if self.distillation_enable:
                    if self.distillation_mode == "supervised-sd":
                        distill_prompt_ids.extend(final_env_feedback_ids)
                        distill_response_mask.extend([0] * len(final_env_feedback_ids))
                        distill_teacher_response_mask += [0] * len(final_env_feedback_ids)
                    elif self.distillation_mode == "op-sd":
                        distill_teacher_prompt_ids.extend(final_env_feedback_ids)
                        distill_teacher_response_mask += [0] * len(final_env_feedback_ids)
                        distill_response_mask.extend([0] * len(final_env_feedback_ids))

                if self.final_reward_readjustment:
                    reward_lst[-1] += invalid_action_reward
                    reward_lst += [0.] * len(final_env_feedback_ids)
                else:
                    reward_lst += [0.] * len(final_env_feedback_ids)
                    reward_lst[-1] += invalid_action_reward

                break
            
            clean_turn_message[-1].extend([{"role": "assistant", "content": f"{self.answer_start_tag}{planned_action}{self.answer_end_tag}"}])

            move_matching_flag = False
            if self.use_stockfish_move_matching_reward:
                if planned_action_valid:
                    stockfish_proposed_move_list = await self.env.analyse_position(num_moves=self.stockfish_move_topk)
                    if planned_action in stockfish_proposed_move_list:
                        move_matching_flag = True

            # Step the environment with the extracted action
            with simple_timer("env_step", metrics):
                next_state, reward, env_done, env_info = await self.env.step(planned_action)

                reward = reward * self.act_reward_scale
            
            if not planned_action_valid:
                reward -= self.act_action_format_penality
            
            if self.use_stockfish_move_matching_reward and move_matching_flag:
                reward += self.stockfish_move_matching_reward

            # reward_lst[-1] += reward

            step += 1
            # print(f"step = {step}, length = {len(prompt_ids)}")
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
                if self.distillation_enable:
                    if self.distillation_mode == "supervised-sd":
                        distill_prompt_ids.extend(final_env_feedback_ids)
                        distill_response_mask.extend([0] * len(final_env_feedback_ids))
                        distill_teacher_response_mask += [0] * len(final_env_feedback_ids)
                    elif self.distillation_mode == "op-sd":
                        distill_teacher_prompt_ids.extend(final_env_feedback_ids)
                        distill_teacher_response_mask += [0] * len(final_env_feedback_ids)
                        distill_response_mask.extend([0] * len(final_env_feedback_ids))
                
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
            env_feedback = self._format_env_feedback(next_state, self.env.legal_moves_string, step, env_info, self.env.task_goal if hasattr(self.env, 'task_goal') else "")
            env_message = [{"role": "user", "content": action_feedback + "\n" + env_feedback}]
            turn_message.append([{"role": "user", "content": action_feedback + "\n" + env_feedback}])
            clean_turn_message.append([{"role": "user", "content": action_feedback + "\n" + env_feedback}])

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
            if self.distillation_enable:
                if self.distillation_mode == "supervised-sd":
                    distill_prompt_ids.extend(env_feedback_ids)
                    distill_response_mask.extend([0] * len(env_feedback_ids))
                    distill_teacher_response_mask += [0] * len(env_feedback_ids)
                elif self.distillation_mode == "op-sd":
                    distill_teacher_prompt_ids.extend(env_feedback_ids)
                    distill_teacher_response_mask += [0] * len(env_feedback_ids)
                    distill_response_mask.extend([0] * len(env_feedback_ids))

            # print(f"## 4, len = {len(clean_turn_message)}, prompt length = {len(prompt_ids)}")

            user_turns += 1

        response_ids = prompt_ids[-len(response_mask):]
        prompt_ids = prompt_ids[:len(prompt_ids) - len(response_mask)]
        if self.distillation_enable:
            if self.distillation_mode == "supervised-sd":
                distill_response_ids = distill_prompt_ids[-len(distill_response_mask):]
            elif self.distillation_mode == "op-sd":
                distill_teacher_response_ids = distill_teacher_prompt_ids[-len(distill_teacher_response_mask):]

        # Must await close() for AsyncScienceWorldEnvClient - otherwise env is never released to server pool
        close_result = await self.env.close()
        if asyncio.iscoroutine(close_result):
            await close_result

        metrics['others'] = {
            "plan_ever_win_rate": int(sum(plan_ever_win_lst) > 0),
            "num_diverse_move": num_diverse_move,
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
        extra_ids = {}

        # process state_summary_pair for periodic SFT Loss
        if self.process_sft_dataset: # and env_success:   # only keep the successful state_summary_pair
            if self.sft_dataset_type == "multiturn":
                raise NotImplementedError("Multiturn SFT dataset is not supported yet for Value Infer Loop")
                sft_dataset = await self._process_multiturn_sft_dataset(instruction_message, turn_message, env_success)
            elif self.sft_dataset_type == "singleturn":
                sft_dataset = await self._process_value_sft_dataset(state_summary_pair, env_success)
        else:
            sft_dataset=[]

        # Knowledge Distillation Loss
        if self.distillation_mode == "supervised-sd":
            # need to compute student and teacher logprob
            # original traj:        <history> <obs> <understand>          .....simulation... <summary understand> <final action> <next obs> ...
            # for  logp:            <history> <obs> [<summary understand>]  .....simulation... [<summary understand>] <final action> <next obs> ...
            # for teacher logp:     same as student
            extra_ids['distill_teacher_response_mask'] = distill_teacher_response_mask[:self.response_length]
            extra_ids['distill_student_response'] = distill_response_ids[:self.response_length]
            extra_ids['distill_student_response_mask'] = distill_response_mask[:self.response_length]
            extra_ids['distill_teacher_response'] = response_ids[:self.response_length]
        elif self.distillation_mode == "op-sd":
            # need to process the data for self-distill
            # for student logp:    <history> <obs> 
            extra_ids['distill_teacher_response'] = distill_teacher_response_ids[:self.response_length]
            extra_ids['distill_teacher_response_mask'] = distill_teacher_response_mask[:self.response_length]
            extra_ids['distill_student_response_mask'] = distill_response_mask[:self.response_length]
            extra_ids['distill_student_response'] = response_ids[:self.response_length]
        else:
            pass
        
        # print(len(response_ids),len(distill_teacher_response_ids), len(distill_teacher_response_mask), len(response_ids), len(distill_response_mask))
        # print(sum(distill_teacher_response_mask), sum(distill_response_mask))
        # print(self.tokenizer.decode(distill_teacher_response_ids[distill_teacher_response_mask.index(1):distill_teacher_response_mask[distill_teacher_response_mask.index(1):].index(0)+distill_teacher_response_mask.index(1)])); print(self.tokenizer.decode(response_ids[distill_response_mask.index(1):distill_response_mask[distill_response_mask.index(1):].index(0)+distill_response_mask.index(1)]))
        
        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[:self.response_length],
            response_mask=response_mask[:self.response_length],
            reward_lst=reward_lst[:self.response_length],
            num_turns=step + 1,
            metrics=metrics,
            sft_dataset=sft_dataset,
            extra=extra_ids
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
                    plan_message, add_generation_prompt=False, tokenize=True
                ),
            )

        plan_early_stop = False
        idx = 0
        num_diverse_move = 0
        ever_win = 0
        plan_response_length = []
        while True:
            idx += 1
            if len(plan_message) > 2:
                last_user_ids = await self.loop.run_in_executor(
                        None,
                        lambda: self.tokenizer.apply_chat_template(
                            [plan_message[-1]], add_generation_prompt=False, tokenize=True
                        )
                    )
                last_user_ids = last_user_ids[len(self.system_prompt):]
                plan_ids += last_user_ids
                plan_response_mask += [0] * len(last_user_ids)
                plan_reward_lst += [0.] * len(last_user_ids)

            # two stage value inference

            (
                response_text, 
                state_understanding_text, 
                infer_message, 
                plan_ids, 
                infer_response_mask, 
                infer_reward_lst, 
                metrics
                ) = await self._two_stage_value_inference(
                    simulator.observe(), 
                    plan_ids, 
                    metrics, 
                    sampling_params,
                    move_decision_prompt=self.prompt_set['state_move_query_prompt'],
                    value_decision_prompt=self.prompt_set['state_value_query_prompt'].format(state=simulator.observe()),
                )
            plan_message.extend(infer_message)
            plan_response_mask.extend(infer_response_mask)
            plan_reward_lst.extend(infer_reward_lst)
            plan_response_length.append(len(infer_response_mask))            

            # plan_action_text = self._extract_action_from_assistant(response_text)

            # ====================== Parse Decision ====================================================#
            # response_text = await self.loop.run_in_executor(
                # None, lambda: self.tokenizer.decode(response_ids, skip_special_tokens=True)
            # )
            # plan_message.append({"role": "assistant", "content": response_text})

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
                
                return plan_action_text, plan_message[1:], plan_ids, plan_response_mask, planning_metrics, metrics, plan_reward_lst, state_understanding_text, response_text, plan_early_stop, idx

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
                            ever_win = 1
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
                
                await simulator.reset_state(root_planning_stats_dict)
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
                    history=history_string
                )
            else:
                plan_summarize_prompt = self.prompt_set['plan_action_query_prompt']

            plan_message.append({"role": "user", "content": plan_summarize_prompt})


            _plan_summarize_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": plan_summarize_prompt}], add_generation_prompt=False, tokenize=True
                ),
            )
            _plan_summarize_ids = _plan_summarize_ids[len(self.system_prompt):]
            plan_ids += _plan_summarize_ids
            plan_response_mask += [0] * len(_plan_summarize_ids)
            plan_reward_lst += [0.] * len(_plan_summarize_ids)

            (
                summarize_response_text, 
                summarize_state_understanding_text, 
                infer_message, 
                plan_ids, 
                infer_response_mask, 
                infer_reward_lst, 
                metrics
                ) = await self._two_stage_value_inference(
                    root_planning_state, 
                    plan_ids, 
                    metrics, 
                    sampling_params,
                    move_decision_prompt=self.prompt_set['state_answer_query_prompt'],
                    value_decision_prompt=self.prompt_set['simulation_state_value_query_prompt'].format(state=root_planning_state),
                )
            plan_message.extend(infer_message)
            plan_response_mask.extend(infer_response_mask)
            plan_reward_lst.extend(infer_reward_lst)

            summary_response_length = len(infer_response_mask)
            mean_plan_response_length = np.mean(plan_response_length)
            
        else:   # independent summary
            raise NotImplementedError("Independent summary is not supported for two stage value inference")


        (final_action, final_action_valid) = self._extract_summarize_from_assistant(summarize_response_text)

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

        return final_action, plan_message[1:], plan_ids, plan_response_mask, planning_metrics, metrics, plan_reward_lst, summarize_state_understanding_text, summarize_response_text, plan_early_stop, idx, num_diverse_move, ever_win

    async def _two_stage_value_inference(
        self, 
        state, 
        input_plan_ids, 
        metrics, 
        sampling_params, 
        move_decision_prompt=None, 
        value_decision_prompt=None, 
        exploration_p = 0.,
        allow_random_move=None,
        include_offpolicy_move_in_response=None,
        include_offpolicy_move_tag=None,
        available_move=None,
        extra_message=None,
        ):
        """
        if the model can generate valid moves then everything is fine
        if not, we need to decide:
            1. whether to user random exploration to replace invalid move
            2. whether to use remote off-policy move to replace invalid move
            3. whether to include the replaced move in the response
            4. whether to use epsilon random exploration
        """

        _infer_message = []
        _input_plan_response_mask = []
        _input_plan_reward_lst = []
        _distill_ids_lst = []

        # stage 1: state understanding
        state_understanding_prompt = value_decision_prompt if value_decision_prompt is not None else self.prompt_set['state_value_query_prompt'].format(state=state)
        _infer_message.append({"role": "user", "content": state_understanding_prompt})
        state_understanding_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                [{"role": "user", "content": state_understanding_prompt}], add_generation_prompt=True, tokenize=True
            ),
        )
        state_understanding_ids = state_understanding_ids[len(self.system_prompt):]

        input_plan_ids += state_understanding_ids
        _input_plan_response_mask += [0] * len(state_understanding_ids)
        _input_plan_reward_lst += [0.] * len(state_understanding_ids)
        _distill_ids_lst.append(list(state_understanding_ids))


        state_understanding_text=None
        if self.use_value_table:      # use pre-collected value table
            if random.random() < self.value_table_exploration_p:
                pass
            else:
                value_table_entries = await self.value_table.get(state, retrieve_strategy=self.value_table_retrieve_strategy)
                if value_table_entries.get("found"):
                    state_understanding_text = value_table_entries.get("value")
                    order = value_table_entries.get("order")
                    response_ids = await self.loop.run_in_executor(
                        None,
                        lambda: self.tokenizer(state_understanding_text)['input_ids'],
                    )

        if state_understanding_text is None:    # use model to reason

            with simple_timer("generate_sequences", metrics):
                plan_step_request_id = f"{uuid4().hex}_stage_1"
                sampling_params['max_tokens'] = self.each_response_length
                sampling_params['stop'] = [self.tokenizer.eos_token]
                response_ids = await self.server_manager.generate(
                    request_id=plan_step_request_id, prompt_ids=input_plan_ids, sampling_params=sampling_params
                )
                _has_newline_symbol = None
                if response_ids[-1] == self.tokenizer.eos_token_ids and len(response_ids) > 1:
                    _has_newline_symbol = (response_ids[-2] == self.tokenizer.encode("\n")[0])
                    response_ids += self.tokenizer.encode("\n")   # add \n to align with template
                else:
                    _has_newline_symbol = (response_ids[-1] == self.tokenizer.encode("\n")[0])
                    response_ids += [self.tokenizer.eos_token_ids]
                    response_ids += self.tokenizer.encode("\n")   # add \n to align with template
            
            state_understanding_text = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(response_ids, skip_special_tokens=True)
            )
            if state_understanding_text[-1] == "\n" and not _has_newline_symbol:    # when the decoded text end with \n but the response ids does not, this \n come from the tokenizer with skip_special_tokens=True
                state_understanding_text = state_understanding_text[:-1]

        input_plan_ids += response_ids
        _input_plan_response_mask += [1] * len(response_ids)
        _input_plan_reward_lst += [0.] * len(response_ids)
        _distill_ids_lst.append(list(response_ids))
        _infer_message.append({"role": "assistant", "content": state_understanding_text})

        # stage 2: move decision
        if random.random() < exploration_p:
            # explore
            response_text = ""
        else:
            move_decision_prompt = self.prompt_set['state_move_query_prompt'] if move_decision_prompt is None else move_decision_prompt
            _infer_message.append({"role": "user", "content": move_decision_prompt})
            move_decision_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": move_decision_prompt}], add_generation_prompt=True, tokenize=True
                ),
            )
            move_decision_ids = move_decision_ids[len(self.system_prompt):]
            input_plan_ids += move_decision_ids
            _input_plan_response_mask += [0] * len(move_decision_ids)
            _input_plan_reward_lst += [0.] * len(move_decision_ids)
            _distill_ids_lst.append(list(move_decision_ids))

            with simple_timer("generate_sequences", metrics):
                plan_step_request_id = f"{uuid4().hex}_stage_2"
                sampling_params['max_tokens'] = 100
                sampling_params['stop'] = ["</move>", "</answer>", self.tokenizer.eos_token]
                response_ids = await self.server_manager.generate(
                    request_id=plan_step_request_id, prompt_ids=input_plan_ids, sampling_params=sampling_params
                )

                _has_newline_symbol = None
                # add a eos token to the end if early stop
                if response_ids[-1] != self.tokenizer.eos_token_ids:
                    _has_newline_symbol = (response_ids[-1] == self.tokenizer.encode("\n")[0])
                    response_ids.append(self.tokenizer.eos_token_ids)
                else:
                    _has_newline_symbol = (response_ids[-2] == self.tokenizer.encode("\n")[0]) if len(response_ids) > 1 else False
                response_ids += self.tokenizer.encode("\n")   # add \n to align with template
            
            response_text = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(response_ids, skip_special_tokens=True)
            )
            if response_text[-1] == "\n" and not _has_newline_symbol:    # when the decoded text end with \n but the response ids does not, this \n come from the tokenizer with skip_special_tokens=True
                response_text = response_text[:-1]

            # parse final answer decision
            parsed_action, parsed_action_valid, parsed_feedback = self._extract_action_from_assistant(response_text)

            # parsed_final_action, parsed_final_action_valid = self._extract_summarize_from_assistant(response_text)
            if allow_random_move and not parsed_action_valid and include_offpolicy_move_in_response:
                if self.use_remote_inference_when_invalid_move:
                    if self.remote_inference_prompt_type == "separate":
                        remote_query_prompt = self.remote_query_prompt.format(
                            state=state, 
                            available_moves=available_move, 
                            state_understanding=state_understanding_text
                            ) + f"Based on the state understanding, provide one move with each move wrapped by {include_offpolicy_move_tag[0]} and {include_offpolicy_move_tag[1]}. For example, {include_offpolicy_move_tag[0]}e2e4{include_offpolicy_move_tag[1]}..."
                        remote_message = [
                            {"role": "system", "content": self.remote_sys_prompt},
                            {"role": "user", "content": remote_query_prompt
                            },
                        ]
                    elif self.remote_inference_prompt_type == "original":
                        remote_message = extra_message + _infer_message
                    else:
                        raise ValueError(f"Invalid remote inference prompt type: {self.remote_inference_prompt_type}")

                    remote_response = await self.model_api_client.request_chat(
                        messages_lst=[remote_message],
                    )
                    remote_response = remote_response[0]['content']        # the response is a string

                    (
                        parsed_action, 
                        parsed_action_valid, 
                        parsed_feedback
                        ) = self._extract_action_from_assistant(remote_response)
                    selected_action_string = f"{include_offpolicy_move_tag[0]}{parsed_action}{include_offpolicy_move_tag[1]}"
                    response_text = "Off-Policy Move: " + selected_action_string
                    response_ids = await self.loop.run_in_executor(
                        None,
                        lambda: self.tokenizer(response_text)['input_ids'],
                    )
                    parsed_feedback = parsed_feedback.replace("random move", "off-policy move")
                    parsed_action_valid = 0.

                else:
                    assert parsed_action is not None
                    selected_action_string = f"{include_offpolicy_move_tag[0]}{parsed_action}{include_offpolicy_move_tag[1]}"
                    response_text = "Random Move: " + selected_action_string
                    response_ids = await self.loop.run_in_executor(
                        None,
                        lambda: self.tokenizer(response_text)['input_ids'],
                    )
                    response_ids += [self.tokenizer.eos_token_ids]   # add eos token to the end
                    response_ids += self.tokenizer.encode("\n")   # add \n to align with template
            
            input_plan_ids += response_ids
            _input_plan_response_mask += [1] * len(response_ids)
            _input_plan_reward_lst += [0.] * len(response_ids)
            _distill_ids_lst.append(list(response_ids))

        _infer_message.append({"role": "assistant", "content": response_text})

        return (
            response_text, 
            state_understanding_text, 
            _infer_message, 
            input_plan_ids, 
            _input_plan_response_mask, 
            _input_plan_reward_lst, 
            metrics,
            # parse ret
            parsed_action, 
            parsed_action_valid, 
            parsed_feedback,
            # auxiliary
            _distill_ids_lst,
            )

    async def _two_stage_value_parallel_inference(
        self, 
        state, 
        input_plan_ids, 
        metrics, 
        sampling_params, 
        move_decision_prompt=None, 
        value_decision_prompt=None, 
        exploration_p = None, 
        available_move=None, 
        num_plan_moves_allowed=None,
        simulator=None,
        current_planning_step=None,
        allow_random_move=None,
        include_offpolicy_move_in_response=None,
        include_offpolicy_move_tag=None,
        extra_message=None,     # if we serve the checkpoint model to infer movement
        ):
        """

        """
        _infer_message = []
        _input_plan_response_mask = []
        _input_plan_reward_lst = []
        _ids = input_plan_ids.copy()
        _distill_ids_lst = []

        # stage 1: state understanding
        state_understanding_prompt = value_decision_prompt if value_decision_prompt is not None else self.prompt_set['state_value_query_prompt'].format(state=state)
        _infer_message.append({"role": "user", "content": state_understanding_prompt})
        state_understanding_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                [{"role": "user", "content": state_understanding_prompt}], add_generation_prompt=True, tokenize=True
            ),
        )
        state_understanding_ids = state_understanding_ids[len(self.system_prompt):]
        _ids += state_understanding_ids
        _input_plan_response_mask += [0] * len(state_understanding_ids)
        _input_plan_reward_lst += [0.] * len(state_understanding_ids)
        _distill_ids_lst.append(list(state_understanding_ids))  # copy so later in-place mutations don't affect stored segment

        with simple_timer("generate_sequences", metrics):
            plan_step_request_id = f"{uuid4().hex}_stage_1"
            sampling_params['max_tokens'] = self.each_response_length
            sampling_params['stop'] = [self.tokenizer.eos_token]
            response_ids = await self.server_manager.generate(
                request_id=plan_step_request_id, prompt_ids=_ids, sampling_params=sampling_params
            )
            _has_newline_symbol = None
            if response_ids[-1] == self.tokenizer.eos_token_ids and len(response_ids) > 1:
                _has_newline_symbol = (response_ids[-2] == self.tokenizer.encode("\n")[0])
                response_ids += self.tokenizer.encode("\n")   # add \n to align with template, add it only if ended with eos token
            else:
                _has_newline_symbol = (response_ids[-1] == self.tokenizer.encode("\n")[0])
                response_ids += [self.tokenizer.eos_token_ids]
                response_ids += self.tokenizer.encode("\n") 

            _ids += response_ids
            _input_plan_response_mask += [1] * len(response_ids)
            _input_plan_reward_lst += [0.] * len(response_ids)
            _distill_ids_lst.append(list(response_ids))  # copy so in-place mutations don't affect stored segment
        
        state_understanding_text = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(response_ids, skip_special_tokens=True)
        )
        if state_understanding_text[-1] == "\n" and not _has_newline_symbol:    # when the decoded text end with \n but the response ids does not, this \n come from the tokenizer with skip_special_tokens=True
            state_understanding_text = state_understanding_text[:-1]
        _infer_message.append({"role": "assistant", "content": state_understanding_text})

        # stage 2: move decision
        if exploration_p is None:
            exploration_p = self.current_exploration_plan

        move_decision_prompt = self.prompt_set['state_move_query_prompt'] if move_decision_prompt is None else move_decision_prompt
        _infer_message.append({"role": "user", "content": move_decision_prompt})
        move_decision_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                [{"role": "user", "content": move_decision_prompt}], add_generation_prompt=True, tokenize=True
            ),
        )
        move_decision_ids = move_decision_ids[len(self.system_prompt):]
        _ids += move_decision_ids
        _input_plan_response_mask += [0] * len(move_decision_ids)
        _input_plan_reward_lst += [0.] * len(move_decision_ids)
        _distill_ids_lst.append(list(move_decision_ids))  # copy so in-place mutations don't affect stored segment
        
        if random.random() < exploration_p:
            # explore
            if self.exploration_mode == "random":
                _random_chosen_move = random.sample(available_move, min(num_plan_moves_allowed, len(available_move)))
                response_text = "\n".join([f"<move>{move}</move>" for move in _random_chosen_move])
                response_text = "Random Explore: " + response_text
            elif self.exploration_mode == "expert":
                _expert_chosen_move = await simulator.analyse_position(num_moves=min(num_plan_moves_allowed, len(available_move)))
                response_text = "\n".join([f"<move>{move}</move>" for move in _expert_chosen_move])
                response_text = "Expert Explore: " + response_text
            else:
                raise ValueError(f"Invalid exploration mode: {self.exploration_mode}")
            
            response_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer(response_text)['input_ids'],
            )
            # _ids += response_ids
            # _input_plan_response_mask += [1] * len(response_ids)
            # _input_plan_reward_lst += [0.] * len(response_ids)
            # _infer_ids.append(response_ids)
            # _infer_masks.append([1] * len(response_ids))
        else:
            with simple_timer("generate_sequences", metrics):
                plan_step_request_id = f"{uuid4().hex}_stage_2"
                sampling_params['max_tokens'] = 100
                sampling_params['stop'] = [self.tokenizer.eos_token]    # does not stop at action as we may need multiple moves
                response_ids = await self.server_manager.generate(
                    request_id=plan_step_request_id, prompt_ids=_ids, sampling_params=sampling_params
                )
                _has_newline_symbol = None
                if response_ids[-1] == self.tokenizer.eos_token_ids and len(response_ids) > 1:
                    _has_newline_symbol = (response_ids[-2] == self.tokenizer.encode("\n")[0])
                    response_ids += self.tokenizer.encode("\n")   # add \n to align with template
                else:
                    _has_newline_symbol = (response_ids[-1] == self.tokenizer.encode("\n")[0])
                    response_ids += [self.tokenizer.eos_token_ids]
                    response_ids += self.tokenizer.encode("\n")   # add \n to align with template

            response_text = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(response_ids, skip_special_tokens=True)
            )
            if response_text[-1] == "\n" and not _has_newline_symbol:    # when the decoded text end with \n but the response ids does not, this \n come from the tokenizer with skip_special_tokens=True
                response_text = response_text[:-1]

        # parse decision
        (
            parsed_plan_action_list, 
            parsed_plan_action_valid, 
            parsed_feedback
            ) = self._extract_parallel_plan_from_assistant(
                response_text,
                available_move,
                current_planning_step=current_planning_step,
                num_plan_moves_allowed=num_plan_moves_allowed,
                allow_random_move=allow_random_move,
                )
        
        if allow_random_move and not parsed_plan_action_valid and include_offpolicy_move_in_response:     # off-policy move
            if self.use_remote_inference_when_invalid_move:    # use remote inference to generate the off-policy move
                if self.remote_inference_prompt_type == "separate":
                    # a separate prompt for action selection
                    query_message = self.remote_query_prompt.format(
                        state=state, 
                        available_moves=available_move, 
                        state_understanding=state_understanding_text
                        ) + f"Based on the state understanding, provide {num_plan_moves_allowed} moves with each move wrapped by {include_offpolicy_move_tag[0]} and {include_offpolicy_move_tag[1]}. For example, {include_offpolicy_move_tag[0]}e2e4{include_offpolicy_move_tag[1]}..."
                    
                    remote_message = [
                        {"role": "system", "content": self.remote_sys_prompt},
                        {"role": "user", "content": query_message},
                    ]
                elif self.remote_inference_prompt_type == "original":
                    remote_message = extra_message + _infer_message
                else:
                    raise ValueError(f"Invalid remote inference prompt type: {self.remote_inference_prompt_type}")
                
                remote_response = await self.model_api_client.request_chat(
                    messages_lst=[remote_message],
                )
                remote_response = remote_response[0]['content']        # the response is a string

                (
                    parsed_plan_action_list, 
                    parsed_plan_action_valid, 
                    parsed_feedback
                    ) = self._extract_parallel_plan_from_assistant(
                    remote_response,
                    available_move,
                    current_planning_step=current_planning_step,
                    num_plan_moves_allowed=num_plan_moves_allowed,
                    allow_random_move=allow_random_move,
                )
                selected_action_string = [f"{include_offpolicy_move_tag[0]}{i}{include_offpolicy_move_tag[1]}" for i in parsed_plan_action_list]
                response_text = "Off-Policy Move: " + "\n".join(selected_action_string)
                response_ids = await self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer(response_text)['input_ids'],
                )
                parsed_feedback = parsed_feedback.replace("random move", "off-policy move")
                parsed_plan_action_valid = 0.

            else:
                # switch to random move valid string as model response
                selected_action_string = [f"{include_offpolicy_move_tag[0]}{i}{include_offpolicy_move_tag[1]}" for i in parsed_plan_action_list]
                response_text = "Random Move: " + "\n".join(selected_action_string)
                response_ids = await self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer(response_text)['input_ids'],
                )
                response_ids += [self.tokenizer.eos_token_ids]   # add eos token to the end
                response_ids += self.tokenizer.encode("\n")   # add \n to align with template

        _ids += response_ids
        _input_plan_response_mask += [1] * len(response_ids)
        _input_plan_reward_lst += [0.] * len(response_ids)
        _distill_ids_lst.append(list(response_ids))
        _infer_message.append({"role": "assistant", "content": response_text})

        return (
            # inference ret
            response_text, 
            state_understanding_text, 
            _infer_message, 
            _ids[len(input_plan_ids):], 
            _input_plan_response_mask, 
            _input_plan_reward_lst, 
            metrics,
            # parse ret
            parsed_plan_action_list, 
            parsed_plan_action_valid, 
            parsed_feedback,
            # auxiliary
            _distill_ids_lst
            )

    async def _one_stage_policy_parallel_inference(
        self, 
        state, 
        input_plan_ids, 
        metrics, 
        sampling_params, 
        exploration_p = None, 
        available_move=None, 
        num_plan_moves_allowed=None,
        simulator=None,
        current_planning_step=None,
        allow_random_move=None,
        include_offpolicy_move_in_response=None,
        include_offpolicy_move_tag=None,
        extra_message=None,     # if we serve the checkpoint model to infer movement
        plan_query_prompt=None,
        validate_action=None,
    ):
        _infer_message = []
        _input_plan_response_mask = []
        _input_plan_reward_lst = []
        _infer_ids = []
        _ids = input_plan_ids.copy()

        # step 1:
        # print(self.tokenizer.decode(_ids[:2000]))

        _infer_message.append({"role": "user", "content": plan_query_prompt})
        _query_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                [{"role": "user", "content": plan_query_prompt}], add_generation_prompt=True, tokenize=True
            ),
        )
        _query_ids = _query_ids[len(self.system_prompt):]
        _ids += _query_ids
        _input_plan_response_mask += [0] * len(_query_ids)
        _input_plan_reward_lst += [0.] * len(_query_ids)


        with simple_timer("generate_sequences", metrics):
            plan_step_request_id = f"{uuid4().hex}_stage_1"
            sampling_params['max_tokens'] = self.each_response_length
            sampling_params['stop'] = [self.tokenizer.eos_token]
            response_ids = await self.server_manager.generate(
                request_id=plan_step_request_id, prompt_ids=_ids, sampling_params=sampling_params
            )

            _ids += response_ids
            _input_plan_response_mask += [1] * len(response_ids)
            _input_plan_reward_lst += [0.] * len(response_ids)
            # _infer_ids.append(response_ids)
        
        response_text = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(response_ids, skip_special_tokens=True)
        )
        _infer_message.append({"role": "assistant", "content": response_text})

        # parse decision
        (
            parsed_plan_action_list, 
            parsed_plan_action_valid, 
            parsed_feedback
            ) = self._extract_parallel_plan_from_assistant(
                response_text,
                available_move,
                current_planning_step=current_planning_step,
                num_plan_moves_allowed=num_plan_moves_allowed,
                allow_random_move=allow_random_move,
                validate_action=validate_action,
            )

        return (
            # inference ret
            response_text, 
            _infer_message, 
            _ids[len(input_plan_ids):], 
            _input_plan_response_mask, 
            _input_plan_reward_lst, 
            metrics,
            # parse ret
            parsed_plan_action_list, 
            parsed_plan_action_valid, 
            parsed_feedback,
            )

    async def _one_stage_policy_final_inference(
        self, 
        state, 
        input_plan_ids, 
        metrics, 
        sampling_params, 
        exploration_p = 0.,
        allow_random_move=None,
        include_offpolicy_move_in_response=None,
        include_offpolicy_move_tag=None,
        available_move=None,
        extra_message=None,
        validate_action=True,
    ):
        _infer_message = []
        _input_plan_response_mask = []
        _input_plan_reward_lst = []

        # step 1
        with simple_timer("generate_sequences", metrics):
            plan_step_request_id = f"{uuid4().hex}_stage_1"
            sampling_params['max_tokens'] = self.each_response_length
            sampling_params['stop'] = [self.tokenizer.eos_token]
            response_ids = await self.server_manager.generate(
                request_id=plan_step_request_id, prompt_ids=input_plan_ids, sampling_params=sampling_params
            )
            
        response_text = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(response_ids, skip_special_tokens=True)
        )

        input_plan_ids += response_ids
        _input_plan_response_mask += [1] * len(response_ids)
        _input_plan_reward_lst += [0.] * len(response_ids)
        _infer_message.append({"role": "assistant", "content": response_text})

        parsed_action, parsed_action_valid, parsed_feedback = self._extract_action_from_assistant(response_text, validate_action=validate_action)

        return (
            response_text, 
            _infer_message, 
            input_plan_ids, 
            _input_plan_response_mask, 
            _input_plan_reward_lst, 
            metrics,
            # parse ret
            parsed_action, 
            parsed_action_valid, 
            parsed_feedback,
            )



    async def run_parallel_planning(self, simulator, current_plan_message, sampling_params, metrics):
        """
        run parallel planning for the given simulator and state message
        Here the max_plan_traj_n is the max number of candidate move for each state
        the max_plan_horizon is the turn now as we proceed the planning in a different dimension then previous sequential search

                             |--> a_1 --> s_3
                             |
            |--> a_1 --> s_2-|--> a_2 --> s_3
            |                |
        s_1-|--> a_2 --> s_2 |--> a_3 --> s_3
            |
            |--> a_3 --> s_2 ...
        
                Turn 1              Turn 2
        
        we use this simple version for now:

            |--> a_1 --> s_2-|--> a_1 --> s_3
            |                
        s_1-|--> a_2 --> s_2 |--> a_2 --> s_3
            |
            |--> a_3 --> s_2 |--> a_3 --> s_3
        
                Turn 1              Turn 2


        need to return:
        1. a final planned action
        2. a updated plan message
        3. planning metrics
        """

        plan_step = 1  # the current planning horizon, this is also the plan_turn
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
            "num_diverse_move": []
        }

        root_planning_state = simulator.observe()
        root_planning_stats_dict = simulator.get_key_stats()
        current_state_stats_dict = simulator.get_key_stats()

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
        plan_distill_mask = [0] * len(_query_ids)   # this record the root state understanding and the updated state understanding \hat{Q}

        plan_available_actions = simulator.legal_moves_list     # all the available planning actions
        # simulation_history = [[f"State: {simulator.observe()}"]]
        simulation_history = []
        expanded_traj_n = 0

        plan_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    plan_message, add_generation_prompt=False, tokenize=True
                ),
            )
        
        if self.distillation_enable:
            _distill_plan_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    plan_message, add_generation_prompt=False, tokenize=True
                ),
            )
            distill_plan_ids = [_distill_plan_ids]      # student response, need to truncate at the end
            distill_plan_teacher_ids = [_distill_plan_ids]    # teacher response, need to truncate at the end
            # distill_expansion_position_idx = None
            distill_plan_response_mask = [0] * len(_query_ids)  # student response mask, no need to truncate
            distill_plan_teacher_response_mask = [0] * len(_query_ids)    # teacher response mask, no need to truncate

        plan_early_stop = False
        idx = 0
        plan_response_length = []
        group_simulation_env = None
        group_simulation_done = None
        num_diverse_move = 0
        ever_win = 0
        while True:
            idx += 1

            if idx == 1:   # the first turn, multiple moves are generated
                _num_plan_moves_allowed = self.max_plan_traj_n

                # two stage value inference
                (
                    response_text, 
                    state_understanding_text, 
                    infer_message, 
                    new_plan_ids, 
                    infer_response_mask, 
                    infer_reward_lst, 
                    metrics,
                    plan_action_list,
                    plan_action_valid,
                    feedback,
                    distill_plan_ids_lst,
                    ) = await self._two_stage_value_parallel_inference(
                        simulator.observe(), 
                        plan_ids, 
                        metrics, 
                        sampling_params,
                        move_decision_prompt=self.prompt_set['state_move_query_prompt'].format(num_actions=_num_plan_moves_allowed, available_move=simulator.legal_moves_string),
                        value_decision_prompt=self.prompt_set['state_value_query_prompt'].format(state=simulator.observe()),
                        available_move=simulator.legal_moves_list,
                        num_plan_moves_allowed=_num_plan_moves_allowed,
                        simulator=simulator,
                        current_planning_step=plan_step,
                        allow_random_move=self.random_move_when_invalid_plan_expansion,
                        include_offpolicy_move_in_response=self.include_random_expansion_in_response,
                        include_offpolicy_move_tag=["<move>", "</move>"],
                        extra_message=plan_message,
                    )
                plan_ids += new_plan_ids
                plan_message.extend(infer_message)
                plan_response_mask.extend(infer_response_mask)
                plan_reward_lst.extend(infer_reward_lst)
                plan_response_length.append(len(infer_response_mask)) 
                if self.distillation_enable:
                    assert len(distill_plan_ids_lst) == 4
                    if self.distillation_mode == "supervised-sd":    # for ssd mode, student first planning response will be replaced by the summary text, so set to None for now
                        distill_plan_ids_lst[1] = None
                        # distill_expansion_position_idx = len(distill_plan_ids) + 1
                        distill_plan_ids.extend(distill_plan_ids_lst)
                        distill_plan_teacher_response_mask += [0] * len(infer_response_mask)
                    elif self.distillation_mode == "op-sd":    # for op-sd mode, teacher's summary response will be replaced by the first planning response, so record the first planning response
                        _distill_root_state_response = distill_plan_ids_lst[1]     # for later replacement at summary phase
                        distill_plan_teacher_ids.extend(distill_plan_ids_lst)
                        distill_plan_teacher_response_mask += [0] * len(infer_response_mask)

                        distill_plan_response_mask.extend([0] * len(distill_plan_ids_lst[0]))
                        distill_plan_response_mask.extend([1] * len(distill_plan_ids_lst[1]))    # unmask the root state understanding
                        distill_plan_response_mask.extend([0] * len(distill_plan_ids_lst[2]))
                        distill_plan_response_mask.extend([0] * len(distill_plan_ids_lst[3]))

                
                expanded_traj_n = len(plan_action_list)   # number of actual traj expanded

                group_simulation_env = []
                group_simulation_done = []
                group_simulation_status = []
                for i in range(expanded_traj_n):
                    _env = await get_async_env(self.env_config)
                    _ = await _env.reset_state(root_planning_stats_dict)
                    group_simulation_env.append(_env)
                    group_simulation_done.append(False)
                    group_simulation_status.append("")
                
                simulation_history.extend([[f"State: {simulator.observe()}"] for _ in range(len(plan_action_list))])    # taking record
                planning_metrics['planning_valid_action'].append(plan_action_valid)
                # if len(feedback) > 0:
                    # plan_message.append({"role": "user", "content": feedback})
                feedback_ids = await self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": feedback}], add_generation_prompt=False, tokenize=True
                    ),
                )
                feedback_ids = feedback_ids[len(self.system_prompt):]
                plan_ids += feedback_ids
                plan_response_mask += [0] * len(feedback_ids)
                plan_reward_lst += [0.] * len(feedback_ids)
                plan_message.append({"role": "user", "content": feedback})
                if self.distillation_enable:
                    if self.distillation_mode == "supervised-sd":
                        distill_plan_ids.append(feedback_ids)
                        distill_plan_teacher_response_mask += [0] * len(feedback_ids)
                    elif self.distillation_mode == "op-sd":
                        distill_plan_teacher_ids.append(feedback_ids)
                        distill_plan_response_mask += [0] * len(feedback_ids)
                        distill_plan_teacher_response_mask += [0] * len(feedback_ids)


                # Flatten plan_action_list (which contains lists and possibly None) before creating set
                flattened_actions = []
                for action in plan_action_list:
                    if action is None:
                        continue
                    if isinstance(action, list):
                        flattened_actions.extend(action)
                    else:
                        flattened_actions.append(action)
                num_diverse_move = len(set(flattened_actions))

            else:
                _num_plan_moves_allowed = 1
                # sequential request for each expanded traj if the simulator is not done
                assert not all(group_simulation_done), "The simulator should not be done before the first expansion"
                plan_action_list = []
                parallel_plan_valid = []
                parallel_plan_feedback_ids = []

                # for storing each decision response
                new_plan_ids = []
                new_infer_response_mask = []
                new_infer_reward_lst = []
                new_infer_message = []

                for i in range(expanded_traj_n):
                    if not group_simulation_done[i]:
                        (
                            _response_text, 
                            _state_understanding_text, 
                            _infer_message, 
                            _new_plan_ids, 
                            _infer_response_mask, 
                            _infer_reward_lst, 
                            metrics,
                            _plan_action_list,
                            _plan_action_valid,
                            _feedback,
                            _distill_ids_lst,
                        ) = await self._two_stage_value_parallel_inference(
                            group_simulation_env[i].observe(), 
                            plan_ids, 
                            metrics, 
                            sampling_params,
                            move_decision_prompt=self.prompt_set['state_move_query_prompt'].format(num_actions=_num_plan_moves_allowed, available_move=group_simulation_env[i].legal_moves_string),
                            value_decision_prompt=self.prompt_set['state_value_query_prompt'].format(state=group_simulation_env[i].observe()),
                            available_move=group_simulation_env[i].legal_moves_list,
                            num_plan_moves_allowed=_num_plan_moves_allowed,
                            simulator=group_simulation_env[i],
                            current_planning_step=plan_step,
                            allow_random_move=self.random_move_when_invalid_plan_simulation,
                            include_offpolicy_move_in_response=self.include_random_simulation_in_response,
                            include_offpolicy_move_tag=["<move>", "</move>"],
                            exploration_p=1.0 if self.full_exploration_in_simulation else self.current_exploration_plan,
                            extra_message=plan_message,
                        )

                        new_plan_ids.append(_new_plan_ids)
                        new_infer_response_mask.append(_infer_response_mask)
                        new_infer_reward_lst.append(_infer_reward_lst)
                        new_infer_message.append(_infer_message)

                        if len(_plan_action_list) == 0:
                            _plan_action_list = [None]
                        
                        plan_action_list.append(_plan_action_list[0])
                        parallel_plan_valid.append(_plan_action_valid)
                        _feedback_ids = await self.loop.run_in_executor(
                            None,
                            lambda: self.tokenizer.apply_chat_template(
                                [{"role": "user", "content": _feedback}], add_generation_prompt=False, tokenize=True
                            ),
                        )
                        _feedback_ids = _feedback_ids[len(self.system_prompt):]
                        parallel_plan_feedback_ids.append(_feedback_ids)

                    else:
                        plan_action_list.append(None)
                
                plan_action_valid = np.mean(parallel_plan_valid)
                if self.include_turn2_simulation_in_response:
                    for i in range(len(new_plan_ids)):
                        plan_ids += new_plan_ids[i]
                        plan_ids += parallel_plan_feedback_ids[i]
                        plan_response_mask += new_infer_response_mask[i] + [0] *  len(parallel_plan_feedback_ids[i])
                        plan_reward_lst += new_infer_reward_lst[i] + [0.] *  len(parallel_plan_feedback_ids[i])
                        if self.distillation_enable:
                            if self.distillation_mode == "supervised-sd":
                                distill_plan_ids.append(new_plan_ids[i])
                                distill_plan_teacher_response_mask += [0] * len(new_infer_response_mask[i]) + [0] *  len(parallel_plan_feedback_ids[i])
                            elif self.distillation_mode == "op-sd":
                                distill_plan_teacher_ids.append(new_plan_ids[i])
                                distill_plan_response_mask += [0] * len(new_infer_response_mask[i]) + [0] *  len(parallel_plan_feedback_ids[i])
                                distill_plan_teacher_response_mask += [0] * len(new_infer_response_mask[i]) + [0] *  len(parallel_plan_feedback_ids[i])

                    plan_message.extend(new_infer_message)


            # planning_metrics['num_diverse_move'].append(len(set(flattened_actions)))
            # ====================== Execute parallel search Decision ====================================================#

            if len(plan_action_list) > 0:   # found planned move candidate
                one_step_simulation_message = ""
                for move_idx, plan_move_candidate in enumerate(plan_action_list):
                    if group_simulation_done[move_idx]:
                        _status = group_simulation_status[move_idx]
                        one_step_simulation_message += f"Traj {move_idx+1}:\n{_status}\n\n"
                        continue
                        
                    if not group_simulation_done[move_idx] and plan_move_candidate is None:
                        # invalid action, terminate
                        env_feedback = "The action is invalid. End current planning turn."
                        group_simulation_done[move_idx] = True
                        _move_results = f"Traj {move_idx+1}:\nThe action is invalid. End current planning turn."
                        one_step_simulation_message += _move_results + "\n\n"
                        group_simulation_status[move_idx] = "Ended due to invalid move."
                        continue

                    # simulator.reset_state(current_state_stats_dict)   # reset to previous state and execute the move
                    _simulator = group_simulation_env[move_idx]
                    next_state, reward, done, env_info = await _simulator.step(plan_move_candidate)

                    simulation_history[move_idx].append(f"Move: {plan_move_candidate}")
                    # if not done:
                        # simulation_history[move_idx].append(f"Opponent's move: {env_info['oppo_move']}")
                    if 'oppo_move' in env_info:
                        simulation_history[move_idx].append(f"Opponent's move: {env_info['oppo_move']}")
                    simulation_history[move_idx].append(f"Reward: {reward}")
                    simulation_history[move_idx].append(f"State: {next_state}")
                    simulation_history[move_idx].append(f"Game Terminated: {done}")
                    
                    if done or (plan_step >= self.max_plan_horizon):
                        if done:
                            _game_status = _simulator.check_results()
                            env_success = (_game_status == "win")
                            env_lose = (_game_status == "lose")
                            env_tie = (_game_status == "tie")
                            if env_success:
                                env_feedback = self.prompt_set['plan_success_prompt']
                                _status = "Won"
                                ever_win = 1
                            elif env_lose:
                                env_feedback = self.prompt_set['plan_fail_prompt'].format(state=next_state)
                                _status = "Lost"
                            elif env_tie:
                                env_feedback = self.prompt_set['plan_tie_prompt'].format(state=next_state)
                                _status = "Tied"
                            else:
                                raise NotImplementedError(f"Unknown game status: {_game_status}")

                            if 'oppo_move' in env_info:
                                env_feedback = f"The opponent played the move {env_info['oppo_move']}.\n" + env_feedback
                            group_simulation_status[move_idx] = _status
                        else:
                            env_feedback = f"The opponent played the move {env_info['oppo_move']}.\n" + f"The next state of current simulation: {next_state}"
                            group_simulation_status[move_idx] = "Ended due to reaching maximum planning steps."
                        group_simulation_done[move_idx] = True
                    else:
                        env_feedback = f"The opponent played the move {env_info['oppo_move']}.\n" + f"The next state of current simulation: {next_state}"
                    
                    _move_results = f"Traj {move_idx+1}:\nSubsequent result of move {plan_move_candidate}: \n{env_feedback}"
                    one_step_simulation_message += _move_results + "\n\n"
                
                # if plan_message[-1]['role'] == 'user':
                    # plan_message[-1]['content'] += f"\n{one_step_simulation_message}"
                # else:
                plan_message.append({"role": "user", "content": one_step_simulation_message})

                plan_step += 1

                if idx > 1 and not self.include_turn2_simulation_in_response:
                    pass
                else:
                    # encode env feedback
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
                    if self.distillation_enable:
                        if self.distillation_mode == "supervised-sd":
                            distill_plan_ids.append(env_feedback_ids)
                            distill_plan_teacher_response_mask += [0] * len(env_feedback_ids)
                        elif self.distillation_mode == "op-sd":
                            distill_plan_teacher_ids.append(env_feedback_ids)
                            distill_plan_response_mask += [0] * len(env_feedback_ids)
                            distill_plan_teacher_response_mask += [0] * len(env_feedback_ids)

                if plan_step > self.max_plan_horizon:
                    break
                    
                if all(group_simulation_done):   # all simulator has ended
                    break
            
            else:
                # no action detected, jump to summarize
                break


        if not self.independent_summary:
            if self.include_simulation_in_summary:
                history_string = ""
                for i, lst in enumerate(simulation_history):
                    _traj = ", ".join(lst)
                    _traj = f"{i+1}: " + _traj
                    history_string += _traj + "\n"
                plan_summarize_prompt = self.prompt_set['independent_summary_prompt'].format(
                    history=history_string
                )
            else:
                plan_summarize_prompt = self.prompt_set['plan_action_query_prompt']

            plan_message.append({"role": "user", "content": plan_summarize_prompt})


            _plan_summarize_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": plan_summarize_prompt}], add_generation_prompt=False, tokenize=True
                ),
            )
            _plan_summarize_ids = _plan_summarize_ids[len(self.system_prompt):]
            plan_ids += _plan_summarize_ids
            plan_response_mask += [0] * len(_plan_summarize_ids)
            plan_reward_lst += [0.] * len(_plan_summarize_ids)
            if self.distillation_enable:
                if self.distillation_mode == "supervised-sd":
                    distill_plan_ids.append(_plan_summarize_ids)
                    distill_plan_teacher_response_mask += [0] * len(_plan_summarize_ids)
                elif self.distillation_mode == "op-sd":
                    distill_plan_teacher_ids.append(_plan_summarize_ids)
                    distill_plan_response_mask += [0] * len(_plan_summarize_ids)
                    distill_plan_teacher_response_mask += [0] * len(_plan_summarize_ids)    # unmask the root state understanding

            (
                summarize_response_text, 
                summarize_state_understanding_text, 
                infer_message, 
                plan_ids, 
                infer_response_mask, 
                infer_reward_lst, 
                metrics,
                final_action,
                final_action_valid,
                final_action_feedback,
                _distill_ids_lst,
                ) = await self._two_stage_value_inference(
                    root_planning_state, 
                    plan_ids, 
                    metrics, 
                    sampling_params,
                    move_decision_prompt=self.prompt_set['state_answer_query_prompt'],
                    value_decision_prompt=self.prompt_set['simulation_state_value_query_prompt'].format(state=root_planning_state),
                    allow_random_move=self.random_move_when_invalid_act,
                    include_offpolicy_move_in_response=self.include_random_act_in_response,
                    include_offpolicy_move_tag=["<answer>", "</answer>"],
                    available_move=simulator.legal_moves_string,
                    extra_message=plan_message,
                )
            plan_message.extend(infer_message)
            plan_response_mask.extend(infer_response_mask)
            plan_reward_lst.extend(infer_reward_lst)
            if self.distillation_enable:
                if self.distillation_mode == "supervised-sd":
                    distill_plan_ids.extend(_distill_ids_lst)
                    distill_summary_ids = _distill_ids_lst[1]
                    # distill_plan_ids[distill_expansion_position_idx] = _distill_ids_lst[1]    # replace the expansion position with the summarize response
                    assert len(_distill_ids_lst) == 4
                    distill_plan_teacher_response_mask += [0] * len(_distill_ids_lst[0])
                    distill_plan_teacher_response_mask += [1] * len(_distill_ids_lst[1])
                    distill_plan_teacher_response_mask += [0] * len(_distill_ids_lst[2])
                    distill_plan_teacher_response_mask += [0] * len(_distill_ids_lst[3])
                elif self.distillation_mode == "op-sd":
                    assert len(_distill_ids_lst) == 4
                    distill_plan_teacher_ids.append(_distill_ids_lst[0])
                    distill_plan_teacher_ids.append(_distill_root_state_response)     # replace with the root state understanding
                    distill_plan_teacher_ids.append(_distill_ids_lst[2])
                    distill_plan_teacher_ids.append(_distill_ids_lst[3])

                    distill_plan_teacher_response_mask += [0] * len(_distill_ids_lst[0])
                    distill_plan_teacher_response_mask += [1] * len(_distill_root_state_response)     # replace with the root state understanding
                    distill_plan_teacher_response_mask += [0] * len(_distill_ids_lst[2])
                    distill_plan_teacher_response_mask += [0] * len(_distill_ids_lst[3])

                    distill_plan_response_mask += [0] * len(_distill_ids_lst[0])
                    distill_plan_response_mask += [0] * len(_distill_ids_lst[1])
                    distill_plan_response_mask += [0] * len(_distill_ids_lst[2])
                    distill_plan_response_mask += [0] * len(_distill_ids_lst[3])

            summary_response_length = len(infer_response_mask)
            mean_plan_response_length = np.mean(plan_response_length)
            
        else:   # independent summary
            raise NotImplementedError("Independent summary is not supported for two stage value inference")


        # (final_action, final_action_valid) = self._extract_summarize_from_assistant(summarize_response_text)

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

        # process distillation ids
        if self.distillation_enable:
            instruction_ids_length = len(plan_ids) - len(plan_response_mask)
            if self.distillation_mode == "supervised-sd":
                final_distill_plan_ids = []
                final_distill_plan_mask = []
                for i in distill_plan_ids:
                    if i is None:
                        final_distill_plan_ids.extend(distill_summary_ids)
                        final_distill_plan_mask += [1] * len(distill_summary_ids)
                    else:
                        final_distill_plan_ids.extend(i)
                        final_distill_plan_mask += [0] * len(i)

                final_distill_plan_ids = final_distill_plan_ids[instruction_ids_length:]
                final_distill_plan_mask = final_distill_plan_mask[instruction_ids_length:]  # here the mask contain system prompt so need truncation
                final_distill_plan_teacher_mask = distill_plan_teacher_response_mask
                final_distill_plan_teacher_ids = []   # this is the same as the original teacher response ids
            elif self.distillation_mode == "op-sd":
                final_distill_plan_teacher_ids = []
                for i in distill_plan_teacher_ids: 
                    final_distill_plan_teacher_ids.extend(i)
                
                final_distill_plan_teacher_ids = final_distill_plan_teacher_ids[instruction_ids_length:]
                final_distill_plan_mask = distill_plan_response_mask
                final_distill_plan_ids = []          # this is the same as the original response ids
                final_distill_plan_teacher_mask = distill_plan_teacher_response_mask

        else:
            final_distill_plan_ids = []
            final_distill_plan_mask = []
        
        plan_ids = plan_ids[-len(plan_response_mask):]

        # kill env - cleanup simulation environments to release memory
        del group_simulation_env

        return (
            final_action,
            final_action_valid,
            final_action_feedback, 
            plan_message[1:], 
            plan_ids, 
            plan_response_mask, 
            planning_metrics, 
            metrics, 
            plan_reward_lst, 
            summarize_state_understanding_text, 
            summarize_response_text, 
            plan_early_stop, 
            idx, 
            num_diverse_move, 
            ever_win,
            final_distill_plan_ids,
            final_distill_plan_mask,
            final_distill_plan_teacher_ids,
            final_distill_plan_teacher_mask,
            )

    async def run_policy_centric_parallel_planning(self, simulator, current_plan_message, sampling_params, metrics):
        # print("Running policy centric parallel planning")
        plan_step = 1  # the current planning horizon, this is also the plan_turn
    
        root_planning_state = simulator.observe()
        root_planning_stats_dict = simulator.get_key_stats()
        root_legal_action_string = simulator.legal_moves_string
        # current_state_stats_dict = simulator.get_key_stats()
        root_task_goal = simulator.task_goal if hasattr(simulator, 'task_goal') else ""

        planning_metrics = {
            'planning_valid_action': [],
        }

        # plan_message = [{"role": "system", "content": planning_instruction}, {"role":"user", "content": planning_user_prompt}]
        plan_message = copy.deepcopy(current_plan_message)      #+ [{"role":"user", "content": planning_user_prompt}]


        plan_response_mask = [] #[0] * len(_query_ids)
        plan_reward_lst = [] #[0.] * len(_query_ids)

        plan_available_actions = simulator.legal_moves_list     # all the available planning actions
        # simulation_history = [[f"State: {simulator.observe()}"]]
        simulation_history = []
        expanded_traj_n = 0

        _ = await simulator.close()

        plan_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    plan_message, add_generation_prompt=False, tokenize=True
                ),
            )

        plan_early_stop = False
        idx = 0
        plan_response_length = []
        group_simulation_env = None
        group_simulation_done = None
        num_diverse_move = 0
        ever_win = 0
        while True:
            idx += 1

            if idx == 1:   # the first turn, multiple moves are generated
                _num_plan_moves_allowed = self.max_plan_traj_n
                _plan_query_prompt = self.prompt_set['plan_query_prompt'].format(
                    turn_idx=idx,
                    step_idx=plan_step,
                    state=root_planning_state,
                    available_move=root_legal_action_string,
                    turn_left=self.max_plan_traj_n - idx,
                    max_step=self.max_plan_horizon - plan_step + 1,
                    extra_info="the simulation starts",
                    num_actions=_num_plan_moves_allowed,
                    task_description=root_task_goal,
                )

                # one stage value inference
                (
                    response_text, 
                    infer_message, 
                    new_plan_ids, 
                    infer_response_mask, 
                    infer_reward_lst, 
                    metrics,
                    plan_action_list,
                    plan_action_valid,
                    feedback,
                    ) = await self._one_stage_policy_parallel_inference(
                        root_planning_state, 
                        plan_ids, 
                        metrics, 
                        sampling_params,
                        available_move=root_legal_action_string,
                        num_plan_moves_allowed=_num_plan_moves_allowed,
                        simulator=None,
                        current_planning_step=plan_step,
                        allow_random_move=self.random_move_when_invalid_plan_expansion,
                        include_offpolicy_move_in_response=self.include_random_expansion_in_response,
                        include_offpolicy_move_tag=["<move>", "</move>"],
                        extra_message=plan_message,
                        plan_query_prompt=_plan_query_prompt,
                        validate_action=self.validate_action,
                    )
                plan_ids += new_plan_ids
                plan_message.extend(infer_message)
                plan_response_mask.extend(infer_response_mask)
                plan_reward_lst.extend(infer_reward_lst)
                plan_response_length.append(len(infer_response_mask)) 

                expanded_traj_n = len(plan_action_list)   # number of actual traj expanded

                group_simulation_env = []
                group_simulation_done = []
                group_simulation_status = []
                for i in range(expanded_traj_n):
                    _env = await get_async_env(self.env_config)
                    _ = await _env.reset_state(root_planning_stats_dict)
                    group_simulation_env.append(_env)
                    group_simulation_done.append(False)
                    group_simulation_status.append("")
                
                simulation_history.extend([[f"State: {root_planning_state}"] for _ in range(len(plan_action_list))])    # taking record
                # if len(feedback) > 0:
                    # plan_message.append({"role": "user", "content": feedback})
                if len(feedback) > 0:
                    feedback_ids = await self.loop.run_in_executor(
                        None,
                        lambda: self.tokenizer.apply_chat_template(
                            [{"role": "user", "content": feedback}], add_generation_prompt=False, tokenize=True
                        ),
                    )
                    feedback_ids = feedback_ids[len(self.system_prompt):]
                    plan_ids += feedback_ids
                    plan_response_mask += [0] * len(feedback_ids)
                    plan_reward_lst += [0.] * len(feedback_ids)
                    plan_message.append({"role": "user", "content": feedback})

                planning_metrics['planning_valid_action'].append(plan_action_valid)

                # Flatten plan_action_list (which contains lists and possibly None) before creating set
                flattened_actions = []
                for action in plan_action_list:
                    if action is None:
                        continue
                    if isinstance(action, list):
                        flattened_actions.extend(action)
                    else:
                        flattened_actions.append(action)
                num_diverse_move = len(set(flattened_actions))

            else:
                _num_plan_moves_allowed = 1
                # sequential request for each expanded traj if the simulator is not done
                assert not all(group_simulation_done), "The simulator should not be done before the first expansion"
                plan_action_list = []
                parallel_plan_valid = []
                parallel_plan_feedback_ids = []

                # for storing each decision response
                new_plan_ids = []
                new_infer_response_mask = []
                new_infer_reward_lst = []
                new_infer_message = []

                for i in range(expanded_traj_n):
                    if not group_simulation_done[i]:
                        _plan_query_prompt = self.prompt_set['plan_query_prompt'].format(
                            turn_idx=idx,
                            step_idx=plan_step,
                            state=group_simulation_env[i].observe(),
                            available_move=group_simulation_env[i].legal_moves_list,
                            turn_left=self.max_plan_traj_n - idx,
                            max_step=self.max_plan_horizon - plan_step + 1,
                            extra_info="",
                            num_actions=_num_plan_moves_allowed,
                            task_description=group_simulation_env[i].task_goal if hasattr(group_simulation_env[i], 'task_goal') else "",
                        )
                        (
                            _response_text, 
                            _infer_message, 
                            _new_plan_ids, 
                            _infer_response_mask, 
                            _infer_reward_lst, 
                            metrics,
                            _plan_action_list,
                            _plan_action_valid,
                            _feedback,
                        ) = await self._one_stage_policy_parallel_inference(
                            group_simulation_env[i].observe(), 
                            plan_ids, 
                            metrics, 
                            sampling_params,
                            available_move=group_simulation_env[i].legal_moves_list,
                            num_plan_moves_allowed=_num_plan_moves_allowed,
                            simulator=group_simulation_env[i],
                            current_planning_step=plan_step,
                            allow_random_move=self.random_move_when_invalid_plan_simulation,
                            include_offpolicy_move_in_response=self.include_random_simulation_in_response,
                            include_offpolicy_move_tag=["<move>", "</move>"],
                            exploration_p=1.0 if self.full_exploration_in_simulation else self.current_exploration_plan,
                            extra_message=plan_message,
                            validate_action=self.validate_action,
                            plan_query_prompt=f"Traj: {i+1}\n" + _plan_query_prompt,
                        )

                        new_plan_ids.append(_new_plan_ids)
                        new_infer_response_mask.append(_infer_response_mask)
                        new_infer_reward_lst.append(_infer_reward_lst)
                        new_infer_message.extend(_infer_message)

                        if len(_plan_action_list) == 0:
                            _plan_action_list = [None]
                        
                        plan_action_list.append(_plan_action_list[0])
                        parallel_plan_valid.append(_plan_action_valid)
                        _feedback_ids = await self.loop.run_in_executor(
                            None,
                            lambda: self.tokenizer.apply_chat_template(
                                [{"role": "user", "content": _feedback}], add_generation_prompt=False, tokenize=True
                            ),
                        )
                        _feedback_ids = _feedback_ids[len(self.system_prompt):]
                        parallel_plan_feedback_ids.append(_feedback_ids)

                    else:
                        plan_action_list.append(None)
                
                plan_action_valid = np.mean(parallel_plan_valid)
                if self.include_turn2_simulation_in_response:
                    for i in range(len(new_plan_ids)):
                        plan_ids += new_plan_ids[i]
                        plan_ids += parallel_plan_feedback_ids[i]
                        plan_response_mask += new_infer_response_mask[i] + [0] *  len(parallel_plan_feedback_ids[i])
                        plan_reward_lst += new_infer_reward_lst[i] + [0.] *  len(parallel_plan_feedback_ids[i])
                    plan_message.extend(new_infer_message)


            # planning_metrics['num_diverse_move'].append(len(set(flattened_actions)))
            # ====================== Execute parallel search Decision ====================================================#

            if len(plan_action_list) > 0:   # found planned move candidate
                one_step_simulation_message = ""
                for move_idx, plan_move_candidate in enumerate(plan_action_list):
                    if group_simulation_done[move_idx]:
                        _status = group_simulation_status[move_idx]
                        one_step_simulation_message += f"Traj {move_idx+1}:\n{_status}\n\n"
                        continue
                        
                    if not group_simulation_done[move_idx] and plan_move_candidate is None:
                        # invalid action, terminate
                        env_feedback = "The action is invalid. End current planning turn."
                        group_simulation_done[move_idx] = True
                        _move_results = f"Traj {move_idx+1}:\nThe action is invalid. End current planning turn."
                        one_step_simulation_message += _move_results + "\n\n"
                        group_simulation_status[move_idx] = "Ended due to invalid move."
                        continue

                    # simulator.reset_state(current_state_stats_dict)   # reset to previous state and execute the move
                    _simulator = group_simulation_env[move_idx]
                    next_state, reward, done, env_info = await _simulator.step(plan_move_candidate)
                    # print(f"Stepping env id: {_simulator.env_id}")

                    simulation_history[move_idx].append(f"Move: {plan_move_candidate}")
                    # if not done:
                        # simulation_history[move_idx].append(f"Opponent's move: {env_info['oppo_move']}")
                    if 'oppo_move' in env_info:
                        simulation_history[move_idx].append(f"Opponent's move: {env_info['oppo_move']}")
                    simulation_history[move_idx].append(f"Reward: {reward}")
                    simulation_history[move_idx].append(f"State: {next_state}")
                    simulation_history[move_idx].append(f"Game Terminated: {done}")
                    
                    if done or (plan_step >= self.max_plan_horizon):
                        if done:
                            _game_status = _simulator.check_results()
                            env_success = (_game_status == "win")
                            env_lose = (_game_status == "lose")
                            env_tie = (_game_status == "tie")
                            if env_success:
                                env_feedback = self.prompt_set['plan_success_prompt']
                                _status = "Won"
                                ever_win = 1
                            elif env_lose:
                                env_feedback = self.prompt_set['plan_fail_prompt'].format(state=next_state)
                                _status = "Lost"
                            elif env_tie:
                                env_feedback = self.prompt_set['plan_tie_prompt'].format(state=next_state)
                                _status = "Tied"
                            else:
                                raise NotImplementedError(f"Unknown game status: {_game_status}")
                            
                            if 'oppo_move' in env_info:
                                env_feedback = f"The opponent played the move {env_info['oppo_move']}.\n" + env_feedback
                            group_simulation_status[move_idx] = _status
                        else:
                            if "oppo_move" in env_info:
                                env_feedback = f"The opponent played the move {env_info['oppo_move']}.\n" + f"The next state of current simulation: {next_state}"
                            else:
                                env_feedback = f"The next state of current simulation: {next_state}"
                            
                            group_simulation_status[move_idx] = "Ended due to reaching maximum planning steps."
                        group_simulation_done[move_idx] = True
                        _close_result = _simulator.close()
                        if asyncio.iscoroutine(_close_result):
                            await _close_result
                    else:
                        if "oppo_move" in env_info:
                            env_feedback = f"The opponent played the move {env_info['oppo_move']}.\n" + f"The next state of current simulation: {next_state}"
                        else:
                            env_feedback = f"The next state of current simulation: {next_state}"
                    
                    _move_results = f"Traj {move_idx+1}:\nSubsequent result of move {plan_move_candidate}: \n{env_feedback}"
                    one_step_simulation_message += _move_results + "\n\n"
                
                # if plan_message[-1]['role'] == 'user':
                    # plan_message[-1]['content'] += f"\n{one_step_simulation_message}"
                # else:
                plan_message.append({"role": "user", "content": one_step_simulation_message})

                plan_step += 1

                if idx > 1 and not self.include_turn2_simulation_in_response:
                    pass
                else:
                    # encode env feedback
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

                if plan_step > self.max_plan_horizon:
                    break
                    
                if all(group_simulation_done):   # all simulator has ended
                    break
                    
                # print(f"@@ 3. plan message = {len(plan_message)}, plan ids = {len(plan_ids)}")
            
            else:
                # no action detected, jump to summarize
                break
        

        if not self.independent_summary:
            if self.include_simulation_in_summary:
                history_string = ""
                for i, lst in enumerate(simulation_history):
                    _traj = ", ".join(lst)
                    _traj = f"{i+1}: " + _traj
                    history_string += _traj + "\n"
                plan_summarize_prompt = self.prompt_set['independent_summary_prompt'].format(
                    history=history_string,
                    task_description=root_task_goal,
                    state=root_planning_state,
                )
            else:
                plan_summarize_prompt = self.prompt_set['plan_action_query_prompt']

            plan_message.append({"role": "user", "content": plan_summarize_prompt})


            _plan_summarize_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": plan_summarize_prompt}], add_generation_prompt=False, tokenize=True
                ),
            )
            _plan_summarize_ids = _plan_summarize_ids[len(self.system_prompt):]
            plan_ids += _plan_summarize_ids
            plan_response_mask += [0] * len(_plan_summarize_ids)
            plan_reward_lst += [0.] * len(_plan_summarize_ids)

            (
                summarize_response_text, 
                infer_message, 
                plan_ids, 
                infer_response_mask, 
                infer_reward_lst, 
                metrics,
                final_action,
                final_action_valid,
                final_action_feedback,
                ) = await self._one_stage_policy_final_inference(
                    root_planning_state, 
                    plan_ids, 
                    metrics, 
                    sampling_params,
                    allow_random_move=self.random_move_when_invalid_act,
                    include_offpolicy_move_in_response=self.include_random_act_in_response,
                    include_offpolicy_move_tag=["<answer>", "</answer>"],
                    available_move=plan_available_actions,
                    extra_message=plan_message,
                    validate_action=self.validate_action,
                )
            plan_message.extend(infer_message)
            plan_response_mask.extend(infer_response_mask)
            plan_reward_lst.extend(infer_reward_lst)

            summary_response_length = len(infer_response_mask)
            mean_plan_response_length = np.mean(plan_response_length)
            
        else:   # independent summary
            raise NotImplementedError("Independent summary is not supported for two stage value inference")


        # (final_action, final_action_valid) = self._extract_summarize_from_assistant(summarize_response_text)

        for env in group_simulation_env:
            _ = await env.close()
        
        del group_simulation_env

        plan_ids = plan_ids[-len(plan_response_mask):]

        # kill env - cleanup simulation environments to release memory
        # del group_simulation_env

        return (
            final_action,
            final_action_valid,
            final_action_feedback, 
            copy.deepcopy(plan_message[len(current_plan_message):]), 
            plan_ids, 
            plan_response_mask, 
            planning_metrics, 
            metrics, 
            plan_reward_lst, 
            "", 
            summarize_response_text, 
            plan_early_stop, 
            idx, 
            num_diverse_move, 
            ever_win
            )

    def _extract_parallel_plan_from_assistant(
            self, 
            response_text, 
            plan_available_actions, 
            current_planning_step, 
            num_plan_moves_allowed, 
            allow_random_move,
            validate_action=True,
        ):
        """
        the response text may contain multiple move tags, need to handle separately
        response_text: "<move>move1</move><move>move2</move><move>move3</move>"
        """

        # extract all move
        plan_moves_matches = self.plan_action_pattern.findall(response_text)
        if not plan_moves_matches:
            if not validate_action:
                return [response_text], 1.0, "Invalid move format"

            # if self.random_move_when_invalid_plan:
            if allow_random_move:
                extracted_moves = random.sample(plan_available_actions, min(num_plan_moves_allowed, len(plan_available_actions)))   # get max number of allowed random moves
                feedback = f"No valid plan move detected. Switch to random move: {extracted_moves}"
            else:
                extracted_moves = []
                feedback = "You did not output a move. End current planning turn."

            return extracted_moves, 0., feedback
        else:
            lower_letter_plan_available_actions = [i.lower() for i in plan_available_actions]
            extracted_moves = []
            for each_plan_move in plan_moves_matches:
                each_plan_move = each_plan_move.strip()
                if not validate_action:
                    extracted_moves.append(each_plan_move)
                    continue

                for special_token in self.special_token_list:
                    each_plan_move = each_plan_move.replace(special_token, "").strip()
                if each_plan_move.lower() in lower_letter_plan_available_actions:
                    position_idx = lower_letter_plan_available_actions.index(each_plan_move.lower())
                    valid_move = plan_available_actions[position_idx]
                    extracted_moves.append(valid_move)
                else:
                    # the move is not valid
                    # if self.random_move_when_invalid_plan:   # switch to random move
                    if allow_random_move:
                        # Filter out moves that are already in extracted_moves
                        available_moves = [move for move in plan_available_actions if move not in extracted_moves]
                        if available_moves:
                            random_move = random.choice(available_moves)
                        else:
                            # If all moves are already extracted, fall back to all available actions
                            random_move = random.choice(plan_available_actions)
                        extracted_moves.append(random_move)
                    else:
                        # not valid plan move is detected, skip
                        pass
            
            extracted_moves = extracted_moves[:num_plan_moves_allowed]   # only choose the first {num_plan_moves_allowed} moves
            feedback_text = f"Found {len(extracted_moves)} valid moves: {extracted_moves}"
        
        return extracted_moves, len(extracted_moves) > 0, feedback_text
                        
            
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

    def _extract_action_from_assistant(self, final_planned_action: str, validate_action: bool=True):
        """Extract action text from assistant messages using <action> tags."""
        # Collect assistant contents from the latest generation
        # print(f"\nllm response = {tmp_template}\n")
        if final_planned_action is None:
            if self.random_move_when_invalid_act:
                random_action = random.choice(self.available_actions)
                return random_action, 0., f"No output action detected. Swith to random move: {random_action}"
            else:
                return None, 0., "No output action detected. End current turn."

        match = self.answer_pattern.search(final_planned_action)
        if not match:
            # random_action = random.choice(self.available_actions)
            # return random_action, 0., f"No output action detected. Swith to random move: {random_action}"
            action_content = final_planned_action   # check if action tag is detected
        else:
            action_content = match.group(1).strip()
        
        if not validate_action:
            return action_content, 1.0, ""
        
        lower_letter_available_actions = [i.lower() for i in self.available_actions]

        for special_token in self.special_token_list:
            action_content = action_content.replace(special_token, "").strip()

        if action_content.lower() in lower_letter_available_actions:
            position_idx = lower_letter_available_actions.index(action_content.lower())

            return self.available_actions[position_idx], 1.0, f"Action {self.available_actions[position_idx]} is valid."
        else:
            if self.random_move_when_invalid_act:    # switch to random act when invalid act is detected
                random_action = random.choice(self.available_actions)
                return random_action, 0., f"Action {action_content.lower()[:20]} is not valid. Switch to random move: {random_action}"
            else:
                return None, 0., f"Action {action_content.lower()[:20]} is invalid. End current turn."

    def _format_env_feedback(self, state: Any, available_actions: list, step_idx, env_info: dict, task_goal: str) -> str:
        # try:
        # actions_text = ", ".join(list(available_actions)[:100])
        # except Exception:

        if "oppo_move" in env_info and "act_query_prompt_with_info" in self.prompt_set:
            next_query = self.prompt_set['act_query_prompt_with_info'].format(
                state=state,
                available_move=available_actions,
                turn_idx=step_idx,
                turn_left=self.max_steps-step_idx,
                oppo_move=env_info['oppo_move'],
                task_description=task_goal,
                )
        else:
            next_query = self.prompt_set['act_query_prompt'].format(
                state=state,
                available_move=available_actions,
                turn_idx=step_idx,
                turn_left=self.max_steps-step_idx,
                task_description=task_goal,
                )

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

    async def _process_value_sft_dataset(self, state_summary_pair: list[dict], env_success: bool) -> list[dict]:
        sft_dataset = []
        if not env_success:
            return sft_dataset

        for state_summary in state_summary_pair:
            _sft_prompt = [
                {"role": "user", "content": self.prompt_set['state_value_query_prompt'].format(
                    state=state_summary['state'],
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

            _sft_response_chat_str = state_summary['summary'] + self.tokenizer.eos_token

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

    async def _process_value_table_dataset(self, state_summary_pair: list[dict], env_success: bool) -> list[dict]:
        sft_dataset = []
        if not env_success:
            return sft_dataset

        for state_summary in state_summary_pair:
            # _sft_prompt = [
            #     {"role": "user", "content": self.prompt_set['state_value_query_prompt'].format(
            #         state=state_summary['state'],
            #     )},
            # ]
            # prompt_ids = await self.loop.run_in_executor(
            #     None,
            #     lambda: self.tokenizer.apply_chat_template(
            #         _sft_prompt, add_generation_prompt=True, tokenize=True, return_tensors="pt",
            #     )
            # )
            prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer(state_summary['state'], return_tensors="pt", add_special_tokens=False)['input_ids'],
            )
            prompt_ids = prompt_ids[0]
            prompt_attention_mask = torch.ones_like(prompt_ids, dtype=prompt_ids.dtype)

            _sft_response_chat_str = state_summary['summary'] + self.tokenizer.eos_token

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

    async def _process_supervised_kd_dataset(self, turn_message: list[dict], prompt_ids: list[int], response_ids: list[int]) -> list[dict]:
        """
        the original turn message is used to compute teacher logp at each summary phase
        the transformed turn message is used to compute student logp at each expansion phase with response replaced with summary
        for each text instance, also need to compute the distillation mask (1 for summary query and response for teacher, 1 for expansion query and response for student) 
        """
        teacher_message = []
        student_message = []

        async def _encode(message, remove_header):
            ret = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    message, add_generation_prompt=False, tokenize=True
                ),
            )
            if remove_header:
                return ret[len(self.system_prompt):]
            else:
                return ret

        # Initial prompt encoding
        teacher_prompt_ids = await _encode([{"role": "system", "content": self.prompt_set['system_prompt']}], remove_header=False)
        student_prompt_ids = await _encode([{"role": "system", "content": self.prompt_set['system_prompt']}], remove_header=False)

        teacher_highlight_message = []   # message that need to be unmasked in the distillation loss
        student_highlight_message = []   # message that need to be masked in the distillation loss
        
        for turn_idx, each_message in enumerate(turn_message):
            fillin_idx = None
            teacher_turn_message = []
            student_turn_message = []
            for idx, m in enumerate(each_message):

                if "label" in m:
                    # add the summarize response to the expansion phase
                    if m['label'] == "understand_response":
                        assert fillin_idx is None, "fillin_idx should be None"
                        fillin_idx = len(student_turn_message)
                    elif m['label'] == "summarize_understand_response":
                        assert fillin_idx is not None, "fillin_idx should not be None"
                        student_turn_message[fillin_idx]['content'] = m['content']

                        # unmask
                        teacher_highlight_message.append(m)
                        student_highlight_message.append(student_turn_message[fillin_idx])

                        fillin_idx = None
                    else:
                        pass
                    
                # Use a copy for the message we will replace content for, so turn_message is not mutated
                student_turn_message.append(copy.copy(m) if m.get("label", "") == "understand_response" else m)
                teacher_turn_message.append(m)
            
            student_message.extend(student_turn_message)
            teacher_message.extend(teacher_turn_message)

        # encode 
        teacher_turn_ids = await _encode(teacher_message, remove_header=True)
        student_turn_ids = await _encode(student_message, remove_header=True)

        teacher_response_mask = [0] * len(teacher_turn_ids)      # 1 for summarize understanding response, 0 for others
        student_response_mask = [0] * len(student_turn_ids)     # 1 for the replaced expansion understanding response, 0 for others

        async def _find_sublist_start(haystack: list, needle: list) -> int:
            """Return start index of first occurrence of needle in haystack, or -1 if not found."""
            n = len(needle)
            for i in range(len(haystack) - n + 1):
                if haystack[i : i + n] == needle:
                    return i
            return -1

        # unmask: set mask=1 for token positions that belong to the highlight (summarize) content
        for unmask_m in teacher_highlight_message:
            unmask_ids = await _encode([unmask_m], remove_header=True)
            start = await _find_sublist_start(teacher_turn_ids, unmask_ids)
            if start >= 0:
                for j in range(start, start + len(unmask_ids)):
                    teacher_response_mask[j] = 1

        for unmask_m in student_highlight_message:
            unmask_ids = await _encode([unmask_m], remove_header=True)
            start = await _find_sublist_start(student_turn_ids, unmask_ids)
            if start >= 0:
                for j in range(start, start + len(unmask_ids)):
                    student_response_mask[j] = 1

        # self.tokenizer.decode(teacher_turn_ids[teacher_response_mask.index(1):teacher_response_mask[teacher_response_mask.index(1):].index(0)+teacher_response_mask.index(1)])
        # self.tokenizer.decode(student_turn_ids[student_response_mask.index(1):student_response_mask[student_response_mask.index(1):].index(0)+student_response_mask.index(1)])
        # print(f"first 1 index = {teacher_response_mask.index(1)}, last 1 index = {teacher_response_mask[teacher_response_mask.index(1):].index(0)+teacher_response_mask.index(1)}")
        teacher_input_ids = teacher_prompt_ids + teacher_turn_ids
        student_input_ids = student_prompt_ids + student_turn_ids
        full_teacher_response_mask = [0] * len(teacher_prompt_ids) + teacher_response_mask
        full_student_response_mask = [0] * len(student_prompt_ids) + student_response_mask

        if len(teacher_input_ids) != len(prompt_ids) + len(response_ids):
            import pdb;pdb.set_trace()
        assert len(teacher_input_ids) == len(prompt_ids) + len(response_ids), f"teacher length = {len(teacher_input_ids)}, student length = {len(student_input_ids)}, ids length = {len(prompt_ids)+len(response_ids)}"
        # print(f"teacher length = {len(teacher_input_ids)}, student length = {len(student_input_ids)}, ids length = {len(prompt_ids)+len(response_ids)}");self.tokenizer.decode(teacher_input_ids, skip_special_tokens=True); self.tokenizer.decode(prompt_ids + response_ids, skip_special_tokens=True)
        # self.tokenizer.decode(teacher_input_ids); self.tokenizer.decode(prompt_ids + response_ids)
        # self.tokenizer.decode(prompt_ids + response_ids)
        # print(f"identical string = {self.tokenizer.decode(teacher_input_ids, skip_special_tokens=False) == self.tokenizer.decode(prompt_ids + response_ids, skip_special_tokens=False)}")
        # print(f"len teacher input ids = {len(teacher_input_ids)}, len prompt ids = {len(prompt_ids)+len(response_ids)}, identical ids = {teacher_input_ids[:500] == (prompt_ids + response_ids)[:500]}")
        # print(f"len teacher input ids = {len(teacher_input_ids)}, len prompt ids = {len(prompt_ids)+len(response_ids)}, teacher = {teacher_input_ids}, prompt+response = {prompt_ids + response_ids}")

        # resplit into prompt and response
        length_prompt_ids = len(prompt_ids)
        teacher_response_mask = full_teacher_response_mask[length_prompt_ids:]
        student_prompt_ids = student_input_ids[:length_prompt_ids]
        student_response_ids = student_input_ids[length_prompt_ids:]
        student_response_mask = full_student_response_mask[length_prompt_ids:]

        # teacher response mask is the original response mask, verified by assertion above
        return teacher_response_mask, student_prompt_ids, student_response_ids, student_response_mask




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
                
            await simulator.reset_state(original_planning_stats_dict)

            available_moves = simulator.legal_moves_string

            current_plan_message = [instruction_message[0], turn_message[-1][0]]    # the system prompt and the current state query

            (
                planned_action, 
                updated_plan_message, 
                plan_ids,
                plan_response_mask,
                planning_metrics,
                metrics,
                summarize_state_understanding_text,
                summarize_response_text,
                ever_win
                ) = await self.run_independent_summary(simulator, current_plan_message, sampling_params, metrics)

            prompt_ids += plan_ids
            response_mask += plan_response_mask
            reward_lst += ([0.] * len(plan_response_mask))
            plan_ever_win_lst.append(ever_win)

            turn_message[-1].extend(updated_plan_message)
            turn_valid_records.extend(planning_metrics['planning_valid_action'])

            if self.process_sft_dataset:
                state_summary_pair.append(
                    {"state": original_planning_stats_dict['FEN'], 
                    "available_moves": available_moves,
                    "summary": summarize_state_understanding_text,
                    "decision": summarize_response_text,
                    }
                )

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
                next_state, reward, env_done, env_info = await self.env.step(planned_action)

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
                sft_dataset = await self._process_value_sft_dataset(state_summary_pair, env_success)
            elif self.sft_dataset_type == "value_table":
                sft_dataset = await self._process_value_table_dataset(state_summary_pair, env_success)
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
                    plan_message, add_generation_prompt=False, tokenize=True
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

                # if random exploration
                # random_action = random.choice(simulator.legal_moves_list)
                # response_text = f"<move>{random_action}</move>"

            else:
                if len(plan_message) > 2:
                    last_user_ids = await self.loop.run_in_executor(
                            None,
                            lambda: self.tokenizer.apply_chat_template(
                                [plan_message[-1]], add_generation_prompt=False, tokenize=True
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
                
                await simulator.reset_state(root_planning_stats_dict)
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
                _input_message, add_generation_prompt=False, tokenize=True
            ),
        )

        # need to remove the system prompt ids
        _summary_query_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                [{"role": "user", "content": plan_summarize_prompt}], add_generation_prompt=False, tokenize=True
            ),
        )
        _summary_query_ids = _summary_query_ids[len(self.system_prompt):]
        plan_response_mask = [0] * len(_summary_query_ids)


        # two stage value inference
        (
            summarize_response_text, 
            summarize_state_understanding_text,
            infer_message, 
            plan_summarize_ids, 
            infer_response_mask, 
            infer_reward_lst, 
            metrics,
            _, _, _
            ) = await self._two_stage_value_inference(
            root_planning_state, 
            plan_summarize_ids, 
            metrics, 
            sampling_params,
            move_decision_prompt=self.prompt_set['state_answer_query_prompt'],
            value_decision_prompt=self.prompt_set['simulation_state_value_query_prompt'].format(state=root_planning_state),
        )
        plan_message.extend(infer_message)
        plan_response_mask.extend(infer_response_mask)


        (final_action, final_action_valid) = self._extract_summarize_from_assistant(summarize_response_text)

        plan_ids = plan_summarize_ids[-len(plan_response_mask):]      # remove the system prompt ids

        summary_text = summarize_response_text

        # return final_action, plan_message[1:], plan_ids, plan_response_mask, planning_metrics, metrics, summary_text
        return final_action, plan_message[1:], plan_ids, plan_response_mask, planning_metrics, metrics, summarize_state_understanding_text, summarize_response_text, ever_win
