import time
import numpy as np
import asyncio
import json
import logging
import os
import re
import random
import ray
from typing import Any
from uuid import uuid4
import importlib

from tqdm import tqdm

from omegaconf import OmegaConf
from transformers import AutoTokenizer
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register, _DummyConfig, AsyncLLMServerManager
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

from tasks.classic_games.chess.prompt import DIRECT_PROMPT_SET



logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def get_env(env_config):
    if env_config.env_name=="ChessPuzzles":
        from tasks.classic_games.chess.chess_puzzle_env import ChessPuzzleEnv_Wrapper
        return ChessPuzzleEnv_Wrapper(env_config=env_config)

    elif env_config.env_name=="FrozenLake":
        from tasks.classic_games.frozen_lake.frozenlake import FrozenLakeEnv_Wrapper
        return FrozenLakeEnv_Wrapper(env_config=env_config)
    
    elif env_config.env_name=="Sokoban":
        from tasks.classic_games.sokoban.sokoban_env import SokobanEnv_Wrapper
        return SokobanEnv_Wrapper(env_config=env_config)

    else:
        raise ValueError(f"Environment {env_config.env_name} is not supported.")



class SeedGenerator:
    def __init__(self, start_range: int, end_range: int, shuffle: bool=True):

        if end_range < start_range:
            raise ValueError("End of range cannot be less than the start of the range.")

        # Store the original parameters to allow for a full reset
        self._start_range = start_range
        self._end_range = end_range
        self._shuffle = shuffle

        self._numbers = list(range(self._start_range, self._end_range + 1))
        self._current_index = 0

        if self._shuffle:
            random.shuffle(self._numbers)

    def __iter__(self):
        """
        Returns the iterator object itself.
        This makes the class an iterable.
        """
        return self

    def __next__(self) -> int:
        """
        Returns the next unique number from the range.

        Raises:
            StopIteration: When all numbers in the range have been yielded.
        """
        if self._current_index < len(self._numbers):
            number = self._numbers[self._current_index]
            self._current_index += 1
            return number
        else:
            # All numbers have been yielded
            # raise StopIteration
            self.reset()

    def reset(self):
        """
        Resets the iterator to its initial state.
        The list of numbers is regenerated and, if shuffle was True,
        the numbers will be re-shuffled.
        """
        self._numbers = list(range(self._start_range, self._end_range + 1))
        self._current_index = 0
        if self._shuffle:
            random.shuffle(self._numbers)

    def __len__(self):
        """
        Returns the total number of unique items this iterator can produce
        before exhaustion.
        """
        return len(self._numbers)

    def remaining(self) -> int:
        """
        Returns the number of unique items still available to be yielded.
        """
        return len(self._numbers) - self._current_index



@register("game_value_infer_agent")
class GameValueInfer_AgentLoop(AgentLoopBase):
    """Agent loop with custom SearchClient for search API execution"""

    def __init__(self, trainer_config, server_manager, tokenizer, **kwargs):
        """Initialize the agent loop instance."""
        super().__init__(trainer_config, server_manager, tokenizer, **kwargs)
        # Initialize instance variable for tracking action validity
        self.action_valid_records = []

    @classmethod
    def init_class(cls, config, tokenizer, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level GameValueInfer_AgentLoop initialization")

        # oss new chat template
        # oss_chat_template_path = "resources.models.gpt_oss_20b.tokenizer.GPT_OSS_20B_TEMPLATE"
        # module_path, class_name = oss_chat_template_path.rsplit(".", 1)
        # module = importlib.import_module(module_path)
        # new_chat_template = getattr(module, class_name)
        # tokenizer.chat_template = new_chat_template

        # Initialize tokenizer and config
        cls.tokenizer = tokenizer
        cls.max_window_size = config.actor_rollout_ref.rollout.multi_turn.get('max_window_size', None)

        # Game/environment configuration and tags
        env_config = config.actor_rollout_ref.rollout.multi_turn.envs.env_config
        cls.env_config = env_config
        cls.max_steps = env_config.max_steps

        # action processor
        module_path, class_name = env_config.action_processor.rsplit(".", 1)
        module = importlib.import_module(module_path)
        cls.extract_answer = staticmethod(getattr(module, class_name))

        cls.each_response_length = config.actor_rollout_ref.rollout.each_response_length

        # cls.model_api_client = DirectModelAPIClient(max_output_tokens=max_output_tokens)

        # prompt set
        prompt_set_path = env_config.prompt_set
        module_path, class_name = prompt_set_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        cls.prompt_set = getattr(module, class_name)
        cls.conversation_prefix = env_config.rollout.conversation_prefix


        # Query/answer/action detection patterns (configurable)
        action_start_tag = '<answer>'
        action_end_tag = '</answer>'

        cls.start_tool_marker = ""

        cls.action_pattern = re.compile(f'{re.escape(action_start_tag)}(.*?){re.escape(action_end_tag)}', re.DOTALL)

        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        cls.system_prompt = tokenizer.apply_chat_template([{}], add_generation_prompt=False, tokenize=True)

        cls.intermediate_instruction = config.actor_rollout_ref.rollout.multi_turn.get("intermediate_instruction", False)
        cls.act_action_format_penality = config.actor_rollout_ref.rollout.multi_turn.envs.env_config.act_action_format_penality
        cls.random_move_when_invalid_act = config.actor_rollout_ref.rollout.multi_turn.random_move_when_invalid_act

        cls.enable_thinking = config.actor_rollout_ref.rollout.enable_thinking

        cls._env_initialized = False

    def build_env(self, seed):
        # Initialize a single environment instance for this agent loop
        self.env = get_env(self.env_config)

        init_state = self.env.reset(specified_game_idx=seed)

        self.available_actions = getattr(self.env, 'legal_moves_list', [])
        self._env_initialized = True

        _query = self.prompt_set['query_prompt'].format(
            state=init_state,
            available_move=self.env.legal_moves_string,     # if has exceed max_legal_action_n, will be truncated
            turn_idx=0,
            turn_left=self.max_steps)

        return _query

    @rollout_trace_op
    async def run(self, messages: list[dict[str, Any]], sampling_params: dict[str, Any], seed: int) -> AgentLoopOutput:
        # Default to sequential interaction with environment
        return await self.run_mdp(messages, sampling_params, seed)

    @rollout_trace_op
    async def run_mdp(self, messages: list[dict[str, Any]], sampling_params: dict[str, Any], seed) -> AgentLoopOutput:
        metrics = {}
        conversation_id = uuid4().hex
        worker_start_time = time.time()
        sampling_params['max_tokens'] = self.each_response_length

        # process instruction prompt
        system_prompt = self.prompt_set['system_prompt']

        # reset game and add user content
        initial_query = self.build_env(seed)        # this seed is actually a game index

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
                [instruction_message[0], turn_message[0][0]], add_generation_prompt=True, tokenize=True
            )
        )
        response_mask = []
        reward_lst = []

        user_turns, assistant_turns = 0, 0
        early_stop_flag = 0     # if not stop at agent response

        step = 0
        env_success=False
        env_end = False

        while True:
            # prepare for message
            if self.max_window_size is not None:
                turn_prompt = turn_message[-self.max_window_size:]  # a list of role message
                # input_message = []
                # input_message.extend(instruction_message)
                # for m in turn_prompt:
                #     input_message.extend(m)
                # merge user query if there is any
                clean_message = []
                clean_message.extend(instruction_message)

                for _turn_message in turn_prompt:
                    for m in _turn_message:
                        if m["role"] != 'user':
                            clean_message.append(m)
                        else:
                            if clean_message[-1]["role"] == "user":
                                clean_message[-1]["content"] += "\n" + m["content"]
                            else:
                                clean_message.append(m)
                input_message = clean_message
                
                input_ids = await self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.apply_chat_template(
                        input_message, add_generation_prompt=True, tokenize=True
                    )
                )
            else:
                input_ids = prompt_ids
            
            (
                response_text, 
                state_understanding_text, 
                infer_message, 
                response_ids, 
                infer_response_mask, 
                infer_reward_lst, 
                metrics
                ) = await self._two_stage_value_inference(
                state=self.env.observe(), 
                input_plan_ids=input_ids, 
                metrics=metrics, 
                sampling_params=sampling_params, 
                move_decision_prompt=self.prompt_set['state_answer_query_prompt'], 
                value_decision_prompt=self.prompt_set['state_value_query_prompt'].format(state=self.env.observe())
                )

            prompt_ids += response_ids
            response_mask += infer_response_mask
            reward_lst += infer_reward_lst

            assistant_turns += 1

            turn_message[-1].append({"role": "assistant", "content": state_understanding_text + "\n" + response_text})

            # Extract action from assistant message
            action_text, action_valid = self._extract_action_from_assistant(response_text)

            if action_text is None: # handle None action before env step
                done = True
                env_early_stop = True
                end_info = "The game ends. You lose the game because no valid action is detected."
                env_message = [{"role": self.conversation_prefix, "content": end_info}]
                turn_message[-1].append(env_message)

                env_feedback_ids = await self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": end_info}], add_generation_prompt=False, tokenize=True
                    )
                )
                env_feedback_ids = env_feedback_ids[len(self.system_prompt):]
                prompt_ids += env_feedback_ids
                response_mask += [0] * len(env_feedback_ids)
                reward_lst += [0.] * len(env_feedback_ids)

                break


            # allow random action when invalid move
            if action_valid:
                action_feedback = f"Action {action_text} is valid."
            else:
                action_feedback = f"The action is invalid. Switch to random move: {action_text}"
            turn_message[-1].append({"role": "user", "content": action_feedback})

            # Record the action validity for tracking
            self.action_valid_records.append(action_valid)

            # Step the environment with the extracted action
            with simple_timer("env_step", metrics):
                next_state, reward, env_done, env_info = self.env.step(action_text)

            if not action_valid:
                reward -= self.act_action_format_penality
            reward_lst[-1] = reward

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
                env_success = self.env.check_success()
                end_info = self.prompt_set['success_prompt'].format(state=next_state) if env_success else self.prompt_set['fail_prompt'].format(
                    state=next_state, fail_reason="")
                env_feedback_text = end_info.format(state=next_state)
                env_feedback_ids = await self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": env_feedback_text}], add_generation_prompt=False, tokenize=True
                    )
                )
                env_feedback_ids = env_feedback_ids[len(self.system_prompt):]
                prompt_ids += env_feedback_ids
                response_mask += [0] * len(env_feedback_ids)
                reward_lst += [0.] * len(env_feedback_ids)
                    
                break

            # Update internal state and available actions
            self.state = next_state
            self.available_actions = getattr(self.env, 'legal_moves_list', [])

            # Build feedback message to the agent
            env_feedback_text = self._format_env_feedback(next_state, self.env.legal_moves_string, step, env_info, action_feedback)
            turn_message.append([{"role": "user", "content": env_feedback_text}])    # start a new turn message
            
            env_feedback_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": action_feedback + "\n" + env_feedback_text}], add_generation_prompt=True, tokenize=True
                )
            )
            env_feedback_ids = env_feedback_ids[len(self.system_prompt):]
            prompt_ids += env_feedback_ids
            response_mask += [0] * len(env_feedback_ids)
            reward_lst += [0.] * len(env_feedback_ids)

        # Print action validity summary for this run
        if self.action_valid_records:
            run_valid_rate = np.mean(self.action_valid_records)
        else:
            pass


        # final process
        response_ids = prompt_ids[-len(response_mask):]
        prompt_ids = prompt_ids[:len(prompt_ids) - len(response_mask)]

        metrics['others'] = {
            "env_end": int(env_end),
            "overlong": int(len(response_ids) > self.response_length),
            "success_rate": env_success,
            "action_valid_rate": np.mean(self.action_valid_records)
        }
        # print(f"\n$$$ success = {env_success},")
        worker_end_time = time.time()
        worker_duration = worker_end_time - worker_start_time
        metrics['worker_run_time'] = worker_duration

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[:self.response_length],
            response_mask=response_mask[:self.response_length],
            reward_lst=reward_lst[:self.response_length],
            num_turns=assistant_turns + 1,
            metrics=metrics,
        )
        return output


    async def _two_stage_value_inference(self, state, input_plan_ids, metrics, sampling_params, move_decision_prompt=None, value_decision_prompt=None):

        _infer_message = []
        _input_plan_response_mask = []
        _input_plan_reward_lst = []

        # stage 1: state understanding
        state_understanding_prompt = value_decision_prompt if value_decision_prompt is not None else self.prompt_set['state_value_query_prompt'].format(state=state)
        _infer_message.append({"role": "user", "content": state_understanding_prompt})
        state_understanding_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                [{"role": "user", "content": state_understanding_prompt}], add_generation_prompt=True, tokenize=True, enable_thinking=self.enable_thinking
            ),
        )
        state_understanding_ids = state_understanding_ids[len(self.system_prompt):]
        input_plan_ids += state_understanding_ids
        _input_plan_response_mask += [0] * len(state_understanding_ids)
        _input_plan_reward_lst += [0.] * len(state_understanding_ids)

        with simple_timer("generate_sequences", metrics):
            plan_step_request_id = f"{uuid4().hex}_stage_1"
            sampling_params['max_tokens'] = self.each_response_length
            sampling_params['stop'] = [self.tokenizer.eos_token]
            response_ids = await self.server_manager.generate(
                request_id=plan_step_request_id, prompt_ids=input_plan_ids, sampling_params=sampling_params
            )
            input_plan_ids += response_ids
            _input_plan_response_mask += [1] * len(response_ids)
            _input_plan_reward_lst += [0.] * len(response_ids)
        
        state_understanding_text = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(response_ids, skip_special_tokens=True)
        )
        _infer_message.append({"role": "assistant", "content": state_understanding_text})

        # stage 2: move decision
        move_decision_prompt = self.prompt_set['state_move_query_prompt'] if move_decision_prompt is None else move_decision_prompt
        _infer_message.append({"role": "user", "content": move_decision_prompt})
        move_decision_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                [{"role": "user", "content": move_decision_prompt}], add_generation_prompt=True, tokenize=True, enable_thinking=self.enable_thinking
            ),
        )
        move_decision_ids = move_decision_ids[len(self.system_prompt):]
        input_plan_ids += move_decision_ids
        _input_plan_response_mask += [0] * len(move_decision_ids)
        _input_plan_reward_lst += [0.] * len(move_decision_ids)

        with simple_timer("generate_sequences", metrics):
            plan_step_request_id = f"{uuid4().hex}_stage_2"
            sampling_params['max_tokens'] = 100
            sampling_params['stop'] = ["</move>", "</answer>"]
            response_ids = await self.server_manager.generate(
                request_id=plan_step_request_id, prompt_ids=input_plan_ids, sampling_params=sampling_params
            )
            input_plan_ids += response_ids
            _input_plan_response_mask += [1] * len(response_ids)
            _input_plan_reward_lst += [0.] * len(response_ids)
        
        response_text = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(response_ids, skip_special_tokens=True)
        )
        _infer_message.append({"role": "assistant", "content": response_text})

        return response_text, state_understanding_text, _infer_message, input_plan_ids[-len(_input_plan_response_mask):], _input_plan_response_mask, _input_plan_reward_lst, metrics


    def _extract_action_from_assistant(self, response_text: str):
        """Extract action text from assistant messages using <action> tags."""
        # Collect assistant contents from the latest generation

        # return self.extract_answer(response_text, self.action_pattern, self.available_actions)

        if response_text is None:
            if self.random_move_when_invalid_act:
                return random.choice(self.available_actions), 0.
            else:
                return None, 0.
        
        match = self.action_pattern.search(response_text)
        if not match:
            return random.choice(self.available_actions), 0.
        else:
            action_content = match.group(1).strip()

        lower_letter_available_actions = [i.lower() for i in self.available_actions]
        special_token_list = ["<think>", "</think>", "<move>", "</move>", "<|im_start|>", "<|im_end|>", "+", "#"]
        for special_token in special_token_list:
            action_content = action_content.replace(special_token, "").strip()

        if action_content.lower() in lower_letter_available_actions:
            position_idx = lower_letter_available_actions.index(action_content.lower())

            return self.available_actions[position_idx], 1.0
        else:
            if self.random_move_when_invalid_act:
                return random.choice(self.available_actions), 0.
            else:
                return None, 0.


    def _format_env_feedback(self, state: Any, truncated_legal_actions: str, step_idx, env_info: dict, action_feedback: str) -> str:
        """Return a compact feedback string for the agent with next state and legal actions."""
        if "oppo_move" in env_info and "query_prompt_with_info" in self.prompt_set:
            next_query = self.prompt_set['query_prompt_with_info'].format(
                state=state,
                available_move=truncated_legal_actions,
                turn_idx=step_idx,
                turn_left=self.max_steps-step_idx,
                oppo_move=env_info['oppo_move'])
        else:
            next_query = self.prompt_set['query_prompt'].format(
                state=state,
                available_move=truncated_legal_actions,
                turn_idx=step_idx,
                turn_left=self.max_steps-step_idx)

        return next_query


    def _limit_context_by_turns(self, raw_message, max_window_size):
        """
        Limit the context to the last max_window_size turns, keeping the system instruction
        and only the last N turns of (user instruction, assistant response, env_message) pairs.
        
        Args:
            raw_message: List of message dictionaries
            max_window_size: Maximum number of turns to keep in context
            
        Returns:
            List of message dictionaries with limited context
        """
        if max_window_size is None or len(raw_message) <= max_window_size:
            return raw_message
        
        # Find system messages to keep at the beginning
        system_messages = []
        for msg in raw_message:
            if msg.get('role') == 'system':
                system_messages.append(msg)
        
        # Find all conversation turns (user + assistant pairs)
        conversation_turns = []
        i = 0
        while i < len(raw_message):
            msg = raw_message[i]
            if msg.get('role') == self.conversation_prefix:
                # Found a user message, look for the corresponding assistant response
                turn = [msg]  # Start with user message
                i += 1
                
                # Look for assistant response
                while i < len(raw_message) and raw_message[i].get('role') == 'assistant':
                    turn.append(raw_message[i])
                    i += 1
                
                # Look for any env_message that follows (like success/fail messages)
                while i < len(raw_message) and raw_message[i].get('role') == self.conversation_prefix:
                    turn.append(raw_message[i])
                    i += 1
                
                conversation_turns.append(turn)
            else:
                i += 1
        
        # Keep only the last max_window_size turns
        if len(conversation_turns) <= max_window_size:
            # If we have fewer turns than the limit, keep all
            limited_turns = conversation_turns
        else:
            # Keep only the last max_window_size turns
            limited_turns = conversation_turns[-max_window_size:]
        
        # Flatten the limited turns back into a message list
        limited_messages = []
        for turn in limited_turns:
            limited_messages.extend(turn)
        
        # Combine system messages + limited conversation turns
        result = system_messages + limited_messages
        
        return result


final_results = []


async def _async_main():
    import argparse
    from pathlib import Path
    import os
    from datetime import datetime

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/config/oss_local_loop_trainer.yaml")
    # dataset-style args (align with evaluation/scripts/eval_async_search.py)
    parser.add_argument("--data_source", type=str, nargs="+", default=["mock_env"], help="name(s) under datasets/ to load; use 'mock_env' to generate synthetic env instances")
    parser.add_argument("--tool_config_path", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None, help="Local model path. Required if config model.path is null.")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Tokenizer path. Defaults to --model_path or config model.path.")
    parser.add_argument("--max_turns", type=int, default=3)
    parser.add_argument("--output", type=str, default="logs/eval/value_infer/infer_logs.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="limit number of samples for a quick run")
    parser.add_argument("--mock_dataset_size", type=int, default=50, help="size of mock env dataset when using mock_env")
    parser.add_argument("--each_response_length", type=int, default=2048)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--reasoning_effort", type=str, default="low")
    parser.add_argument("--max_window_size", type=int, default=None, help="Maximum number of conversation turns to keep in context for vLLM server (overrides config)")
    args = parser.parse_args()

    # Load config
    config = OmegaConf.load(args.config)

    # Ensure tool config path is set
    if not config.actor_rollout_ref.rollout.multi_turn.tool_config_path:
        repo_root = Path(__file__).resolve().parents[2]
        tool_cfg = repo_root / "src" / "config" / "tool_cfg" / "local_search_config.yaml"
        config.actor_rollout_ref.rollout.multi_turn.tool_config_path = str(tool_cfg)

    # Model/tokenizer path overrides
    if args.model_path is not None:
        config.actor_rollout_ref.model.path = args.model_path
    elif config.actor_rollout_ref.model.get("path", None) is None:
        raise ValueError("A local model path is required. Pass --model_path or set actor_rollout_ref.model.path in config.")

    # Tokenizer from model path (can override)
    if args.tokenizer_path is not None:
        tokenizer_path = args.tokenizer_path
    else:
        tokenizer_path = config.actor_rollout_ref.model.path
    tokenizer = AutoTokenizer.from_pretrained(os.path.expanduser(tokenizer_path), trust_remote_code=True)

    # Minimal sampling params
    sampling_params = {
        "temperature": config.actor_rollout_ref.rollout.temperature,
        "top_p": config.actor_rollout_ref.rollout.top_p,
        "stop": list(config.actor_rollout_ref.rollout.get("stop_words", [])),
    }

    # Multi-turn and stopwords (follow eval settings)
    config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns = args.max_turns
    config.actor_rollout_ref.rollout.multi_turn.max_user_turns = args.max_turns
    config.actor_rollout_ref.rollout.response_length=8192
    config.actor_rollout_ref.rollout.each_response_length=args.each_response_length
    config.actor_rollout_ref.rollout.stop_words=["</action>", "</answer>"]
    
    # Max window size override
    if args.max_window_size is not None:
        config.actor_rollout_ref.rollout.multi_turn.max_window_size = args.max_window_size

    # Tool config override
    if args.tool_config_path is not None:
        config.actor_rollout_ref.rollout.multi_turn.tool_config_path = args.tool_config_path

    # Ensure dataset returns raw chat messages
    if not hasattr(config, "data"):
        config["data"] = {}
    config.data.return_raw_chat = True

    # Initialize local model server for inference (similar to ray_trainer.py)
    # Set up proxy env for openai client compatibility
    os.environ["no_proxy"] = ""
    os.environ["http_proxy"] = ""
    os.environ["https_proxy"] = ""

    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            namespace="default",
            logging_level=logging.WARNING,
        )

    # Import required classes for server setup
    from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
    from verl.single_controller.ray.base import create_colocated_worker_cls
    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
    from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker
    from src.workers.rollout.vllm_rollout.async_server import AsyncLLMServerManager

    # Create hybrid ActorRollout workers with Ray
    role_worker_mapping = {
        Role.ActorRollout: ray.remote(AsyncActorRolloutRefWorker),
    }
    global_pool_id = "global_pool"
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
    }
    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    resource_pool_manager.create_resource_pool()
    resource_pool_to_cls = {pool: {} for pool in resource_pool_manager.resource_pool_dict.values()}

    # Create actor and rollout worker classes
    resource_pool = resource_pool_manager.get_resource_pool(Role.ActorRollout)
    actor_rollout_cls = RayClassWithInitArgs(cls=role_worker_mapping[Role.ActorRollout], config=config.actor_rollout_ref, role="actor_rollout")
    resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls

    # Initialize worker groups
    all_wg = {}
    for resource_pool, class_dict in resource_pool_to_cls.items():
        worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
        wg_dict = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
        spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
        all_wg.update(spawn_wg)
    actor_rollout_wg = all_wg["actor_rollout"]
    actor_rollout_wg.init_model()

    # Create AsyncLLMServerManager with the worker group for local inference
    server_manager = AsyncLLMServerManager(
        config=config,
        worker_group=actor_rollout_wg,
        scheduler_kwargs=None,
    )
    print("✓ Local model server initialized successfully")

    # Always use mock env dataset
    use_mock_env = True
    parquet_files = []
    jsonl_files = []

    # Build output file if requested
    output_fp = None
    if args.output is not None:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        output_fp = open(args.output, "w", encoding="utf-8")


    total_processed = 0

    # Build a simple instruction for the agent
    # instruction = (
    #     "You are an environment-playing agent. Read <state> and <available_actions>, then output your move in <action>...</action>."
    # )
    instruction = DIRECT_PROMPT_SET['system_prompt']
    # Seed generator for reproducibility
    seed_pool = SeedGenerator(0, 600, shuffle=True)

    # Prepare env_config from the loaded config
    env_cfg = config.actor_rollout_ref.rollout.multi_turn.envs.env_config

    # Prepare total count and seeds upfront to avoid sharing the generator across tasks
    total_count = args.mock_dataset_size if args.limit is None else min(args.mock_dataset_size, args.limit)
    seeds = [next(seed_pool) for _ in range(total_count)]


    # Run tasks asynchronously using the local server manager
    # Create tasks with limited concurrency using semaphore
    semaphore = asyncio.Semaphore(args.concurrency)
    
    async def run_one_with_semaphore(seed):
        async with semaphore:
            try:
                local_agent = GameValueInfer_AgentLoop(
                    trainer_config=_DummyConfig(config=config),
                    server_manager=server_manager,
                    tokenizer=tokenizer,
                    reasoning_effort=args.reasoning_effort
                )
                local_messages = [
                    {"role": "system", "content": instruction},
                ]
                return await local_agent.run_sequential(local_messages, sampling_params, seed=seed)
            except Exception as e:
                error_msg = f"Error running seed {seed}: {str(e)}"
                print(f"Task error: {error_msg}")
                import traceback
                traceback.print_exc()
                return ([], [], False, {"error": error_msg, "exception_type": type(e).__name__})
    
    # Create tasks for all seeds
    tasks = [run_one_with_semaphore(seed) for seed in seeds]
    
    # Run tasks with progress bar
    pbar = tqdm(total=total_count, desc="Running env instances", unit="task", dynamic_ncols=True)
    
    for coro in asyncio.as_completed(tasks):
        try:
            res = await coro
            final_results.append(res)
            pbar.update(1)
        except Exception as e:
            print(f"Task failed with error: {e}")
            import traceback
            traceback.print_exc()
            final_results.append(([], [], False, {"error": str(e), "exception_type": type(e).__name__}))
            pbar.update(1)
    
    pbar.close()
    
    # Count successful tasks
    total_processed = len(final_results)
    if total_processed % 10 == 0 or total_processed == total_count:
        print(f"Processed {total_processed} samples...")

    success_rate = [i[2] for i in final_results]
    print(f"===============================")
    print(f"Game num = {len(success_rate)}, Avg Success Rate = {np.mean(success_rate)}")
    print(f"===============================")
    
    # Calculate and print mean action valid rate
    all_action_valid_records = []
    for result in final_results:
        if len(result) >= 4 and isinstance(result[3], dict) and 'action_valid_records' in result[3]:
            all_action_valid_records.extend(result[3]['action_valid_records'])
    
    if all_action_valid_records:
        mean_valid_rate = np.mean(all_action_valid_records)
        print(f"===============================")
        print(f"Total Actions = {len(all_action_valid_records)}, Mean Valid Rate = {mean_valid_rate:.4f}")
        print(f"===============================")
    else:
        print(f"===============================")
        print(f"No action valid records found")
        print(f"===============================")
    
    # Check failure rate and error details
    failed_tasks = []
    for i, result in enumerate(final_results):
        if len(result[0]) == 0:  # If first element is empty list, task failed
            error_info = result[-1] if len(result) > 0 else {}  # Last element contains the error
            failed_tasks.append({
                'task_index': i,
                'error_info': error_info
            })
    
    failure_rate = len(failed_tasks) / len(final_results) if final_results else 0
    print(f"===============================")
    print(f"Failure Rate: {failure_rate:.2%} ({len(failed_tasks)}/{len(final_results)} tasks failed)")
    print(f"===============================")
    
    if failed_tasks:
        print("Failed task details:")
        for task in failed_tasks[:10]:  # Show first 10 failed tasks
            print(f"  Task {task['task_index']}: {task['error_info']}")
        if len(failed_tasks) > 10:
            print(f"  ... and {len(failed_tasks) - 10} more failed tasks")
    
    # Save final results to output file
    if output_fp is not None:
        print(f"Saving results to {args.output}...")
        
        # Prepare results for JSON output
        formatted_results = []
        for i, result in enumerate(final_results):
            if len(result) == 4:  # Successful result: (raw_message, template_message, env_success, metrics)
                raw_message, template_message, env_success, metrics = result
                formatted_result = {
                    "task_index": i,
                    "seed": seeds[i] if i < len(seeds) else None,
                    "success": bool(env_success),
                    "raw_message": raw_message,
                    "template_message": template_message,
                    "metrics": metrics,
                    "error": None
                }
            else:  # Failed result: usually ([], [], False, {"error": "..."})
                error_info = result[-1] if len(result) > 0 else {}
                formatted_result = {
                    "task_index": i,
                    "seed": seeds[i] if i < len(seeds) else None,
                    "success": False,
                    "raw_message": [],
                    "template_message": [],
                    "metrics": {},
                    "error": error_info
                }
            formatted_results.append(formatted_result)
        
        # Add summary statistics and metadata
        output_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "config_file": args.config,
                "model_path": args.model_path,
                "tokenizer_path": args.tokenizer_path,
                "total_count": total_count,
                "concurrency": args.concurrency,
                "execution_mode": "parallel",
                "seeds": seeds
            },
            "summary": {
                "total_tasks": len(final_results),
                "successful_tasks": sum(1 for r in final_results if len(r) == 4 and r[2]),
                "failed_tasks": len(failed_tasks),
                "success_rate": float(np.mean(success_rate)) if success_rate else 0.0,
                "failure_rate": failure_rate,
                "total_actions": len(all_action_valid_records) if all_action_valid_records else 0,
                "mean_valid_rate": float(np.mean(all_action_valid_records)) if all_action_valid_records else 0.0,
                "execution_time": time.time() - start_time
            },
            "results": formatted_results
        }
        
        # Write to file
        json.dump(output_data, output_fp, indent=2, ensure_ascii=False, default=str)
        output_fp.close()
        print(f"Results saved successfully to {args.output}")
    else:
        print("No output file specified, results not saved")


if __name__ == "__main__":
    start_time = time.time()
    asyncio.run(_async_main())

    time_cost = time.time() - start_time
    print(f"Time: {time_cost}")
    import pdb
    pdb.set_trace()
    final_results
    print(1)
    
