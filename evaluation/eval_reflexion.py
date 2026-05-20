from openai import AsyncOpenAI
from openai import APIConnectionError, APIError
import os
import importlib
import random
import re
from omegaconf import OmegaConf
import json
import numpy as np
import time
import argparse
import asyncio
from tqdm import tqdm
import shutil
from typing import List, Dict, Any


def _safe_run_tag(value: str | None, fallback: str = "default_model") -> str:
    if not value:
        return fallback
    tag = os.path.basename(str(value).rstrip("/")) or fallback
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", tag)


def _redacted_args_dict(args: argparse.Namespace) -> Dict[str, Any]:
    args_dict = dict(vars(args))
    for key in ("api_key", "llm_url"):
        if args_dict.get(key):
            args_dict[key] = "<redacted>"
    return args_dict


async def vllm_generate(
        input_message, 
        max_tokens=2048, 
        temperature=0., 
        stop=None, 
        llm_url=None,
        llm_model_name="Qwen2.5-7B", 
        disable_thinking=False,
        api_key=None
    ):
    """
    """
    if not llm_url:
        raise ValueError("llm_url is required for remote LLM generation.")
    if not api_key:
        raise ValueError("api_key is required for remote LLM generation.")

    client = AsyncOpenAI(
        base_url=llm_url,
        api_key=api_key,
    )

    completion_params = {
        "messages": input_message,
    }
    if llm_model_name is not None:
        completion_params["model"] = llm_model_name

    completion_params["max_tokens"] = max_tokens
    completion_params["temperature"] = temperature
    if stop is not None:
        completion_params['stop'] = stop
        completion_params['extra_body'] = {
            "include_stop_str_in_output": True,
            "chat_template_kwargs": {"enable_thinking": not disable_thinking}
            }
    else:
        completion_params['extra_body'] = {
            "chat_template_kwargs": {"enable_thinking": not disable_thinking}
            }
    
    # Retry logic: attempt up to 5 times
    max_retries = 5
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            completion = await client.chat.completions.create(**completion_params)
            token_usage = completion.usage.completion_tokens
            return completion.choices[0].message.content, token_usage
        except (APIConnectionError, APIError, Exception) as e:
            last_exception = e
            if attempt < max_retries - 1:
                # Wait a bit before retrying (exponential backoff)
                await asyncio.sleep(0.5 * (2 ** attempt))
                continue
            else:
                # All retries exhausted, raise connection error
                raise ConnectionError(f"Failed to get valid result after {max_retries} attempts. Last error: {str(last_exception)}") from last_exception


def build_env(env_config, seed, query_prompt=None):
    _query = None
    if env_config.env_name == "ChessPuzzles":
        from tasks.classic_games.chess.chess_puzzle_env import ChessPuzzleEnv_Wrapper
        max_steps = env_config.max_steps
        
        # Initialize a single environment instance for this agent loop
        env = ChessPuzzleEnv_Wrapper(env_config=env_config)

        init_state = env.reset(specified_game_idx=seed)

        available_actions = getattr(env, 'legal_moves_list', [])

        if query_prompt is not None:
            _query = query_prompt.format(
                state=init_state,
                available_move=env.legal_moves_string,
                turn_idx=0,
                turn_left=max_steps)
    
    elif env_config.env_name == "FrozenLake":
        from tasks.classic_games.frozen_lake.frozenlake import FrozenLakeEnv_Wrapper
        max_steps = env_config.max_steps
        
        # Initialize a single environment instance for this agent loop
        env = FrozenLakeEnv_Wrapper(env_config=env_config)
        
        init_state = env.reset(specified_game_idx=seed)
        
        available_actions = getattr(env, 'legal_moves_list', [])

        if query_prompt is not None:
            _query = query_prompt.format(
                state=init_state,
                available_move=env.legal_moves_string,
                turn_idx=0,
                turn_left=max_steps)
    
    elif env_config.env_name == "Sokoban":
        from tasks.classic_games.sokoban.sokoban_env import SokobanEnv_Wrapper
        max_steps = env_config.max_steps
        
        # Initialize a single environment instance for this agent loop
        env = SokobanEnv_Wrapper(env_config=env_config)
        
        init_state = env.reset(specified_game_idx=seed)
        
        available_actions = getattr(env, 'legal_moves_list', [])

        if query_prompt is not None:
            _query = query_prompt.format(
                state=init_state,
                available_move=env.legal_moves_string,
                turn_idx=0,
                turn_left=max_steps)

    elif env_config.env_name == "AlfWorld":
        from tasks.classic_games.alf_world.alfworld_env import AlfWorldEnv_Wrapper
        max_steps = env_config.max_steps
        
        # Initialize a single environment instance for this agent loop
        env = AlfWorldEnv_Wrapper(env_config=env_config, train_eval="eval_out_of_distribution")
        
        init_state = env.reset(specified_game_idx=seed)
        
        available_actions = getattr(env, 'legal_moves_list', [])

        if query_prompt is not None:
            _query = query_prompt.format(
                state=init_state,
                available_move=env.legal_moves_string,
                turn_idx=0,
                turn_left=max_steps,
                task_description=env.task_goal)

    elif env_config.env_name=="ScienceWorld":
        from tasks.classic_games.scienceworld.scienceworld_env import ScienceWorldEnv_Wrapper
        max_steps = env_config.max_steps
        
        # Initialize a single environment instance for this agent loop
        env = ScienceWorldEnv_Wrapper(env_config=env_config)
        
        init_state = env.reset(specified_game_idx=seed)
        
        available_actions = getattr(env, 'legal_moves_list', [])
        
        if query_prompt is not None:
            _query = query_prompt.format(
                state=init_state,
                available_move=env.legal_moves_string,
                turn_idx=0,
                turn_left=max_steps,
                task_description=env.task_goal)

    else:
        raise NotImplementedError(f"Environment {env_config.env_name} is not supported.")

    if _query is None:
        return env, available_actions

    return _query, env, available_actions


class EnvironmentHistory:
    def __init__(self, base_query: str, start_info, memory: List[str], history: List[Dict[str, str]] = []) -> None:
        self._cur_query: str = f'{_get_base_query(base_query, start_info, memory)}'
        self._history: List[Dict[str, str]] = history
        self._last_action: str = ''
        self._is_exhausted: bool = False

    def add(self, label: str, value: str) -> None:
        assert label in ['action', 'observation', 'human_edit']
        self._history += [{
            'label': label,
            'value': value,
        }]
        if label == 'action':
            if value == self._last_action:
                self._is_exhausted = True
            else:
                self._last_action = value

    def check_is_exhausted(self) -> bool:
        return self._is_exhausted

    def reset(self) -> None:
        self._history = []

    def __str__(self) -> str:
        s: str = self._cur_query + '\n'
        for i, item in enumerate(self._history):
            if item['label'] == 'action':
                s += f'> {item["value"]}'
            elif item['label'] == 'observation':
                s += item['value']
            # NOT CURRENTLY SUPPORTED
            elif item['label'] == 'human_edit':
                s += f'[human edit]: {item["value"]}'
            if i != len(self._history) - 1:
                s += '\n'
        return s

def _get_base_query(base_query: str, start_info: str, memory: List[str]) -> str:
    query = base_query

    # add memory if it exists
    if len(memory) > 0:
        query += '\n\nYour memory for the task below:'
        for i, m in enumerate(memory):
            query += f'\nTrial {i}:\n{m.strip()}'
    query += f"\nHere is the task:\n{start_info}"
    return query


def extract_action_from_assistant(response_text, available_actions, validate_action=True):
    """Extract action from assistant response, similar to eval_api.py"""
    if response_text is None:
        return random.choice(available_actions), 0.
    
    action_start_tag, action_end_tag = '<answer>', '</answer>'
    action_pattern = re.compile(f'{re.escape(action_start_tag)}(.*?){re.escape(action_end_tag)}', re.DOTALL)
    match = action_pattern.search(response_text)
    if not match:
        if validate_action:
            return random.choice(available_actions), 0.
        else:
            return response_text, 0.
    else:
        action_content = match.group(1).strip()
    
    if not validate_action:
        return action_content, 1.0

    lower_letter_available_actions = [i.lower() for i in available_actions]
    special_token_list = ["<think>", "</think>", "<move>", "</move>", "<|im_start|>", "<|im_end|>", "+", "#"]
    for special_token in special_token_list:
        action_content = action_content.replace(special_token, "").strip()

    if action_content.lower() in lower_letter_available_actions:
        position_idx = lower_letter_available_actions.index(action_content.lower())
        return available_actions[position_idx], 1.0
    else:
        return random.choice(available_actions), 0.


async def generate_reflection(trial_messages, env_success, llm_generate_func, max_tokens=None, temperature=None):
    """Generate a reflection on what went wrong in the failed trial"""
    reflection_prompt = """You just attempted to solve a chess puzzle but failed. Please analyze what went wrong and provide a reflection that will help improve future attempts.

Consider:
1. What mistakes were made in the moves?
2. What strategic errors occurred?
3. What should be avoided in the next attempt?
4. What should be prioritized instead?

Provide a concise reflection (2-3 sentences) that will guide the next attempt."""
    
    # Include the last few messages from the trial for context
    context_messages = []
    if trial_messages:
        # Get the last few turns for context
        recent_messages = trial_messages[-6:] if len(trial_messages) > 6 else trial_messages
        for msg_group in recent_messages:
            if isinstance(msg_group, list):
                context_messages.extend(msg_group)
            else:
                context_messages.append(msg_group)
    
    reflection_messages = [
        {"role": "system", "content": reflection_prompt}
    ]
    reflection_messages.extend(context_messages)
    reflection_messages.append({
        "role": "user", 
        "content": f"The puzzle was {'solved successfully' if env_success else 'not solved'}. Please provide your reflection on what went wrong and how to improve."
    })
    
    reflection_text, _ = await llm_generate_func(reflection_messages, max_tokens=max_tokens or 512, temperature=temperature or 0.7)
    return reflection_text

async def two_stage_value_inference(
        infer_fn, 
        hist_messages, 
        max_tokens, 
        move_decision_prompt, 
        value_decision_prompt,
        input_knowledge = None,
    ):

    _infer_message = []
    _input_plan_response_mask = []
    _input_plan_reward_lst = []
    inference_token_usage = 0

    # stage 1: state understanding
    state_understanding_prompt = value_decision_prompt
    _infer_message.append({"role": "user", "content": state_understanding_prompt})

    if input_knowledge is None:
        state_understanding_text, token_usage = await infer_fn(hist_messages + _infer_message, max_tokens=max_tokens)
    else:
        state_understanding_text = input_knowledge
        token_usage = 0

    inference_token_usage += token_usage
    _infer_message.append({"role": "assistant", "content": state_understanding_text})

    # stage 2: move decision
    _infer_message.append({"role": "user", "content": move_decision_prompt})
    response_text, token_usage = await infer_fn(hist_messages + _infer_message, max_tokens=100, disable_thinking=True)
    _infer_message.append({"role": "assistant", "content": response_text})
    inference_token_usage += token_usage

    return response_text, state_understanding_text, _infer_message, inference_token_usage



async def one_episode_mdp(
    cfg,
    max_window_size,
    turn_message,
    instruction_message,
    llm_generate_func,
    max_tokens,
    temperature,
    trial_idx,
    step,
    env,
    max_steps,
    prompt_set,
    validate_action,
    inference_mode,
    ):
    # Run one trial/episode
    action_valid_ratio = []
    simulation_history = [f"State: {env.observe()}"]
    env_success = False
    inference_token_usage_total = 0
    good_move_ratio = []

    while True:
        if max_window_size is not None:
            turn_prompt = turn_message[-max_window_size:]  # a list of role message
            # Shallow-copy dicts: merging user messages mutates content in-place; sharing refs
            # with turn_message would repeatedly append into the stored transcript.
            clean_message = [dict(m) for m in instruction_message]
            for _turn_message in turn_prompt:
                for m in _turn_message:
                    if m["role"] != 'user':
                        clean_message.append(dict(m))
                    else:
                        if clean_message[-1]["role"] == "user":
                            clean_message[-1]["content"] += "\n" + m["content"]
                        else:
                            clean_message.append(dict(m))
            input_message = clean_message

        else:
            raise NotImplementedError("max_window_size is not supported")
        
        if inference_mode == "single_step":
            # Generate response from LLM
            output_message, token_usage = await llm_generate_func(input_message, max_tokens=max_tokens, temperature=temperature)
            inference_token_usage_total += int(token_usage or 0)
            # print(f"Trial {trial_idx + 1}, step = {step}, output_message = {output_message}")

            turn_message[-1].append({"role": "assistant", "content": output_message})
        elif inference_mode == "two_step":
            (
                output_message, 
                state_understanding_text, 
                infer_message,
                inference_token_usage
            ) = await two_stage_value_inference(
                llm_generate_func,
                input_message, 
                max_tokens,
                prompt_set['state_answer_query_prompt'].format(available_move=env.legal_moves_list),
                prompt_set['state_value_query_prompt'].format(state=env.observe(), available_move=env.legal_moves_list)
                )
            inference_token_usage_total += int(inference_token_usage or 0)

            turn_message[-1].extend(infer_message)
        else:
            raise NotImplementedError(f"Inference mode {inference_mode} is not supported.")


        available_actions = env.legal_moves_list
        # print(f"pre available actions = {available_actions}")
        action_text, action_valid = extract_action_from_assistant(output_message, available_actions, validate_action)
        # print(f"\naction_text = {action_text}, action_valid = {action_valid}, available={available_actions}\n")
        action_valid_ratio.append(int(action_valid))

        if validate_action:
            if action_valid:
                action_feedback = f"The action {action_text} is valid."
            else:
                action_feedback = f"The action is invalid. Switch to random move: {action_text}"
                end_info = prompt_set['fail_prompt'].format(state=env.observe(), fail_reason="Invalid Action Format.")
                env_feedback_text = end_info
                turn_message.append({
                    "role": "user",
                    "content": env_feedback_text
                })
                break
        else:
            action_feedback = ""
            

        if len(action_feedback) > 0:
            turn_message[-1].append({"role": "user", "content": action_feedback})

        next_state, reward, env_done, env_info = env.step(action_text)
        simulation_history.append(f"Action: {action_text}, done = {env_done}, state = {next_state}")
        if "oppo_move" in env_info:
            simulation_history.append(f"Opponent's move: {env_info['oppo_move']}")
        simulation_history.append(f"Reward: {reward}")
        simulation_history.append(f"State: {next_state}")
        simulation_history.append(f"Game Terminated: {env_done}")
        good_move_ratio.append(int(env_info['good']))

        step += 1
        if env_done:
            done = True
        else:
            if step > max_steps:
                done = True
            else:
                done = env_done
        
        if done:
            env_success = env.check_success()
            if step >= max_steps:
                fail_reason = "because reaching maximum steps ahead."
            else:
                fail_reason = "because falling into holes."
            
            end_info = prompt_set['success_prompt'].format(state=next_state) if env_success else prompt_set['fail_prompt'].format(
                state=next_state, fail_reason=fail_reason)
            env_feedback_text = end_info.format(state=next_state)
            turn_message.append({
                "role": "user",
                "content": env_feedback_text
            })
            break
            
        state = next_state
        available_actions = getattr(env, 'legal_moves_list', [])

        if "oppo_move" in env_info and "query_prompt_with_info" in prompt_set:
            env_feedback_text = prompt_set['query_prompt_with_info'].format(
                state=state,
                available_move=available_actions,
                turn_idx=step,
                turn_left=max_steps-step,
                oppo_move=env_info['oppo_move'],
                task_description=env.task_goal if hasattr(env, 'task_goal') else "")
        else:
            env_feedback_text = prompt_set['query_prompt'].format(
                state=state,
                available_move=available_actions,
                turn_idx=step,
                turn_left=max_steps-step,
                task_description=env.task_goal if hasattr(env, 'task_goal') else "")
        
        turn_message.append([{"role": "user", "content": env_feedback_text}])

    trial_stats = {
        "num_step": len(action_valid_ratio),
        "trial_action_valid_ratio": float(np.mean(action_valid_ratio)) if action_valid_ratio else 0.0,
        "total_token_usage": int(inference_token_usage_total),
        "good_move_ratio": float(np.mean(good_move_ratio)) if good_move_ratio else 0.0,
    }
    return turn_message, action_valid_ratio, simulation_history, env_success, trial_stats


async def run_reflexion(
    cfg, 
    seed, 
    max_tokens=None, 
    temperature=None, 
    llm_generate_func=None, 
    max_trials=3, 
    global_memory=None, 
    global_memory_lock=None, 
    num_iteration = 3,
    no_write_memory=False,   # if True, the memory will not be updated
    inference_mode="single_step",  # if two_step, then do value_infer reasoning
    ):

    """
    Run reflexion algorithm for chess: multiple trials with reflection on failures.
    
    Args:
        cfg: Configuration object
        seed: Seed for the puzzle/game
        max_tokens: Maximum tokens for LLM generation
        temperature: Temperature for LLM generation
        llm_generate_func: Async LLM generation function
        max_trials: Maximum number of trials to attempt (default: 3)
        global_memory: Shared list of dicts containing reflections from all parallel runs.
                      Each dict should have keys like: {"seed": int, "reflection": str, "timestamp": float, ...}
        global_memory_lock: asyncio.Lock for thread-safe updates to global_memory
        max_global_reflections: Maximum number of global reflections to include in prompts (default: 5)
    
    Returns:
        tuple: (best_trial_messages, best_success, best_action_valid_ratio, all_trials_info)
    """
    max_window_size = cfg.actor_rollout_ref.rollout.multi_turn.max_window_size

    prompt_set_path = cfg.actor_rollout_ref.rollout.multi_turn.envs.env_config.prompt_set
    module_path, class_name = prompt_set_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    prompt_set = getattr(module, class_name)
    system_prompt = prompt_set["system_prompt"]

    env_config = cfg.actor_rollout_ref.rollout.multi_turn.envs.env_config
    conversation_prefix = env_config.rollout.conversation_prefix
    max_steps = env_config.max_steps

    validate_action = cfg.actor_rollout_ref.rollout.multi_turn.validate_action
    
    # Get temperature and max_tokens from config if not provided
    if temperature is None:
        temperature = getattr(cfg.actor_rollout_ref.rollout, 'temperature', 0.0)
    if max_tokens is None:
        # Try to get from response_length or each_response_length
        max_tokens = getattr(cfg.actor_rollout_ref.rollout, 'response_length', None) or \
                     getattr(cfg.actor_rollout_ref.rollout, 'each_response_length', None) or 2048



    # retrieve current game seed memory
    _current_game_memory = global_memory[seed]
    final_success = False
    full_history = []
    all_action_valid_ratio = []
    trials: list[dict[str, Any]] = []
    total_infer_tokens = 0
    total_reflection_tokens = 0

    for iteration in range(num_iteration):
        
        # build env
        initial_query, env, available_actions = build_env(env_config, seed, prompt_set['query_prompt'])

        # mdp rollout
        add_on_memory = "\n".join(_current_game_memory['memory'][-3:])
        instruction_message = [
                {"role": "system", "content": system_prompt + f"\n\nPre-defined Plan: {add_on_memory}"}
            ]

        turn_message = [[
                {"role": conversation_prefix, "content": initial_query}
            ]]
        step = 0

        turn_message, action_valid_ratio, simulation_history, env_success, trial_stats = await one_episode_mdp(
            cfg,
            max_window_size,
            turn_message,
            instruction_message,
            llm_generate_func,
            max_tokens,
            temperature,
            seed,
            step,
            env,
            max_steps,
            prompt_set,
            validate_action,
            inference_mode,
        )
        total_infer_tokens += int(trial_stats.get("total_token_usage", 0) or 0)
        all_action_valid_ratio.extend(action_valid_ratio)
        trial_action_valid_ratio = float(np.mean(action_valid_ratio)) if action_valid_ratio else 0.0
        trials.append(
            {
                "trial_idx": iteration,
                "success": bool(env_success),
                "metrics": {
                    **trial_stats,
                    "trial_action_valid_ratio": trial_action_valid_ratio,
                },
                "transcript": turn_message,
            }
        )
        print(
            f"[seed={seed}] trial={iteration + 1}/{num_iteration} "
            f"success={env_success} steps={trial_stats.get('num_step', 0)} "
            f"action_valid={trial_action_valid_ratio:.4f} "
            f"inference_tokens={trial_stats.get('total_token_usage', 0)}"
        )

        # parse simulation history
        history_string = ", ".join(simulation_history)
        # for i, lst in enumerate(simulation_history):
        #     _traj = ", ".join(lst)
        #     _traj = f"{i+1}: " + _traj
        #     history_string += _traj + "\n"

        if global_memory is not None and global_memory_lock is not None and not no_write_memory:
            async with global_memory_lock:
                _current_game_memory['history'].append(history_string)
                _current_game_memory['is_success'] = env_success
        
        full_history.extend(instruction_message)
        full_history.extend(turn_message)

        # update memory
        if not _current_game_memory['is_success'] and not _current_game_memory['skip']:
            if no_write_memory:
                continue

            if len(_current_game_memory['memory']) > 3:
                memory = _current_game_memory['memory'][-3:]
            else:
                memory = _current_game_memory['memory']
            
            if "reflexion_prompt" in prompt_set:
                reflexion_instruction = prompt_set["reflexion_prompt"]
            else:
                reflexion_instruction = """You will be given the history of a past experience in which you were trying to solve a chess puzzle but failed. \
        Do not summarize your environment, but rather think about the strategy and path you took to attempt to solve the puzzle. \
        Devise a concise, new plan of actions that accounts for your mistake with reference to specific actions that you should have taken. \
        For example, if you tried A and B but forgot C, then devise a plan to achieve C with environment-specific actions. \
        You will need this later when you are solving the same puzzle again. 
        """
            reflexion_messages = [
                {"role": "system", "content": reflexion_instruction}
            ]

            reflexion_query = f"New Collected history:\n{history_string}\n"
            if len(memory) > 0:
                reflexion_query += "\n\nPlans from past attemps:\n"
                for i, m in enumerate(memory):
                    reflexion_query += f"Trial #{i}: {m}\n"
            else:
                reflexion_query = ""
            
            reflexion_query += "\n\nNew Plan:"
            reflexion_messages.append({
                "role": "user", 
                "content": reflexion_query
            })

            # get new completion
            output_message, reflection_token_usage = await llm_generate_func(
                reflexion_messages, max_tokens=max_tokens, temperature=temperature
            )
            total_reflection_tokens += int(reflection_token_usage or 0)
            output_message = f"{iteration+1}: {output_message}"
            _current_game_memory['memory'] += [output_message]
            reflexion_messages.append({"role": "assistant", "content": output_message})
            full_history.extend(reflexion_messages)
            print(
                f"[seed={seed}] reflection_generated after trial={iteration + 1}, "
                f"reflection_tokens={int(reflection_token_usage or 0)}"
            )

        else:
            final_success = True
            break

    other_stats = {
        "seed": seed,
        "num_trials_used": len(trials),
        "total_token_usage": total_infer_tokens + total_reflection_tokens,
        "total_inference_tokens": total_infer_tokens,
        "total_reflection_tokens": total_reflection_tokens,
        "avg_trial_action_valid_ratio": (
            float(np.mean([float(t["metrics"]["trial_action_valid_ratio"]) for t in trials]))
            if trials
            else 0.0
        ),
        "last_trial_num_step": int(trials[-1]["metrics"].get("num_step", 0)) if trials else 0,
        "iteration": iteration + 1,
        "good_move_ratio_list": [i['metrics']['good_move_ratio'] for i in trials]
    }
    print(
        f"[seed={seed}] reflexion_done success={final_success} "
        f"trials_used={other_stats['num_trials_used']} "
        f"total_tokens={other_stats['total_token_usage']} "
        f"(infer={other_stats['total_inference_tokens']}, reflection={other_stats['total_reflection_tokens']})"
    )
    return full_history, final_success, all_action_valid_ratio, other_stats







async def main():
    parser = argparse.ArgumentParser(description="Evaluation API with LLM generation method selection")
    parser.add_argument(
        "--llm_method",
        type=str,
        choices=["api", "vllm"],
        default="vllm",
    )
    parser.add_argument(
        "--llm_url",
        type=str,
        default=None,
        help="OpenAI-compatible API base URL. Required for --llm_method api/vllm.",
    )
    parser.add_argument(
        "--llm_model_name",
        type=str,
        default=None,
        help="Optional model name to send to the OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="API key for the OpenAI-compatible endpoint. Required for --llm_method api/vllm.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help="Maximum tokens for LLM generation. If not specified, will use config values (response_length or each_response_length)",
    )
    parser.add_argument(
        "--inference_mode",
        type=str,
        default="single_step",
        choices=["single_step", "two_step"],
    )
    parser.add_argument(
        "--cutoff",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--max_concurrency",
        type=int,
        default=10,
        help="Maximum number of concurrent evaluations",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to config file (relative to src/config/)",
    )
    parser.add_argument(
        "--repetition",
        type=int,
        default=1,
        help="Number of repetitions to run",
    )

    parser.add_argument(
        "--load_global_memory",
        type=str,
        default=None,
        help="Path to load global_memory from a previous run (JSON file)",
    )
    parser.add_argument(
        "--max_global_reflections",
        type=int,
        default=5,
        help="Maximum number of global reflections to include in prompts (default: 5)",
    )
    parser.add_argument(
        "--run_eval",
        action="store_true",
    )

    args = parser.parse_args()
    if args.llm_method in ("vllm", "api"):
        if not args.llm_url:
            parser.error("--llm_url is required when --llm_method is api or vllm.")
        if not args.api_key:
            parser.error("--api_key is required when --llm_method is api or vllm.")

    if args.llm_method == "api":
        # selected_llm_func = api_generate
        pass
    elif args.llm_method == "vllm":
        # Create a wrapper function that binds llm_url and llm_model_name
        async def vllm_generate_wrapper(input_message, max_tokens=2048, temperature=0., stop=None, disable_thinking=False):
            return await vllm_generate(
                input_message,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
                llm_url=args.llm_url,
                llm_model_name=args.llm_model_name,
                disable_thinking=disable_thinking,
                api_key=args.api_key
            )
        selected_llm_func = vllm_generate_wrapper
    else:
        raise NotImplementedError(f"LLM generation method {args.llm_method} is not supported")

    run_fn = run_reflexion
    run_tag = "reflexion"

    cfg_path = os.path.join("src/config", args.config_path)
    cfg = OmegaConf.load(cfg_path)

    # specify max tokens
    if args.max_tokens is not None:
        cfg.actor_rollout_ref.rollout.each_response_length = args.max_tokens
    
    _test_env, _ = build_env(cfg.actor_rollout_ref.rollout.multi_turn.envs.env_config, seed=None)
    # from tasks.classic_games.chess.chess_puzzle_env import ChessPuzzleEnv_Wrapper
    # _test_env = ChessPuzzleEnv_Wrapper(cfg.actor_rollout_ref.rollout.multi_turn.envs.env_config)
    total_num_game = _test_env.num_game
    train_val_split = int(
        cfg.actor_rollout_ref.rollout.multi_turn.envs.env_config.rollout.train_val_ratio * total_num_game)
    
    val_map_index = list(range(0, train_val_split))
    train_map_index = list(range(train_val_split, total_num_game))
    cutoff=args.cutoff
    if cutoff is not None:
        # val_map_index = val_map_index[:cutoff]
        train_map_index = train_map_index[:cutoff]

    # print(f"game rating = {cfg.actor_rollout_ref.rollout.multi_turn.envs.env_config.rating_range}")
    print(f"\n\ntrain_map_index: {train_map_index}, val_map_index: {val_map_index}\n\n")
    print(f"Running with max_concurrency={args.max_concurrency}\n")

    # Create semaphore to limit concurrency
    semaphore = asyncio.Semaphore(args.max_concurrency)
    
    # Initialize global memory for sharing reflections across parallel runs
    global_memory = {}
    if args.load_global_memory and os.path.exists(args.load_global_memory):
        with open(args.load_global_memory, "r") as f:
            global_memory = json.load(f)
        print(f"Loaded {len(global_memory)} entries from global_memory file: {args.load_global_memory}\n")
    else:
        # placeholder
        for train_seed in train_map_index:
            global_memory[train_seed] = {
                "seed": train_seed,
                "history": [],
                "memory": [],
                "timestamp": time.time(),
                "trial_idx": 0,
                "is_success": False,
                "action_valid_ratio": 0.0,
                "skip": False,
            }
    global_memory_lock = asyncio.Lock()
    
    async def run_reflexion_with_semaphore(seed, fn, num_iteration, no_write_memory=False):
        async with semaphore:
            result = await fn(cfg, seed, max_tokens=args.max_tokens, llm_generate_func=selected_llm_func,
                            global_memory=global_memory, global_memory_lock=global_memory_lock,
                            num_iteration=num_iteration, no_write_memory=no_write_memory, inference_mode=args.inference_mode)
            return seed, result

    # Prepare output folder for saving
    save_time = time.strftime("%Y%m%d_%H%M%S")

    model_tag = _safe_run_tag(args.llm_model_name)
    base_save_folder = f"evaluation/logs/eval/reflexion_{args.llm_method}_{run_tag}_{cfg.actor_rollout_ref.rollout.multi_turn.envs.env_config.env_name}_{model_tag}_{save_time}"
    
    
    for exp_idx in range(args.repetition):
        # Reset global memory for each repetition (optional: comment out if you want to accumulate across repetitions)
        # global_memory.clear()
        print(f"Starting repetition {exp_idx + 1}, global_memory reset (size: {len(global_memory)})\n")
        
        tasks = [run_reflexion_with_semaphore(_seed, run_fn, num_iteration=args.max_global_reflections) for _seed in train_map_index]
        save_folder = f"{base_save_folder}_Seed{exp_idx}"
        
        eval_log_path = os.path.join(save_folder, "eval_log.json")
        result_path = os.path.join(save_folder, "result.json")
        args_path = os.path.join(save_folder, "args.json")
        global_memory_path = os.path.join(save_folder, "global_memory.json")
        
        # Ensure the directory exists
        os.makedirs(save_folder, exist_ok=True)

        # Save args configuration
        args_dict = _redacted_args_dict(args)
        with open(args_path, "w") as f:
            json.dump(args_dict, f, indent=2)
        
        # Create an asyncio lock for thread-safe file writes
        file_lock = asyncio.Lock()
        
        # Run all tasks in parallel with progress tracking
        all_message = []
        all_success_rate = []
        all_action_valid_ratio = []
        all_other_stats = {}  # Dictionary to collect all stats from other_stats
        max_reflections = int(args.max_global_reflections)
        iteration_used_all: list = []
        # success_finish_at[k] counts tasks whose *final* success happens at iteration k.
        # Later we compute cumulative success for i >= k.
        success_finish_at = [0] * (max_reflections + 1)
        # good_move_ratio_sum_by_iter[i-1] accumulates good-move ratio value
        # aligned to iteration i (1-indexed), using last value for shorter lists.
        good_move_ratio_sum_by_iter = [0.0] * max_reflections
        
        def _append_to_json_list(file_path: str, row: dict) -> None:
            existing: list = []
            if os.path.exists(file_path):
                try:
                    with open(file_path, "r") as f:
                        existing = json.load(f)
                except (json.JSONDecodeError, FileNotFoundError):
                    existing = []
            existing.append(row)
            with open(file_path, "w") as f:
                json.dump(existing, f, indent=2)

        async def save_result_to_file(_info):
            async with file_lock:
                await asyncio.to_thread(_append_to_json_list, eval_log_path, _info)
        
        # Use asyncio.as_completed to show progress
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Evaluating games"):
            _seed, (_turn_message, _env_success, _action_valid_ratio, other_stats) = await coro
            _info = {
                "messages": _turn_message,
                "env_success": _env_success,
                "action_valid_ratio": _action_valid_ratio,
                "seed": _seed,
                **other_stats
            }
            all_message.append(_info)
            all_success_rate.append(_env_success)
            _iteration_num = int(other_stats.get("iteration", 0))
            iteration_used_all.append(_iteration_num)
            if _env_success:
                # Only successes contribute to the per-iteration success curves.
                if 1 <= _iteration_num <= max_reflections:
                    success_finish_at[_iteration_num] += 1

            good_move_ratio_list = other_stats.get("good_move_ratio_list", [])
            if good_move_ratio_list:
                last_val = float(good_move_ratio_list[-1])
                for i in range(1, max_reflections + 1):
                    idx = i - 1
                    v = float(good_move_ratio_list[idx]) if idx < len(good_move_ratio_list) else last_val
                    good_move_ratio_sum_by_iter[i - 1] += v
            else:
                # If the per-trial good_move_ratio list is missing/empty, treat as 0.0.
                # (This matches "use last value" as there is no last.)
                # Note: this only affects the new per-iteration good_move_ratio averages.
                # Existing mean_other_stats remain as-is.
                pass

            
            # Convert action_valid_ratio to scalar (mean) if it's a list/array
            if isinstance(_action_valid_ratio, (list, tuple, np.ndarray)):
                _action_valid_ratio_scalar = float(np.mean(_action_valid_ratio))
            else:
                _action_valid_ratio_scalar = float(_action_valid_ratio)
            all_action_valid_ratio.append(_action_valid_ratio_scalar)

            # Collect stats from other_stats
            for key, value in other_stats.items():
                if key not in all_other_stats:
                    all_other_stats[key] = []
                all_other_stats[key].append(value)
            
            # Save immediately as each instance completes
            await save_result_to_file(_info)
        
        # Calculate final statistics
        mean_success_rate = np.mean(all_success_rate)
        mean_action_valid_ratio = np.mean(all_action_valid_ratio)

        # Calculate averages for all other_stats
        mean_other_stats = {}
        for key, values in all_other_stats.items():
            if values:  # Only compute if there are values
                # Convert each value to a scalar (handle lists/arrays by taking their mean)
                scalar_values = []
                for val in values:
                    if isinstance(val, (list, tuple, np.ndarray)):
                        # If it's a sequence, take its mean
                        scalar_values.append(float(np.mean(val)))
                    elif isinstance(val, (int, float, np.integer, np.floating, bool)):
                        # If it's already numeric, convert to float
                        scalar_values.append(float(val))
                if scalar_values:
                    mean_other_stats[f"mean_{key}"] = float(np.mean(scalar_values))

        # Success curve vs. iteration (cumulative).
        # success at iteration k counts toward all i >= k.
        total_games = len(all_success_rate)
        successful_games = int(np.sum(all_success_rate))
        per_iteration_successes_by_limit = []
        per_iteration_success_rate = []
        per_iteration_success_rate_ignore_failures = []
        cumulative_success = 0
        for i in range(1, max_reflections + 1):
            cumulative_success += success_finish_at[i]
            per_iteration_successes_by_limit.append(int(cumulative_success))
            per_iteration_success_rate.append(float(cumulative_success) / float(total_games) if total_games else 0.0)
            per_iteration_success_rate_ignore_failures.append(
                float(cumulative_success) / float(successful_games) if successful_games else 0.0
            )

        # Histogram of "iteration used" across all tasks (successful + failed).
        if iteration_used_all:
            iteration_used_hist = (
                np.bincount(np.asarray(iteration_used_all, dtype=int), minlength=max_reflections + 1)
                .astype(int)
                .tolist()
            )
        else:
            iteration_used_hist = [0] * (max_reflections + 1)

        # Per-iteration average good_move_ratio (aligned by index; shorter lists use last value).
        per_iteration_good_move_ratio = []
        for i in range(1, max_reflections + 1):
            per_iteration_good_move_ratio.append(
                float(good_move_ratio_sum_by_iter[i - 1]) / float(total_games) if total_games else 0.0
            )
        print("Per-iteration metrics:")
        for i in range(1, max_reflections + 1):
            n = per_iteration_successes_by_limit[i - 1]
            r_all = per_iteration_success_rate[i - 1]
            r_success_only = per_iteration_success_rate_ignore_failures[i - 1]
            gmr = per_iteration_good_move_ratio[i - 1]
            print(
                f"  iter<={i}: {n}/{total_games}  all_tasks_rate={r_all:.4f}  "
                f"success_only_rate={r_success_only:.4f}  avg_good_move_ratio={gmr:.4f}"
            )

        # Save result.json with success rate and action valid ratio, merged with args
        result_data = {
            "success_rate": float(mean_success_rate),
            "action_valid_ratio": float(mean_action_valid_ratio),
            "total_games": len(all_success_rate),
            "successful_games": successful_games,
            **mean_other_stats,
            "max_global_reflections": max_reflections,
            "per_iteration_successes_by_limit": per_iteration_successes_by_limit,
            "per_iteration_success_rate": per_iteration_success_rate,
            "per_iteration_success_rate_ignore_failures": per_iteration_success_rate_ignore_failures,
            "iteration_used_hist": iteration_used_hist,
            "per_iteration_good_move_ratio": per_iteration_good_move_ratio,
            "args": args_dict  # Merge args content into result.json
        }
        with open(result_path, "w") as f:
            json.dump(result_data, f, indent=2)
        
        # Copy config file to the folder
        config_filename = os.path.basename(cfg_path)
        config_dest_path = os.path.join(save_folder, config_filename)
        shutil.copy2(cfg_path, config_dest_path)
        
        # Save global memory at the end of this repetition
        async with global_memory_lock:
            global_memory_snapshot = global_memory.copy()
        with open(global_memory_path, "w") as f:
            json.dump(global_memory_snapshot, f, indent=2)
        print(f"Global memory saved with {len(global_memory_snapshot)} entries to {global_memory_path}")
        
        print(f"Mean success rate = {mean_success_rate}, action valid = {mean_action_valid_ratio}")
        if mean_other_stats:
            other_stats_str = ", ".join([f"{k} = {v}" for k, v in mean_other_stats.items()])
            print(f"Mean other stats: {other_stats_str}")
        print(f"Results saved to folder: {save_folder}")


        ############# Repeated test on train dataset (read-only memory) #########################
        if not args.run_eval:
            return
        repeated_test_num_repeats = 10
        repeated_test_seeds = train_map_index * repeated_test_num_repeats
        repeated_test_log_path = os.path.join(save_folder, "repeated_test_log.json")
        repeated_test_result_path = os.path.join(save_folder, "repeated_test_result.json")

        def _append_rt_json_list(file_path: str, row: dict) -> None:
            existing: list = []
            if os.path.exists(file_path):
                try:
                    with open(file_path, "r") as f:
                        existing = json.load(f)
                except (json.JSONDecodeError, FileNotFoundError):
                    existing = []
            existing.append(row)
            with open(file_path, "w") as f:
                json.dump(existing, f, indent=2)

        async def run_repeated_test_with_idx(trial_idx: int, seed: int):
            s, r = await run_reflexion_with_semaphore(
                seed, run_fn, num_iteration=1, no_write_memory=True
            )
            return trial_idx, s, r

        rt_tasks = [
            run_repeated_test_with_idx(i, _seed)
            for i, _seed in enumerate(repeated_test_seeds)
        ]
        rt_success: list = []
        rt_action_valid: list = []
        rt_other_stats: dict = {}

        async def save_repeated_test_row(row: dict):
            async with file_lock:
                await asyncio.to_thread(_append_rt_json_list, repeated_test_log_path, row)

        if not rt_tasks:
            print(f"Repetition {exp_idx}: skip repeated_test (no train seeds).")
        else:
            for coro in tqdm(
                asyncio.as_completed(rt_tasks), total=len(rt_tasks), desc="repeated_test train"
            ):
                trial_idx, _seed, (
                    _turn_message,
                    _env_success,
                    _action_valid_ratio,
                    other_stats,
                ) = await coro
                if isinstance(_action_valid_ratio, (list, tuple, np.ndarray)):
                    av_scalar = float(np.mean(_action_valid_ratio))
                else:
                    av_scalar = float(_action_valid_ratio)
                rt_row = {
                    "prefix": "repeated_test",
                    "trial_idx": trial_idx,
                    "messages": _turn_message,
                    "env_success": _env_success,
                    "action_valid_ratio": _action_valid_ratio,
                    "action_valid_ratio_mean": av_scalar,
                    "seed": _seed,
                    "exp_idx": exp_idx,
                    **other_stats,
                }
                rt_success.append(_env_success)
                rt_action_valid.append(av_scalar)
                for key, value in other_stats.items():
                    if key not in rt_other_stats:
                        rt_other_stats[key] = []
                    rt_other_stats[key].append(value)
                await save_repeated_test_row(rt_row)

            rt_mean_success = float(np.mean(rt_success)) if rt_success else 0.0
            rt_mean_action_valid = float(np.mean(rt_action_valid)) if rt_action_valid else 0.0
            rt_mean_other_stats = {}
            for key, values in rt_other_stats.items():
                if values:
                    scalar_values = []
                    for val in values:
                        if isinstance(val, (list, tuple, np.ndarray)):
                            scalar_values.append(float(np.mean(val)))
                        elif isinstance(val, (int, float, np.integer, np.floating, bool)):
                            scalar_values.append(float(val))
                    if scalar_values:
                        rt_mean_other_stats[f"mean_{key}"] = float(np.mean(scalar_values))

            print(
                f"### repeated_test (repetition {exp_idx}, train seeds x{repeated_test_num_repeats}) ###"
            )
            print(
                f"  episodes={len(rt_success)}  episode_success_rate={rt_mean_success}  "
                f"action_valid_ratio={rt_mean_action_valid}  "
                f"successful_episodes={int(np.sum(rt_success))}"
            )
            if rt_mean_other_stats:
                print(
                    f"  mean other stats: {', '.join(f'{k} = {v}' for k, v in rt_mean_other_stats.items())}"
                )

            repeated_test_result_data = {
                "prefix": "repeated_test",
                "episode_success_rate": rt_mean_success,
                "action_valid_ratio": rt_mean_action_valid,
                "episode_games": len(rt_success),
                "episode_successful_games": int(np.sum(rt_success)) if rt_success else 0,
                "repeats_per_seed": repeated_test_num_repeats,
                "unique_train_seeds": len(train_map_index),
                **rt_mean_other_stats,
                "args": args_dict,
            }
            with open(repeated_test_result_path, "w") as f:
                json.dump(repeated_test_result_data, f, indent=2)
            print(
                f"  repeated_test log -> {repeated_test_log_path}  "
                f"summary -> {repeated_test_result_path}"
            )


if __name__ == "__main__":
    asyncio.run(main())
