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
from typing import List, Dict


def _safe_run_tag(value: str | None, fallback: str = "default_model") -> str:
    if not value:
        return fallback
    tag = os.path.basename(str(value).rstrip("/")) or fallback
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", tag)


def _redacted_args_dict(args: argparse.Namespace) -> Dict:
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


def _resolve_local_torch_device(device: str | None):
    import torch

    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_local_torch_dtype(device, dtype_mode: str):
    import torch

    if dtype_mode == "auto":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if device.type == "cuda":
            return torch.float16
        return torch.float32
    return getattr(torch, dtype_mode)


def load_local_causal_lm(
    model_path: str,
    device: str | None,
    dtype_mode: str,
):
    """Load tokenizer + causal LM from ``model_path`` (HF hub id or local directory)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = _resolve_local_torch_device(device)
    torch_dtype = _resolve_local_torch_dtype(dev, dtype_mode)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model.eval()
    model.to(dev)
    return model, tokenizer, dev


def local_generate_sync(
    messages: List[Dict],
    max_tokens: int,
    temperature: float,
    stop,
    model,
    tokenizer,
    device,
    disable_thinking=False,
) -> tuple[str, int]:
    """Run one chat completion; returns ``(assistant_text, new_token_count)``."""
    import torch
    if disable_thinking:
        enc = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=False,
        )
    else:
        enc = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    if isinstance(enc, dict) or hasattr(enc, "input_ids"):
        prompt_ids = enc["input_ids"]
    else:
        prompt_ids = enc
    if prompt_ids.dim() == 1:
        prompt_ids = prompt_ids.unsqueeze(0)
    prompt_ids = prompt_ids.to(device)
    prompt_len = int(prompt_ids.shape[-1])

    gen_kwargs = {
        "max_new_tokens": max_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature and temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
    else:
        gen_kwargs["do_sample"] = False

    with torch.inference_mode():
        out = model.generate(prompt_ids, **gen_kwargs)

    new_ids = out[0, prompt_len:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)

    if stop:
        stops = [stop] if isinstance(stop, str) else list(stop)
        earliest = None
        for s in stops:
            if not s:
                continue
            pos = text.find(s)
            if pos != -1:
                end = pos + len(s)
                if earliest is None or end < earliest:
                    earliest = end
        if earliest is not None:
            text = text[:earliest]

    return text, int(new_ids.shape[0])


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
        return env, available_actions, init_state

    return _query, env, available_actions, init_state



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
    

class StateKnowledge:
    def __init__(self):
        self.knowledge_table = {}
        # Exactly one concurrent coroutine may "own" filling a state; others await the same Future.
        self._lock = asyncio.Lock()
        self._pending = {}

    def write(self, state, knowledge):
        if state in self.knowledge_table:
            self.knowledge_table[state].append({"index": len(self.knowledge_table[state]), "knowledge": knowledge})
        else:
            self.knowledge_table[state] = [{"index": 0, "knowledge": knowledge}]
        
        return

    def read(self, state):
        # Non-blocking lookup for async rollout workers.
        # Return None when no prior knowledge is available for this state.
        if state not in self.knowledge_table:
            return None
        return self.knowledge_table[state][-1]['knowledge']   # get the latest knowledge

    async def acquire_knowledge_slot(self, state):
        """Return (role, knowledge_str).

        role is ``"cached"`` if ``read(state)`` is already available, or ``"owner"`` if this
        coroutine must run the LLM and then call ``commit_knowledge`` / ``abort_knowledge``.
        Waiters block until the owner finishes.
        """
        while True:
            async with self._lock:
                cached = self.read(state)
                if cached is not None:
                    return "cached", cached
                if state not in self._pending:
                    self._pending[state] = asyncio.get_running_loop().create_future()
                    return "owner", None
                fut = self._pending[state]
            await fut

    async def commit_knowledge(self, state, knowledge_text):
        async with self._lock:
            self.write(state, knowledge_text)
            fut = self._pending.pop(state, None)
        if fut is not None and not fut.done():
            fut.set_result(None)

    async def abort_knowledge(self, state):
        """If the owner fails before commit, release waiters without writing."""
        async with self._lock:
            fut = self._pending.pop(state, None)
        if fut is not None and not fut.done():
            fut.set_result(None)


class TrajectoryBuffer:
    """Aggregates unique partial trajectories (state, action, reward, terminal)* for state-conditional lookup.

    Each step is the pre-action observation, the action, reward, terminal flag, and the
    post-transition state (``next_state``). Suffixes are keyed by ``(suffix, exp_idx)`` so the same
    path can appear in different epochs. Query with ``query_continuations`` and optional
    ``scope`` / ``epoch_idx`` to restrict by training epoch.
    """

    def __init__(self):
        self._lock = asyncio.Lock()
        # Normalized suffix tuples:
        # (state_key, action_key, reward, terminal, next_state_key).
        # (suffix, exp_idx) seen across ingest; same suffix may repeat in another epoch.
        self._seen_suffixes: set[tuple] = set()
        # state_key -> list of (suffix tuple, exp_idx).
        self._by_state: dict[str, list[tuple]] = {}
        self._state_example: dict[str, object] = {}
        self._last_exp_idx: int | None = None

    @staticmethod
    def _state_key(state) -> str:
        if isinstance(state, (dict, list, tuple)):
            try:
                return json.dumps(state, sort_keys=True, default=str)
            except TypeError:
                return str(state)
        if hasattr(state, "tolist"):
            try:
                return json.dumps(state.tolist(), sort_keys=True, default=str)
            except Exception:
                pass
        return str(state)

    @staticmethod
    def _action_key(action) -> str:
        if isinstance(action, (str, int, float, bool)) or action is None:
            return json.dumps(action, sort_keys=True, default=str)
        if hasattr(action, "tolist"):
            try:
                return json.dumps(action.tolist(), sort_keys=True, default=str)
            except Exception:
                pass
        try:
            return json.dumps(action, sort_keys=True, default=str)
        except TypeError:
            return str(action)

    def _norm_step(self, state, action, reward, terminal, next_state) -> tuple:
        try:
            r = float(reward)
        except (TypeError, ValueError):
            r = float(np.asarray(reward).flat[0])
        t = bool(terminal)
        sk = self._state_key(state)
        ak = self._action_key(action)
        nsk = self._state_key(next_state)
        return (sk, ak, r, t, nsk)

    def _suffix_to_step_dicts(self, suffix: tuple, exp_idx: int | None = None) -> list[dict]:
        out = []
        for i, (sk, ak, r, t, nsk) in enumerate(suffix):
            try:
                action_val = json.loads(ak)
            except (json.JSONDecodeError, TypeError):
                action_val = ak
            row = {
                "state_key": sk,
                "action": action_val,
                "reward": r,
                "terminal": t,
                "next_state_key": nsk,
            }
            if exp_idx is not None:
                row["exp_idx"] = exp_idx
            if i == 0 and sk in self._state_example:
                row["state_example"] = self._state_example[sk]
            if nsk in self._state_example:
                row["next_state_example"] = self._state_example[nsk]
            out.append(row)
        return out

    @staticmethod
    def _split_suffix_at_root_revisit(suffix: tuple, root_sk: str) -> list[tuple]:
        """Split a suffix that revisits ``root_sk`` into separate continuations from root.

        E.g. ``A -> B -> A -> C`` becomes ``A -> B`` and ``A -> C`` instead of one long path.
        Each emitted tuple is a contiguous suffix starting at ``root_sk`` (except empty skips).
        """
        if not suffix:
            return []
        pieces: list[tuple] = []
        rest = suffix
        while rest:
            k = None
            for idx in range(len(rest)):
                if rest[idx][4] == root_sk:
                    k = idx
                    break
            if k is None:
                pieces.append(rest)
                break
            head = rest[:k]
            if head:
                pieces.append(tuple(head))
            elif rest[0][0] == root_sk and rest[0][4] == root_sk:
                pieces.append((rest[0],))
            rest = rest[k + 1 :]
            if not rest:
                break
            if rest[0][0] != root_sk:
                pieces.append(rest)
                break
        return pieces

    async def ingest_from_episode_dict(self, kt: dict, exp_idx: int) -> None:
        """Ingest one episode from a ``current_knowledge_table``-style dict.

        ``exp_idx`` tags all suffixes from this episode (training epoch / repetition index).
        """
        states = kt.get("state") or []
        actions = kt.get("action") or []
        rewards = kt.get("reward") or []
        terminates = kt.get("terminate") or []
        n = len(actions)
        if n == 0:
            return
        if len(states) != n + 1 or len(rewards) != n or len(terminates) != n:
            return
        async with self._lock:
            self._last_exp_idx = exp_idx
            for s in states:
                sk = self._state_key(s)
                if sk not in self._state_example:
                    try:
                        json.dumps(s, default=str)
                        self._state_example[sk] = s
                    except TypeError:
                        self._state_example[sk] = str(s)
            for i in range(n):
                suffix = tuple(
                    self._norm_step(
                        states[j],
                        actions[j],
                        rewards[j],
                        terminates[j],
                        states[j + 1],
                    )
                    for j in range(i, n)
                )
                key = (suffix, exp_idx)
                if key in self._seen_suffixes:
                    continue
                self._seen_suffixes.add(key)
                sk0 = suffix[0][0]
                self._by_state.setdefault(sk0, []).append((suffix, exp_idx))

    async def query_continuations(
        self,
        state,
        *,
        scope: str = "all",
        epoch_idx: int | None = None,
    ) -> list[list[dict]]:
        """Partial trajectories starting at ``state`` (same keying as ingest).

        ``scope``:
            - ``"all"``: suffixes from every ingested epoch.
            - ``"last_epoch"``: only suffixes with ``exp_idx == epoch_idx``. If ``epoch_idx`` is
              ``None``, uses the most recent ``exp_idx`` seen during ingest (``_last_exp_idx``).

        Suffixes that revisit ``state`` in the middle are split (e.g. ``A->B->A->C`` yields
        ``A->B`` and ``A->C``). Step dicts include ``exp_idx`` when present on the stored suffix.
        """
        sk = self._state_key(state)
        async with self._lock:
            entries = list(self._by_state.get(sk, ()))
            last_e = self._last_exp_idx
        if scope == "last_epoch":
            ei = epoch_idx if epoch_idx is not None else last_e
            if ei is None:
                suffixes = []
            else:
                suffixes = [(suf, e) for suf, e in entries if e == ei]
        else:
            suffixes = entries
        seen: set[tuple] = set()
        out: list[list[dict]] = []
        for suf, eidx in suffixes:
            for piece in self._split_suffix_at_root_revisit(suf, sk):
                if not piece:
                    continue
                dedupe_key = (piece, eidx)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                out.append(self._suffix_to_step_dicts(piece, eidx))
        return out

    async def to_serializable_snapshot(self) -> dict:
        async with self._lock:
            by_state = {}
            for sk, suf_list in self._by_state.items():
                by_state[sk] = [
                    {"exp_idx": eidx, "steps": self._suffix_to_step_dicts(suf, eidx)}
                    for suf, eidx in suf_list
                ]
        return {
            "num_unique_partial_trajectories": len(self._seen_suffixes),
            "num_state_keys": len(self._by_state),
            "last_exp_idx": self._last_exp_idx,
            "by_state_key": by_state,
        }


async def bootstrap_knowledge_update_for_episode(
    experience: dict,
    global_buffer: TrajectoryBuffer,
    global_knowledge_table: StateKnowledge,
    prompt_set: dict,
    llm_generate_func,
    max_tokens: int,
    max_aggregated_traj_num: int,
    max_traj_horizon: int,
    target_state_index: int = 0,
    buffer_query_scope: str = "all",
    query_epoch_idx: int | None = None,
) -> tuple[int, list | None]:
    """Aggregate buffer continuations for the root state and run the knowledge-update LLM calls.

    ``buffer_query_scope`` is ``"all"`` or ``"last_epoch"`` (see ``TrajectoryBuffer.query_continuations``).
    For ``"last_epoch"``, pass ``query_epoch_idx`` to match the current rollout epoch (or ``None`` to use
    the buffer's last ingested epoch).

    Returns ``(token_usage, knowledge_update_messages)`` where ``messages`` is ``None`` if
    skipped (``<skip>`` in model output).
    """
    tokens_used = 0
    history_string = ""
    target_state = experience["state"][target_state_index]
    subsequent_experiences = await global_buffer.query_continuations(
        target_state,
        scope=buffer_query_scope,
        epoch_idx=query_epoch_idx,
    )


    for traj_idx in range(len(subsequent_experiences)):
        if traj_idx >= max_aggregated_traj_num:
            break

        traj_steps = subsequent_experiences[traj_idx]
        if max_traj_horizon is not None and max_traj_horizon > 0:
            traj_steps = traj_steps[:max_traj_horizon]

        traj_string = f"Traj {traj_idx}:\nRoot "
        _step = None
        for _step in traj_steps:
            traj_string += (
                f"State:\n{_step['state_key']}\n, Action {_step['action']}, "
                f"Reward {_step['reward']}, Terminate {_step['terminal']}\n"
            )
        if _step is not None:
            traj_string += f"Final State:\n{_step['next_state_key']}\n"

        history_string += traj_string + "\n\n"

    previous_knowledge = experience["state_understanding"][target_state_index]

    knowledge_update_instruction = prompt_set["knowledge_update_prompt"]
    knowledge_update_messages = [
        {"role": "system", "content": knowledge_update_instruction}
    ]
    knowledge_update_query = (
        f"New Collected history:\n{history_string}\n\n"
        f"Previous Knowledge at state {target_state}:\n {previous_knowledge}\n\n"
    )

    knowledge_update_messages.append(
        {
            "role": "user",
            "content": knowledge_update_query,
        }
    )

    knowledge_update_thinking, token_usage = await llm_generate_func(
        knowledge_update_messages, max_tokens=max_tokens
    )
    tokens_used += token_usage

    # if "<skip>" in knowledge_update_thinking:
        # return tokens_used, None
    knowledge_update_messages.append({"role": "assistant", "content": knowledge_update_thinking})

    knowledge_update_messages.append(
        {
            "role": "user",
            # "content": f"Now removing unnecessary thinking, directly output your updated understanding of the state:\n{target_state}",
            "content": f"Now summarize your analysis and directly output your updated understanding of the state:\n{target_state}\n. Include necessary notes you think is important if re-plan at this state, such as the dynamics you have discovered.",

        }
    )

    knowledge_update_response, token_usage = await llm_generate_func(
        knowledge_update_messages, max_tokens=max_tokens, 
        disable_thinking=True,
    )
    tokens_used += token_usage


    if "<skip>" in knowledge_update_response:
        return tokens_used, None

    global_knowledge_table.write(
        experience["state"][target_state_index], knowledge_update_response.strip()
    )
    # print(f"Knowledge updated at state {experience['state'][target_state_index]}")

    knowledge_update_messages.append(
        {
            "role": "assistant",
            "content": knowledge_update_response,
        }
    )
    return tokens_used, knowledge_update_messages


async def bootstrap_knowledge_update_for_step(
    experience: dict,
    global_buffer: TrajectoryBuffer,
    global_knowledge_table: StateKnowledge,
    prompt_set: dict,
    llm_generate_func,
    max_tokens: int,
    max_aggregated_traj_num: int,
    max_traj_horizon: int | None = None,
    target_state_index: int = 0,
    buffer_query_scope: str = "all",
    query_epoch_idx: int | None = None,
    summarize: bool = True,
    td_step_size: int = 1,
) -> tuple[int, list | None]:
    """TD-style bootstrap: use the first ``td_step_size`` steps of each buffer continuation.

    For each trajectory, ``history_string`` lists that prefix and the **latest knowledge**
    for the **successor state after the prefix** (the n-step "final" state), when the
    deserialized observation is available as ``next_state_example`` on the last step
    (so keys match ``StateKnowledge``, which is keyed by rollout observations).
    """
    tokens_used = 0
    target_state = experience["state"][target_state_index]
    subsequent_experiences = await global_buffer.query_continuations(
        target_state,
        scope=buffer_query_scope,
        epoch_idx=query_epoch_idx,
    )
    history_string = f"Root state:\n{target_state}\n\n"

    deduplicate_state_list = []   # check the last state
    traj_list = []
    _real_traj_idx = 0
    for traj_idx in range(len(subsequent_experiences)):
        if traj_idx >= max_aggregated_traj_num:
            break

        traj_steps = subsequent_experiences[traj_idx]
        if max_traj_horizon is not None and max_traj_horizon > 0:
            traj_steps = traj_steps[:max_traj_horizon]
        td_traj_steps = traj_steps[:td_step_size]
        if not td_traj_steps:
            continue

        _ssi = td_traj_steps[0].get("suffix_start_index")
        traj_hdr = f"- Traj {_real_traj_idx}"
        if _ssi is not None:
            traj_hdr += f" (suffix_start_index={_ssi})"
        traj_string = f"----\n\n{traj_hdr}:\n"
        _prefix_len = len(traj_string)

        _step = None
        for _step in td_traj_steps:
            _info = _step.get("info") or ""
            _info_line = f", Info: {_info}\n" if (_info and _step["terminal"]) else "\n"
            traj_string += (
                f"Action: {_step['action']}, "
                f"Reward: {_step['reward']}, Terminate: {_step['terminal']}"
                f"{_info_line}"
            )
            traj_string += f"Next State:\n{_step['next_state_example']},\n\n"

        if _step is not None:
            traj_string += (
                f"Last State (after {len(td_traj_steps)} step(s)) understanding:\n"
            )
            final_obs = _step.get("next_state_example")
            if final_obs in deduplicate_state_list:
                continue    # same last state
            else:
                deduplicate_state_list.append(final_obs)

            if final_obs is not None:
                final_state_understanding = global_knowledge_table.read(final_obs)
            else:
                final_state_understanding = None
            if final_state_understanding is None:
                traj_string += (
                    "(none — not yet in knowledge table)\n"
                )
            else:
                traj_string += f"{final_state_understanding}\n"
        
        # if traj_string[_prefix_len:] in traj_list:
        #     continue
        # else:
        #     traj_list.append(traj_string[_prefix_len:])
        _real_traj_idx += 1

        history_string += (traj_string + "\n\n")
    
    # import pdb; pdb.set_trace()

    previous_knowledge = experience["state_understanding"][target_state_index]

    knowledge_update_instruction = prompt_set["knowledge_update_prompt_TD"]
    knowledge_update_messages = [
        {"role": "system", "content": knowledge_update_instruction}
    ]
    knowledge_update_query = (
        f"## New Collected history:\n{history_string}\n\n----\n\n"
        f"## Previous Knowledge at Root state:\n\n{previous_knowledge}\n\n"
    )

    knowledge_update_messages.append(
        {
            "role": "user",
            "content": knowledge_update_query,
        }
    )

    knowledge_update_response, token_usage = await llm_generate_func(
        knowledge_update_messages, max_tokens=max_tokens
    )
    tokens_used += token_usage

    knowledge_update_messages.append(
        {
            "role": "assistant",
            "content": knowledge_update_response,
        }
    )

    if summarize:
        knowledge_update_messages.append(
            {
                "role": "user",
                "content": (
                    f"Now summarize your analysis and directly output your updated understanding "
                    f"of the state:\n{target_state}\n. Include necessary notes you think is important "
                    f"if re-plan at this state."
                ),
            }
        )

        knowledge_update_response, token_usage = await llm_generate_func(
            knowledge_update_messages,
            max_tokens=max_tokens,
            disable_thinking=True,
        )
        tokens_used += token_usage

        knowledge_update_messages.append(
            {
                "role": "assistant",
                "content": knowledge_update_response,
            }
        )

    if "<skip>" in knowledge_update_response:
        return tokens_used, None
    
    # import pdb; pdb.set_trace()

    global_knowledge_table.write(
        experience["state"][target_state_index], knowledge_update_response.strip()
    )

    return tokens_used, knowledge_update_messages



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


async def run_value_infer(
        cfg, 
        seed, 
        max_tokens=None, 
        temperature=None, 
        llm_generate_func=None,
        knowledge_table: StateKnowledge = None,    # if knowledge table is given, the decision is made by first retrieve knowledge and then act,
                                # otherwise, the knowledge and decision are made by the LLM directly
        random_explore=False,
        no_knowledge_write=False,  # if True, the knowledge will not be written to the knowledge table
        only_read_at_first_state=False,  # if True, read at the first state, and force generation at the subsequent state
    ):
    max_window_size = cfg.actor_rollout_ref.rollout.multi_turn.max_window_size

    prompt_set_path = cfg.actor_rollout_ref.rollout.multi_turn.envs.env_config.prompt_set
    module_path, class_name = prompt_set_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    prompt_set = getattr(module, class_name)
    system_prompt = prompt_set["system_prompt"]

    env_config = cfg.actor_rollout_ref.rollout.multi_turn.envs.env_config
    conversation_prefix = env_config.rollout.conversation_prefix
    max_steps = env_config.max_steps
    
    # Get temperature and max_tokens from config if not provided
    if temperature is None:
        temperature = getattr(cfg.actor_rollout_ref.rollout, 'temperature', None)
    if max_tokens is None:
        # Try to get from response_length or each_response_length
        max_tokens = getattr(cfg.actor_rollout_ref.rollout, 'each_response_length', None)

    initial_query, env, available_actions, init_state = build_env(env_config, seed, prompt_set['query_prompt'])
    # puzzle_rating = env.puzzle_rating

    instruction_message = [
            {"role": "system", "content": system_prompt},
        ]
    turn_message = [[
        {"role": conversation_prefix, "content": initial_query}
    ]]
    current_knowledge_table = {
        "state": [init_state],
        "state_response": [initial_query],
        "state_understanding": [],
        "action": [],
        "action_response": [],
        "action_valid": [],
        "reward": [],
        "terminate": [],
        "good_move": [],
    }
    # state_hist.append(init_state)

    messages = [instruction_message[0], turn_message[0][0]]

    step = 0
    env_success = False
    env_end = False
    user_turns, assistant_turns = 0, 0
    action_valid_ratio = []
    inference_token_usage = 0
    state_knowledge_table_hits = 0
    state_knowledge_model_generations = 0
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
            input_message = messages
        
        current_knowledge_table['state_response'].append(turn_message[-1][0]['content'])

        obs = env.observe()
        knowledge_role = None
        state_knowledge = None
        if only_read_at_first_state:
            if step == 0:
                # try to read at the first state
                knowledge_role, state_knowledge = await knowledge_table.acquire_knowledge_slot(obs)
                if knowledge_role == "cached":
                    state_knowledge_table_hits += 1
                elif knowledge_role == "owner":
                    raise NotImplementedError("Does not read the updated first state knowledge")
            else:
                state_knowledge = None
                knowledge_role = "owner"   # need to write
        else:
            # normal case
            if knowledge_table is not None:
                knowledge_role, state_knowledge = await knowledge_table.acquire_knowledge_slot(obs)
                if knowledge_role == "cached":
                    state_knowledge_table_hits += 1
                elif knowledge_role == "owner":
                    state_knowledge_model_generations += 1

        try:
            (
                response_text, 
                state_understanding_text, 
                infer_message,
                inference_token_usage
            ) = await two_stage_value_inference(
                    llm_generate_func,
                    input_message, 
                    max_tokens, 
                    prompt_set['state_answer_query_prompt'], 
                    prompt_set['state_value_query_prompt'].format(state=obs),
                    input_knowledge = state_knowledge
                )
        except BaseException:
            if knowledge_table is not None and knowledge_role == "owner":
                await knowledge_table.abort_knowledge(obs)
            raise

        current_knowledge_table['state_understanding'].append(state_understanding_text)
        current_knowledge_table['action_response'].append(response_text)
        if not no_knowledge_write and knowledge_table is not None and knowledge_role == "owner":
            await knowledge_table.commit_knowledge(obs, state_understanding_text)
            
        inference_token_usage += inference_token_usage
        assistant_turns += 1
        # turn_message[-1].append({"role": "assistant", "content": state_understanding_text + "\n" + response_text})
        turn_message[-1].extend(infer_message)

        if random_explore:
            action_text = random.choice(available_actions)
            action_valid = True
        else:
            action_text, action_valid = extract_action_from_assistant(response_text, available_actions)

        action_valid_ratio.append(int(action_valid))

        if action_valid:
            action_feedback = f"The action {action_text} is valid."
        else:
            action_feedback = f"The action is invalid. Switch to random move: {action_text}"
        turn_message[-1].append({"role": "user", "content": action_feedback})

        next_state, reward, env_done, env_info = env.step(action_text)

        current_knowledge_table['state'].append(next_state)
        current_knowledge_table['action'].append(action_text)
        current_knowledge_table['action_valid'].append(action_valid)
        current_knowledge_table['reward'].append(reward)
        current_knowledge_table['good_move'].append(env_info['good'])

        step += 1
        if env_done:
            env_end = True
            done = True
        else:
            if step >= max_steps:
                done = True
            else:
                done = env_done
        
        current_knowledge_table['terminate'].append(done)
        if done:
            env_success = env.check_success()
            if step >= max_steps:
                fail_reason = "because reaching maximum steps."
            else:
                fail_reason = ""

            end_info = prompt_set['success_prompt'].format(state=next_state) if env_success else prompt_set['fail_prompt'].format(
                state=next_state, fail_reason=fail_reason)
            turn_message.append({
                "role": "user",
                "content": end_info
            })        
            break
            
        state = next_state
        available_actions = getattr(env, 'legal_moves_list', [])

        env_feedback_text = prompt_set['query_prompt'].format(
            state=state,
            available_move=available_actions,
            turn_idx=step,
            turn_left=max_steps-step,
            task_description=env.task_goal if hasattr(env, 'task_goal') else "")
        turn_message.append([{"role": "user", "content": env_feedback_text}])

        messages += turn_message[-1]

    other_stats = {
        "total_token_usage": inference_token_usage,
        "num_step": step,
        "good_move_ratio": float(np.mean(current_knowledge_table['good_move'])),
    }
    if knowledge_table is not None:
        _sk_total = state_knowledge_table_hits + state_knowledge_model_generations
        other_stats["state_knowledge_table_hits"] = state_knowledge_table_hits
        other_stats["state_knowledge_model_generations"] = state_knowledge_model_generations
        other_stats["state_knowledge_table_hit_ratio"] = (
            float(state_knowledge_table_hits) / _sk_total if _sk_total > 0 else 0.0
        )

    return (
        instruction_message + turn_message,
        env_success,
        np.mean(action_valid_ratio),
        other_stats,
        current_knowledge_table,
    )





async def main():
    parser = argparse.ArgumentParser(description="Evaluation API with LLM generation method selection")
    ### Rollout
    parser.add_argument(
        "--llm_method",
        type=str,
        choices=["api", "vllm", "openai", "local"],
        default="vllm",
        help="vllm/api/openai: OpenAI-compatible HTTP API; local loads HF weights via transformers on this machine.",
    )
    parser.add_argument(
        "--llm_url",
        type=str,
        default=None,
        help="OpenAI-compatible API base URL. Required for --llm_method api/vllm/openai.",
    )
    parser.add_argument(
        "--llm_model_name",
        type=str,
        default=None,
        help=(
            "Model name for remote endpoints, or local HF model path when "
            "--llm_method local. Required for --llm_method local."
        ),
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="API key for the OpenAI-compatible endpoint. Required for --llm_method api/vllm/openai.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="Maximum tokens for LLM generation. If not specified, will use config values (response_length or each_response_length)",
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
    ### Bootstrapping
    parser.add_argument(
        "--bootstrap_method",
        type=str,
        default="MC",
        choices=["MC", "TD"],
        help="Bootstrap method: MC or TD",
    )
    parser.add_argument(
        "--bootstrap_max_concurrency",
        type=int,
        default=None,
        help="Maximum concurrent bootstrap knowledge-update LLM calls; defaults to --max_concurrency.",
    )
    parser.add_argument(
        "--bootstrap_type",
        type=str,
        default="full_traj",
        choices=["full_traj", "first_state"],
        help="Bootstrap type: full_traj or first_state, if first_state, then we need full context window, and regenerate the subsequent state understand",
    )
    parser.add_argument(
        "--max_traj_num",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--max_traj_horizon",
        type=int,
        default=15,
        help="Bootstrap: max number of steps per aggregated trajectory in the history prompt (omit for full length).",
    )
    parser.add_argument(
        "--bootstrap_summarize",
        action="store_true",
        help="Bootstrap: summarize the knowledge update response.",
    )
    parser.add_argument(
        "--bootstrap_TD_step_size",
        type=int,
        default=1,
        help="Bootstrap: step size for TD bootstrap.",
    )

    ### Others
    parser.add_argument(
        "--config_path",
        type=str,
        default="value_infer_frozenlake_trainer.yaml",
        help="oss_local_loop_trainer.yaml/value_infer_chess_loop_trainer.yaml/value_lpm_chess_loop_trainer.yaml"
    )
    parser.add_argument(
        "--rollout_only_on_failures",
        action="store_true",
        help="Only rollout on failures, not on successes.",
    )
    parser.add_argument(
        "--run_evaluation",
        action="store_true",
        help="Run evaluation on the training set.",
    )
    parser.add_argument(
        "--val_num_repeats",
        type=int,
        default=5,
        help="Number of times to repeat the evaluation on the training set.",
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=1,
        help="Number of repetitions for each game",
    )
    parser.add_argument(
        "--buffer_query_scope",
        type=str,
        choices=["all", "last_epoch"],
        default="last_epoch",
        help="Trajectory buffer retrieval for bootstrap: all epochs vs only the current epoch's suffixes.",
    )
    parser.add_argument(
        "--local_llm_device",
        type=str,
        default=None,
        help="With --llm_method local: torch device (e.g. cuda, cuda:0, cpu, mps). Default: auto.",
    )
    parser.add_argument(
        "--local_llm_dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Weight dtype for local HF inference.",
    )
    parser.add_argument(
        "--local_llm_max_concurrent",
        type=int,
        default=1,
        help="Max concurrent local generate calls (use 1 on a single GPU).",
    )
    args = parser.parse_args()
    if args.llm_method in ("vllm", "api", "openai"):
        if not args.llm_url:
            parser.error("--llm_url is required when --llm_method is api, vllm, or openai.")
        if not args.api_key:
            parser.error("--api_key is required when --llm_method is api, vllm, or openai.")
    elif args.llm_method == "local" and not args.llm_model_name:
        parser.error("--llm_model_name is required when --llm_method is local.")

    bootstrap_max_concurrency = (
        args.bootstrap_max_concurrency
        if args.bootstrap_max_concurrency is not None
        else args.max_concurrency
    )

    async def vllm_generate_wrapper(input_message, max_tokens=2048, temperature=0.0, stop=None, disable_thinking=False):
        return await vllm_generate(
            input_message,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            llm_url=args.llm_url,
            llm_model_name=args.llm_model_name,
            disable_thinking=disable_thinking,
            api_key=args.api_key,
        )

    if args.llm_method in ("vllm", "api", "openai"):
        selected_llm_func = vllm_generate_wrapper
    elif args.llm_method == "local":
        print(
            f"Loading local HF model from {args.llm_model_name!r} "
            f"(device={args.local_llm_device or 'auto'}, dtype={args.local_llm_dtype})..."
        )
        _local_model, _local_tokenizer, _local_dev = load_local_causal_lm(
            args.llm_model_name,
            device=args.local_llm_device,
            dtype_mode=args.local_llm_dtype,
        )
        _local_infer_sem = asyncio.Semaphore(max(1, args.local_llm_max_concurrent))

        async def local_generate_wrapper(input_message, max_tokens=2048, temperature=0., stop=None, disable_thinking=False):
            async with _local_infer_sem:
                return await asyncio.to_thread(
                    local_generate_sync,
                    input_message,
                    max_tokens,
                    temperature,
                    stop,
                    _local_model,
                    _local_tokenizer,
                    _local_dev,
                    disable_thinking,
                )

        selected_llm_func = local_generate_wrapper
    else:
        raise NotImplementedError(
            f"eval_bootstrap: llm_method={args.llm_method!r} is not implemented "
            "(use vllm, api, openai, or local)."
        )

    run_fn = run_value_infer
    run_tag = "value_infer"

    cfg_path = os.path.join("src/config", args.config_path)
    cfg = OmegaConf.load(cfg_path)

    # specify max tokens
    if args.max_tokens is not None:
        cfg.actor_rollout_ref.rollout.each_response_length = args.max_tokens
    
    if args.bootstrap_type == "first_state":
        if cfg.actor_rollout_ref.rollout.multi_turn.max_window_size <= 1:
            raise ValueError(f"Bootstrap type {args.bootstrap_type} requires max_window_size > 1")

        first_state_tag = []   # to record the game seed whether this rollout need regenerate the subsequent state understand
    

    _test_env, _, _ = build_env(cfg.actor_rollout_ref.rollout.multi_turn.envs.env_config, seed=None)
    # from tasks.classic_games.chess.chess_puzzle_env import ChessPuzzleEnv_Wrapper
    # _test_env = ChessPuzzleEnv_Wrapper(cfg.actor_rollout_ref.rollout.multi_turn.envs.env_config)
    total_num_game = _test_env.num_game
    train_val_split = int(
        cfg.actor_rollout_ref.rollout.multi_turn.envs.env_config.rollout.train_val_ratio * total_num_game)
    
    val_map_index = list(range(0, train_val_split))                 # the first few is the evaluation set
    train_map_index = list(range(train_val_split, total_num_game))    # the rest is the training set
    # train_map_index = [20]
    cutoff=args.cutoff
    if cutoff is not None:
        # val_map_index = val_map_index[:cutoff]
        train_map_index = train_map_index[:cutoff]    # cutoff on training set

    # print(f"game rating = {cfg.actor_rollout_ref.rollout.multi_turn.envs.env_config.rating_range}")
    print(f"\n\ntrain_map_index: {train_map_index}, val_map_index: {val_map_index}\n\n")
    print(
        f"Running with max_concurrency={args.max_concurrency}, "
        f"bootstrap_max_concurrency={bootstrap_max_concurrency}\n"
    )

    semaphore = asyncio.Semaphore(args.max_concurrency)
    bootstrap_semaphore = asyncio.Semaphore(bootstrap_max_concurrency)

    async def run_mdp_with_semaphore(
        seed,
        fn,
        knowledge_table=None,
        random_explore=False,
        no_knowledge_write=False,
        only_read_at_first_state=False,
    ):
        async with semaphore:
            result = await fn(
                cfg,
                seed,
                max_tokens=args.max_tokens,
                llm_generate_func=selected_llm_func,
                knowledge_table=knowledge_table,
                random_explore=random_explore,
                no_knowledge_write=no_knowledge_write,
                only_read_at_first_state=only_read_at_first_state,
            )
            return seed, result

    save_time = time.strftime("%Y%m%d_%H%M%S")

    model_tag = _safe_run_tag(args.llm_model_name)
    base_save_folder = f"evaluation/logs/eval_bootstrap/{args.llm_method}_{run_tag}_{cfg.actor_rollout_ref.rollout.multi_turn.envs.env_config.env_name}_{model_tag}_{save_time}"
    
    os.makedirs(base_save_folder, exist_ok=True)

    args_path = os.path.join(base_save_folder, "args.json")
     # Save args configuration
    args_dict = _redacted_args_dict(args)
    with open(args_path, "w") as f:
        json.dump(args_dict, f, indent=2)
    
    prompt_set_path = cfg.actor_rollout_ref.rollout.multi_turn.envs.env_config.prompt_set
    module_path, class_name = prompt_set_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    prompt_set = getattr(module, class_name)

    # global knowledge table for all epochs
    global_knowledge_table = StateKnowledge()
    global_buffer = TrajectoryBuffer()
    max_aggregated_traj_num = args.max_traj_num
    max_traj_horizon = args.max_traj_horizon
    successful_seeds: set[int] = set()
    all_train_seeds: list[int] = list(dict.fromkeys(train_map_index))
    total_bootstrap_num = 0

    async def run_bootstrap_with_semaphore(
        experience: dict,
        state_index: int,
        exp_idx_epoch: int,
    ):
        async with bootstrap_semaphore:
            return await bootstrap_knowledge_update_for_episode(
                experience,
                global_buffer=global_buffer,
                global_knowledge_table=global_knowledge_table,
                prompt_set=prompt_set,
                llm_generate_func=selected_llm_func,
                max_tokens=args.max_tokens,
                max_aggregated_traj_num=max_aggregated_traj_num,
                max_traj_horizon=max_traj_horizon,
                target_state_index=state_index,
                buffer_query_scope=args.buffer_query_scope,
                query_epoch_idx=exp_idx_epoch,
            )

    async def run_bootstrap_TD_with_semaphore(
        experience: dict,
        state_index: int,
        exp_idx_epoch: int,
        summarize: bool,
        td_step_size: int,
    ):
        async with bootstrap_semaphore:
            return await bootstrap_knowledge_update_for_step(
                experience,
                global_buffer=global_buffer,
                global_knowledge_table=global_knowledge_table,
                prompt_set=prompt_set,
                llm_generate_func=selected_llm_func,
                max_tokens=args.max_tokens,
                max_aggregated_traj_num=max_aggregated_traj_num,
                max_traj_horizon=args.max_traj_horizon,
                target_state_index=state_index,
                buffer_query_scope=args.buffer_query_scope,
                query_epoch_idx=exp_idx_epoch,
                summarize=summarize,
                td_step_size=td_step_size,
            )

    for exp_idx in range(args.epoch):

        ############## Rollout on all training set ############################
        # Create tasks for all seeds (must be recreated for each repetition)
        if args.rollout_only_on_failures:   # if one game seed has been successful, skip it
            seeds_to_run: list[int] = [
                s for s in all_train_seeds if s not in successful_seeds
            ]
        else:
            seeds_to_run: list[int] = all_train_seeds

        if args.bootstrap_type == "first_state":
            tasks = [
                run_mdp_with_semaphore(
                    _seed,
                    run_fn,
                    knowledge_table=global_knowledge_table,
                    random_explore=False,  # keep deterministic unless you re-enable idx-based toggling
                    only_read_at_first_state=(_seed in first_state_tag),
                )
                for _seed in seeds_to_run
            ]

        elif args.bootstrap_type == "full_traj":
            tasks = [
                run_mdp_with_semaphore(
                    _seed,
                    run_fn,
                    knowledge_table=global_knowledge_table,
                    random_explore=False,  # keep deterministic unless you re-enable idx-based toggling
                )
                for _seed in seeds_to_run
            ]
        else:
            raise ValueError(f"Invalid bootstrap type: {args.bootstrap_type}")

        if not tasks:
            print(
                f"All remaining seeds already have env_success=True; stopping exp_idx loop at exp_idx={exp_idx}."
            )
            break
        epoch_save_folder = os.path.join(base_save_folder, f"Epoch{exp_idx}")  #f"{base_save_folder}_Epoch{exp_idx}"
        
        rollout_log_path = os.path.join(epoch_save_folder, "rollout_log.json")
        rollout_result_path = os.path.join(epoch_save_folder, "rollout_result.json")
        
        # Ensure the directory exists
        os.makedirs(epoch_save_folder, exist_ok=True)
        
        # Create an asyncio lock for thread-safe file writes
        file_lock = asyncio.Lock()
        
        # Run all tasks in parallel with progress tracking
        all_message = []
        all_success_rate = []
        all_action_valid_ratio = []
        all_good_move_ratio = []
        all_other_stats = {}  # Dictionary to collect all stats from other_stats
        episode_rollout_knowledge_table = []
        rollout_token_usage = 0
        
        async def save_result_to_file(_info):
            """Save a single result to JSON file with locking"""
            async with file_lock:
                # Run file I/O in a thread to avoid blocking the event loop
                await asyncio.to_thread(_append_to_json_list, rollout_log_path, _info)
        
        def _append_to_json_list(file_path, data):
            """Helper function to append data to a JSON list file"""
            # Read existing data if file exists
            existing_data = []
            if os.path.exists(file_path):
                try:
                    with open(file_path, "r") as f:
                        existing_data = json.load(f)
                except (json.JSONDecodeError, FileNotFoundError):
                    existing_data = []
            
            # Append new data
            existing_data.append(data)
            
            # Write back to file
            with open(file_path, "w") as f:
                json.dump(existing_data, f, indent=2)
                f.flush()
        
        # Use asyncio.as_completed to show progress
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Rollouting"):
            _seed, (_turn_message, _env_success, _action_valid_ratio, other_stats, _knowledge_table) = await coro
            _info = {
                "messages": _turn_message,
                "env_success": _env_success,
                "action_valid_ratio": _action_valid_ratio,
                "seed": _seed,
                "exp_idx": exp_idx,
                "epoch_knowledge_table": _knowledge_table,
                **other_stats
            }
            all_message.append(_info)
            all_success_rate.append(_env_success)
            all_action_valid_ratio.append(_action_valid_ratio)
            all_good_move_ratio.append(other_stats['good_move_ratio'])
            episode_rollout_knowledge_table.append(
                {
                    "seed": _seed,
                    "env_success": _env_success,
                    "knowledge_table": _knowledge_table,
                }
            )
            rollout_token_usage += other_stats['total_token_usage']
            await global_buffer.ingest_from_episode_dict(_knowledge_table, exp_idx)
            if _env_success and _seed not in successful_seeds:
                successful_seeds.add(_seed)

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
        mean_good_move_ratio = np.mean(all_good_move_ratio)

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
                    else:
                        # If it's already a scalar, convert to float
                        scalar_values.append(float(val))
                mean_other_stats[f"mean_{key}"] = float(np.mean(scalar_values))
        
        # Save result.json with success rate and action valid ratio, merged with args
        result_data = {
            "episode_success_rate": float(mean_success_rate),
            "total_success_rate": len(successful_seeds) / len(all_train_seeds),
            "action_valid_ratio": float(mean_action_valid_ratio),
            "good_move_ratio": float(mean_good_move_ratio),
            "episode_games": len(all_success_rate),
            "episode_successful_games": int(np.sum(all_success_rate)),
            **mean_other_stats,
            "args": args_dict  # Merge args content into result.json
        }
        with open(rollout_result_path, "w") as f:
            json.dump(result_data, f, indent=2)
        
        # Copy config file to the folder
        config_filename = os.path.basename(cfg_path)
        config_dest_path = os.path.join(epoch_save_folder, config_filename)
        shutil.copy2(cfg_path, config_dest_path)

        print(f"### Epoch {exp_idx} finished ###")
        
        print(f"Mean episode success rate = {mean_success_rate}, action valid = {mean_action_valid_ratio}, good move = {mean_good_move_ratio}, total success rate = {len(successful_seeds) / len(all_train_seeds)}")
        if mean_other_stats:
            other_stats_str = ", ".join([f"{k} = {v}" for k, v in mean_other_stats.items()])
            print(f"Mean other stats: {other_stats_str}")
        

        print(f"Results saved to folder: {epoch_save_folder}")
        print("  - eval_log.json: Contains all evaluation logs")
        print("  - result.json: Contains success rate and action valid ratio")
        print("  - args.json: Contains command-line arguments configuration")
        print(f"  - {config_filename}: Configuration file")


        ############## Bootstrap on the collected trajectory #########################
        if args.bootstrap_method == "MC":
            total_knowledge_update_tokens = 0
            total_knowledge_update_messages = []
            first_state_tag = []  # clear the tag
            bootstrap_aws = []
            for experience_dict in episode_rollout_knowledge_table:
                if experience_dict.get("env_success"):
                    continue
                experience = experience_dict["knowledge_table"]
                if args.bootstrap_type == "full_traj":
                    for state_index in range(len(experience["state"]) - 1):
                        bootstrap_aws.append(
                            run_bootstrap_with_semaphore(experience, state_index, exp_idx)
                        )
                        total_bootstrap_num += 1
                elif args.bootstrap_type == "first_state":   # only bootstrap on the first state
                    bootstrap_aws.append(
                        run_bootstrap_with_semaphore(experience, 0, exp_idx)
                    )
                    total_bootstrap_num += 1
                    first_state_tag.append(experience_dict["seed"])
                    # print(f"number of first state bootstrap = {len(first_state_tag)}")
                else:
                    raise ValueError(f"Invalid bootstrap type: {args.bootstrap_type}")


            if bootstrap_aws:
                for coro in tqdm(
                    asyncio.as_completed(bootstrap_aws),
                    total=len(bootstrap_aws),
                    desc="Bootstrap",
                ):
                    tokens, knowledge_update_messages = await coro
                    total_knowledge_update_tokens += tokens
                    if knowledge_update_messages is not None:
                        total_knowledge_update_messages.append(knowledge_update_messages)

            # save the knowledge update messages
            knowledge_update_messages_path = os.path.join(epoch_save_folder, f"knowledge_update_messages.json")
            with open(knowledge_update_messages_path, "w") as f:
                json.dump(total_knowledge_update_messages, f, indent=2)
            
            # save global_knowledge_table
            global_knowledge_table_path = os.path.join(base_save_folder, "global_knowledge_table.json")
            if os.path.exists(global_knowledge_table_path):
                os.remove(global_knowledge_table_path)
            with open(global_knowledge_table_path, "w") as f:
                json.dump(global_knowledge_table.knowledge_table, f, indent=2)

        elif args.bootstrap_method == "TD":
            total_knowledge_update_tokens = 0
            total_knowledge_update_messages = []

            async def run_td_bootstrap_one_episode(
                experience: dict, task_idx: int
            ) -> tuple[int, list]:
                traj_tokens = 0
                traj_messages: list = []
                for state_index in range(len(experience["state"]) - 2, -1, -1):
                    # print(f"task_idx = {task_idx} state_index = {state_index}")
                    tokens, knowledge_update_messages = await run_bootstrap_TD_with_semaphore(
                        experience,
                        state_index,
                        exp_idx,
                        summarize=args.bootstrap_summarize,
                        td_step_size=args.bootstrap_TD_step_size,
                    )
                    traj_tokens += tokens
                    if knowledge_update_messages is not None:
                        traj_messages.append(knowledge_update_messages)
                return traj_tokens, traj_messages

            td_experiences: list[dict] = []
            for experience_dict in episode_rollout_knowledge_table:
                if experience_dict.get("env_success"):
                    continue
                td_experiences.append(experience_dict["knowledge_table"])

            if td_experiences:
                with tqdm(total=len(td_experiences), desc="Bootstrapping TD") as pbar:

                    async def _run_one_episode(exp: dict, task_idx: int) -> tuple[int, list]:
                        out = await run_td_bootstrap_one_episode(exp, task_idx)
                        pbar.update(1)
                        return out

                    episode_results = await asyncio.gather(
                        *(
                            _run_one_episode(exp, i)
                            for i, exp in enumerate(td_experiences)
                        )
                    )
                for traj_tokens, traj_messages in episode_results:
                    total_knowledge_update_tokens += traj_tokens
                    total_knowledge_update_messages.extend(traj_messages)
            
            # save the knowledge update messages
            knowledge_update_messages_path = os.path.join(epoch_save_folder, f"knowledge_update_messages_TD.json")
            with open(knowledge_update_messages_path, "w") as f:
                json.dump(total_knowledge_update_messages, f, indent=2)
        
        print(f"Total bootstrap num = {total_bootstrap_num}")

        
        ############ Test on train set for multiple times #########################
        if not args.run_evaluation:
            continue

        repeated_test_num_repeats = args.val_num_repeats
        val_seeds_to_run: list[int] = [
            s for s in all_train_seeds for _ in range(repeated_test_num_repeats)
        ]
        repeated_test_log_path = os.path.join(epoch_save_folder, "repeated_test_log.json")
        repeated_test_result_path = os.path.join(epoch_save_folder, "repeated_test_result.json")

        async def run_rt_with_idx(trial_idx: int, seed: int):
            s, r = await run_mdp_with_semaphore(
                seed,
                run_fn,
                knowledge_table=global_knowledge_table,
                random_explore=False,
                no_knowledge_write=True,
            )
            return trial_idx, s, r

        rt_tasks = [
            run_rt_with_idx(i, _seed) for i, _seed in enumerate(val_seeds_to_run)
        ]

        rt_success: list = []
        rt_action_valid: list = []
        rt_other_stats: dict = {}

        async def save_repeated_test_result(rt_info: dict):
            async with file_lock:
                await asyncio.to_thread(_append_to_json_list, repeated_test_log_path, rt_info)

        for coro in tqdm(
            asyncio.as_completed(rt_tasks),
            total=len(rt_tasks),
            desc="repeated_test train set",
        ):
            trial_idx, _seed, (
                _turn_message,
                _env_success,
                _action_valid_ratio,
                other_stats,
                _knowledge_table,
            ) = await coro
            rt_info = {
                "trial_idx": trial_idx,
                "messages": _turn_message,
                "env_success": _env_success,
                "action_valid_ratio": _action_valid_ratio,
                "seed": _seed,
                "exp_idx": exp_idx,
                "epoch_knowledge_table": _knowledge_table,
                **other_stats,
            }
            rt_success.append(_env_success)
            rt_action_valid.append(_action_valid_ratio)
            for key, value in other_stats.items():
                if key not in rt_other_stats:
                    rt_other_stats[key] = []
                rt_other_stats[key].append(value)
            await save_repeated_test_result(rt_info)

        rt_mean_success = float(np.mean(rt_success)) if rt_success else 0.0
        rt_mean_action_valid = float(np.mean(rt_action_valid)) if rt_action_valid else 0.0
        rt_mean_other_stats = {}
        for key, values in rt_other_stats.items():
            if values:
                scalar_values = []
                for val in values:
                    if isinstance(val, (list, tuple, np.ndarray)):
                        scalar_values.append(float(np.mean(val)))
                    else:
                        scalar_values.append(float(val))
                rt_mean_other_stats[f"mean_{key}"] = float(np.mean(scalar_values))

        print(f"### repeated_test (epoch {exp_idx}, train seeds x{repeated_test_num_repeats}) ###")
        print(
            f"  episodes={len(rt_success)}  "
            f"episode_success_rate={rt_mean_success:.4f}  "
            f"action_valid_ratio={rt_mean_action_valid:.4f}  "
            f"successful_episodes={int(np.sum(rt_success))}"
        )
        if rt_mean_other_stats:
            print(f"  mean other stats: {', '.join(f'{k} = {v}' for k, v in rt_mean_other_stats.items())}")

        repeated_test_result_data = {
            "prefix": "repeated_test",
            "episode_success_rate": rt_mean_success,
            "action_valid_ratio": rt_mean_action_valid,
            "episode_games": len(rt_success),
            "episode_successful_games": int(np.sum(rt_success)) if rt_success else 0,
            "repeats_per_seed": repeated_test_num_repeats,
            "unique_seeds": len(all_train_seeds),
            **rt_mean_other_stats,
            "args": args_dict,
        }
        with open(repeated_test_result_path, "w") as f:
            json.dump(repeated_test_result_data, f, indent=2)
        print(
            f"  repeated_test log -> {repeated_test_log_path}  "
            f"summary -> {repeated_test_result_path}"
        )


    n_train = len(all_train_seeds)
    if n_train > 0:
        seed_success_rate = float(len(successful_seeds)) / float(n_train)
        print(
            f"Overall success rate over all_train_seeds: "
            f"{len(successful_seeds)}/{n_train} = {seed_success_rate:.4f} "
            f"({100.0 * seed_success_rate:.2f}%)"
        )
    else:
        print("Overall success rate over all_train_seeds: N/A (no train seeds)")

    # at the end save the knowledge table and trajectory buffer
    with open(os.path.join(base_save_folder, "global_knowledge_table.json"), "w") as f:
        json.dump(global_knowledge_table.knowledge_table, f, indent=2)

    buffer_snapshot = await global_buffer.to_serializable_snapshot()
    with open(os.path.join(base_save_folder, "global_trajectory_buffer.json"), "w") as f:
        json.dump(buffer_snapshot, f, indent=2, default=str)

    print(f"All logs are save to {base_save_folder}")

if __name__ == "__main__":
    asyncio.run(main())
