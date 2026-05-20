import asyncio
from typing import Any, Dict, List, Optional

import torch
from tensordict import TensorDict
from dataclasses import dataclass, field

from verl.protocol import DataProto
from verl.single_controller.ray.base import RayWorkerGroup

@dataclass
class EnvStatus:
    """Status of an environment"""
    truncated: bool = False # done but not success
    terminated: bool = False # done and success
    num_actions: int = 0 # current action step (single action)
    rewards: List[float] = field(default_factory=list) # rewards for each turn
    seed: Optional[int] = None # what seed is used to reset this environment


class RAGENChatCompletionScheduler:
    def __init__(self, config, model_path, actor_wg, mode='train'):
        self.config = config

        env_config = getattr(self.config.es_manager, mode)

        self.env_groups = int(env_config.env_groups)
        self.group_size = env_config.group_size
        self.rollout_cache = None

        # context manager
        self.model_name = "/".join(model_path.split("/")[-2:])

        from verl.utils.fs import copy_to_local
        from verl.utils import hf_processor, hf_tokenizer

        local_path = copy_to_local(model_path)
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=True)
        self.action_sep = self.config.agent_proxy.action_sep
        self.special_token_list = ["<think>", "</think>", "<answer>", "</answer>", "<|im_start|>", "<|im_end|>"]

        self.model_path = model_path
        self.actor_wg = actor_wg


    def _init_prefix_lookup(self):
        prefix_lookup = {}
        prefixes = {}
        env_config_lookup = {}
        env_config = {}
        for env_tag, env_config in self.config.custom_envs.items():
            if env_tag not in self.es_cfg.env_configs.tags:
                continue
            env_config_new = asdict(REGISTERED_ENV_CONFIGS[env_config.env_type]())
            for k, v in env_config.items():
                env_config_new[k] = v
            env_instruction = env_config_new.get("env_instruction", "")
            if env_config_new.get("grid_vocab", False):
                grid_vocab_str = "\nThe meaning of each symbol in the state is:\n" + ", ".join(
                    [f"{k}: {v}" for k, v in env_config_new["grid_vocab"].items()])
                env_instruction += grid_vocab_str
            if env_config_new.get("action_lookup", False):
                action_lookup_str = "\nYour available actions are:\n" + ", ".join(
                    [f"{v}" for k, v in env_config_new["action_lookup"].items()])
                action_lookup_str += f"\nYou can make up to {env_config_new['max_actions_per_traj']} actions, separated by the action separator \" " + self.action_sep + " \"\n"
                env_instruction += action_lookup_str
            prefixes[env_tag] = env_instruction
            env_config_lookup[env_tag] = {
                'max_tokens': env_config.get("max_tokens", self.config.actor_rollout_ref.rollout.response_length)}

        tags = self.es_cfg.env_configs.tags
        n_groups = self.es_cfg.env_configs.n_groups
        group_size = self.es_cfg.group_size

        cur_group = 0
        for env_tag, n_group in zip(tags, n_groups):
            env_instruction = prefixes[env_tag]
            start_idx = cur_group * group_size
            end_idx = (cur_group + n_group) * group_size
            for i in range(start_idx, end_idx):
                prefix_lookup[i] = env_instruction
                env_config_lookup[i] = env_config_lookup[env_tag]
            cur_group += n_group

        self.prefix_lookup = prefix_lookup
        self.env_config_lookup = env_config_lookup

    def get_lm_inputs(self, env_outputs: List[Dict], prepare_for_update: bool) -> DataProto:
        llm_input_texts = []
        messages_list = []  # for api calling
        for env_output in env_outputs:
            if 'state' in env_output['history'][-1] and prepare_for_update:
                env_output['history'] = env_output['history'][
                                        :-1]  # when prepare for update, we do not add the state from the n+1 turn to the trajectory
            messages = [
                {"role": "system", "content": f"You're a helpful assistant. "},
                {"role": "user", "content": self.prefix_lookup[env_output["env_id"]]}
            ]

            for idx, content in enumerate(env_output["history"]):
                messages[-1]["content"] += f"\nTurn {idx + 1}:\n"
                if "state" in content:
                    FORMAT_PROMPT = "<think> [Your thoughts] </think> <answer> [your answer] </answer>" if self.config.agent_proxy.enable_think else "<answer> [your answer] </answer>"
                    LENGTH_PROMPT = f"Max response length: {self.env_config_lookup[env_output['env_id']]['max_tokens']} words (tokens)."
                    messages[-1][
                        "content"] += f"State:\n{content['state']}\nYou have {content['actions_left']} actions left. Always output: {FORMAT_PROMPT} with no extra text. Strictly follow this format. {LENGTH_PROMPT}\n"
                if "llm_response" in content:
                    messages.append({"role": "assistant", "content": content["llm_response"]})
                if "reward" in content and not (prepare_for_update and idx == len(env_output["history"]) - 1):
                    # when prepare for update, we do not add the reward from the n+1 turn to the trajectory
                    messages.append({"role": "user", "content": f"Reward:\n{content['reward']}\n"})

            # NOTE: this assertion is important for loss mask computation
            assert all(msg["role"] == "assistant" for msg in messages[2::2])

            text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=(not prepare_for_update),
                                                      tokenize=False)
            if not prepare_for_update:
                if self.config.agent_proxy.enable_think:
                    text += "<think>"  # force the LLM to think before answering
                else:
                    text += "<answer>"  # force the LLM to answer
            llm_input_texts.append(text)
            messages_list.append(messages)

        inputs = self.tokenizer(llm_input_texts, return_tensors="pt", padding=True, padding_side="left",
                                truncation=False)  # do not truncate here. Process later at TODO
        input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
        position_ids = attention_mask.cumsum(dim=-1)
        if prepare_for_update:
            scores = [[i['reward'] for i in env_output['history']] for env_output in env_outputs]
            loss_mask, score_tensor, response_mask = get_masks_and_scores(input_ids, self.tokenizer, scores,
                                                                          use_turn_scores=self.config.agent_proxy.use_turn_scores)
            if self.config.enable_response_mask:
                loss_mask = response_mask
            normalized_score_tensor = self._normalize_score_tensor(score_tensor, env_outputs)
            response_length = response_mask.sum(dim=-1).float().mean().item()

        llm_inputs = DataProto()
        llm_inputs.batch = TensorDict({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": input_ids[:, 1:],  # remove the first token
        }, batch_size=input_ids.shape[0])

        if prepare_for_update:
            llm_inputs.batch["loss_mask"] = loss_mask  # remove the first token
            llm_inputs.batch["rm_scores"] = normalized_score_tensor  # remove the first token
            llm_inputs.batch["original_rm_scores"] = score_tensor  # remove the first token
        llm_inputs.non_tensor_batch = {
            "env_ids": np.array([env_output["env_id"] for env_output in env_outputs], dtype=object),
            "group_ids": np.array([env_output["group_id"] for env_output in env_outputs], dtype=object),
            "messages_list": np.array(messages_list, dtype=object),
        }

        if prepare_for_update:
            metrics = {}
            for env_output in env_outputs:
                for key, value in env_output["metrics"].items():
                    if key not in metrics:
                        metrics[key] = []
                    metrics[key].append(value)
            metrics = {
                key: np.sum(value) / self.env_nums[key.split("/")[0]]
                for key, value in metrics.items()
            }
            metrics["response_length"] = response_length
            llm_inputs.meta_info = {"metrics": metrics}
        return llm_inputs

    def get_env_inputs(self, lm_outputs: DataProto) -> List[Dict]:
        if lm_outputs.batch is not None and 'responses' in lm_outputs.batch.keys():
            responses = self.tokenizer.batch_decode(
                lm_outputs.batch['responses'],
                skip_special_tokens=True
            )
        else:  # dataproto has textual responses
            responses = lm_outputs.non_tensor_batch['response_texts']
        responses = ["<think>" + response if self.config.agent_proxy.enable_think else "<answer>" + response for
                     response in responses]  # The LLM generation does not include <think> tags. Add them back here.

        env_ids = lm_outputs.non_tensor_batch['env_ids']
        env_inputs = []
        for env_id, response in zip(env_ids, responses):
            llm_response, actions = self._parse_response(response)
            env_inputs.append({
                "env_id": env_id,
                "llm_raw_response": response,
                "llm_response": llm_response,
                "actions": actions,
            })
        return env_inputs

    def env_step(self, all_env_inputs: List[Dict]):
        """Step the environments.
        1. extract valid actions from the action lookup table (if exists) and execute the actions, and update rollout cache
        2. Since rollout does not need to act over done envs, whenever the environment is done, we only update rollout cache, but not output env_outputs.
        Input:
        all_env_inputs: List[Dict]
            {env_id: int, llm_response: str, actions: List[str]}
            NOTE: should use env_id as index for existing some already done envs
        env_outputs: List[Dict]
            {env_id: int, history: List[Dict][{state: str, actions: List[str], reward: float, info: Dict, llm_response: str, llm_raw_response: str, (Optional)images: List[PIL.Image.Image]}]}
        """
        def _execute_actions(env, actions):
            acc_reward, turn_info, turn_done = 0, {}, False
            executed_actions = []
            for action in actions:
                _, reward, done, info = env.step(action)
                acc_reward += reward
                turn_info.update(info) # NOTE: currently use last info for multi-action
                executed_actions.append(action)
                if done:
                    turn_done = True
                    break
            return acc_reward, turn_info, turn_done, executed_actions

        def _log_env_state(status, history, cur_obs, max_actions_per_traj, executed_actions, all_actions, acc_reward, turn_done, turn_info, env_input):
            obs = self._handle_mm_state(cur_obs)
            status.num_actions += len(executed_actions)
            status.rewards.append(acc_reward) # NOTE use turn-wise acc_reward
            actions_left = max_actions_per_traj - status.num_actions
            if turn_done:
                status.terminated = True # TODO check terminated definition in gymnasium
                status.truncated = not turn_info.get('success', False)
            history = self._update_cache_history(history, next_state=obs, actions_left=actions_left, num_actions_info={
                'actions': executed_actions, 'reward': acc_reward, 'info': turn_info,
                'llm_response': env_input['llm_response'], 'llm_raw_response': env_input['llm_raw_response']
            })
            # filter out invalid actions
            # history = [content for content in history[:-1] if content['actions']] + [history[-1]]
            return status, history

        envs = self.batched_env
        env_outputs = []

        for env_input in all_env_inputs:
            acc_reward, turn_info, turn_done = 0, {}, False
            entry = envs[env_input['env_id']]
            env_id, env = entry['env_id'], entry['env']
            actions_left_before = entry['max_actions_per_traj'] - entry['status'].num_actions

            # execute actions in envs
            valid_actions = self._extract_map_valid_actions(entry, env_input['actions'])
            acc_reward, turn_info, turn_done, executed_actions = _execute_actions(env,
                                                                                  valid_actions[:actions_left_before])
            if len(valid_actions) != len(env_input['actions']) or not valid_actions:
                self.rollout_cache[env_id]["penalty"] += self.sys_config.es_manager.format_penalty

            status, history = _log_env_state(entry['status'], self.rollout_cache[env_id]['history'],
                                             entry['env'].render(), entry['max_actions_per_traj'], executed_actions,
                                             valid_actions, acc_reward, turn_done, turn_info, env_input)
            entry['status'] = status
            if entry['status'].num_actions >= entry['max_actions_per_traj'] and not turn_done:
                entry['status'].truncated = True
                entry['status'].terminated = True
                turn_done = True
            self.rollout_cache[env_id]['history'] = history
            if not turn_done:  # NOTE done environments are not sent for further llm generation (for efficiency)
                env_outputs.append(self.rollout_cache[env_id])

        return env_outputs


    def submit_chat_completions(self, lm_inputs: DataProto):
        if isinstance(self.actor_wg, RayWorkerGroup):
            padded_lm_inputs, pad_size = pad_dataproto_to_divisor(lm_inputs, self.actor_wg.world_size)
            padded_lm_outputs = self.actor_wg.generate_sequences(padded_lm_inputs)
            lm_outputs = unpad_dataproto(padded_lm_outputs, pad_size=pad_size)
            lm_outputs.meta_info = lm_inputs.meta_info
            lm_outputs.non_tensor_batch = lm_inputs.non_tensor_batch
        elif isinstance(self.actor_wg, VllmWrapperWg) or isinstance(self.actor_wg, ApiCallingWrapperWg):
            lm_outputs = self.actor_wg.generate_sequences(lm_inputs)
        else:
            raise ValueError(f"Unsupported actor worker type: {type(self.actor_wg)}")

        return lm_outputs

    def build_env(self):
        env_list = []
        done_groups = 0
        for tag, n_group in zip(self.config.env_configs.tags, self.config.env_configs.n_groups):
            for env_id in range(done_groups * self.group_size, (done_groups + n_group) * self.group_size):
                cfg_template = self.sys_config.custom_envs[tag]
                max_actions_per_traj = cfg_template.max_actions_per_traj

                env_obj = get_env(self.env_config)

                entry = {'tag': tag, 'group_id': env_id // self.group_size, 'env_id': env_id,
                        'env': env_obj, 'config': env_config, 'status': EnvStatus(), 'max_actions_per_traj': max_actions_per_traj}
                env_list.append(entry)
            done_groups += n_group
        self.batched_env = env_list

    def reset_env(self, seed=None):
        def _expand_seed(seed: int):
            seeds = [[seed + i] * self.group_size for i in range(self.env_groups)] # [[seed, ..., seed], [seed+1, ..., seed+1], ...]
            return sum(seeds, [])

        envs = self.batched_env
        rollout_cache = [{"env_id": entry['env_id'], "history": [], "group_id": entry['group_id'], "tag": entry['tag'], "penalty": 0} for entry in envs]

        # reset all environments
        if self.mode == "train":
            seed = random.randint(0, 1000000) if seed is None else seed  # get a random seed
        else:
            seed = 123
        seeds = _expand_seed(seed)
        for seed, entry in zip(seeds, envs):
            entry['env'].reset(seed=seed)
            entry['status'] = EnvStatus(seed=seed)

        # update rollout cache
        for cache, env in zip(rollout_cache, envs):
            next_state = self._handle_mm_state(env['env'].render())
            cache['history'] = self._update_cache_history(cache['history'], next_state=next_state,
                                                          actions_left=env['max_actions_per_traj'],
                                                          num_actions_info=None)

        self.rollout_cache = rollout_cache
        return rollout_cache


    def generate_sequences(self, batch: DataProto, **sampling_params) -> DataProto:
        """
        Rollout
        Args:
            batch:
            **sampling_params:

        Returns:

        """

        kwargs = dict(
            n=self.config.n,
            max_completion_tokens=self.config.response_length,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )

        do_sample = batch.meta_info.get("do_sample", True)
        is_validate = batch.meta_info.get("validate", False)
        if not do_sample or is_validate:
            kwargs["n"] = 1
            kwargs["temperature"] = 0

        kwargs.update(sampling_params)
        print(f"[NaiveChatCompletionScheduler] generate_sequences sampling params: {kwargs}")

        self.build_env()

        import pdb
        pdb.set_trace()

        env_outputs = self.reset_env()
        for i in range(self.config.agent_proxy.max_turn):
            # get context
            lm_inputs: DataProto = self.get_lm_inputs(env_outputs, prepare_for_update=False)
            lm_inputs.meta_info = dataproto.meta_info
            lm_outputs: DataProto = self.submit_chat_completions(lm_inpyts)
            env_inputs: List[Dict] = self.get_env_inputs(lm_outputs)
            env_outputs: List[Dict] = self.env_step(env_inputs)
            if len(env_outputs) == 0:
                break

        rollout_states = self.get_rollout_states()
        rollouts = self.formulate_rollouts(rollout_states)

        return rollouts



    def get_rollout_states(self):
        """Get the final output for all environment"""
        envs = self.batched_env
        rollout_cache = self.rollout_cache

        # add metrics to rollout cache
        for entry, cache in zip(envs, rollout_cache):
            status = entry['status']
            env_metric = {
                'success': float(status.terminated and (not status.truncated)),
                'num_actions': status.num_actions,
            }
            custom_metric = {}
            for turn in cache['history']:
                for k, v in turn.get('info', {}).items():
                    if k == 'success':
                        continue
                    if k not in custom_metric:
                        custom_metric[k] = []
                    custom_metric[k].append(float(v))
            for k, v in custom_metric.items():
                env_metric[k] = np.sum(v) / (len(cache['history']) - 1) # NOTE: exclude the last observation

            cache['history'][-1]['metrics'] = custom_metric
            env_metric = {f"{entry['tag']}/{k}": v for k, v in env_metric.items()}
            cache['metrics'] = env_metric
        return rollout_cache

    def formulate_rollouts(self, env_outputs: List[Dict]) -> DataProto:
        llm_inputs = self.get_lm_inputs(env_outputs, prepare_for_update=True)
        return llm_inputs