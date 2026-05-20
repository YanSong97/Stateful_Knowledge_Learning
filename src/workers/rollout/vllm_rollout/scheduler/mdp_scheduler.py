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


class EnvChatCompletionScheduler(NaiveChatCompletionScheduler):
    """This is a official chat completion scheduler that supports env rollout
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
        self.query_prompt = prompt_set["query_prompt"]

        # env termination feedback
        self.success_prompt = prompt_set["success_prompt"]
        self.fail_prompt = prompt_set["fail_prompt"]

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

        # reasoning tag
        self.think_tag = ["<think>", "</think>"]
        self.answer_tag = ["<answer>", "</answer>"]

        self.conversation_prefix = "tool"
        self.max_turns=self.config.multi_turn.max_turns
        self.is_validate=None
        # seed generator
        train_seed = eval(str(self.config.multi_turn.envs.env_config.rollout.train_seed))
        val_seed = eval(str(self.config.multi_turn.envs.env_config.rollout.val_seed))
        self.train_seed_pool = SeedGenerator(start_range=train_seed[0], end_range=train_seed[1], shuffle=True)
        self.val_seed_pool = SeedGenerator(start_range=val_seed[0], end_range=val_seed[1], shuffle=True)

        self.each_response_max_length = self.config.each_response_length
        self.total_response_max_length = self.config.response_length


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
                _query = self.query_prompt.format(state=_state, available_move=_single_env.legal_moves_list, turn_idx=0, turn_left=self.max_turns)
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
            stop=[self.answer_tag[-1]],
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

        async def callback(completions: ChatCompletion, info: Dict[str, Any], exception: Exception):

            # ==================== assistant response ==============================================#
            batch_conversations, batch_traj, batch_token_n, batch_index, turn = (
                info["batch_conversations"],
                info["batch_traj"],
                info["batch_token_n"],
                info["batch_index"],
                info["turn"],
            )
            role, content = completions.choices[0].message.role, completions.choices[0].message.content
            batch_conversations[batch_index].append({"role": role, "content": content})
            token_usage = completions.usage.completion_tokens
            assert token_usage <= self.each_response_max_length, print(f"current token usage = {token_usage}, exceeding max {self.each_response_max_length}")
            batch_token_n[batch_index] += token_usage

            #=========================== Env Step Fn ====================================================#
            extracted_action, action_is_valid = self.extract_answer(content, available_actions=self.batched_env[batch_index].legal_moves_list)
            batch_traj[batch_index]['action'].append(extracted_action)
            batch_traj[batch_index]['action_is_valid'].append(action_is_valid)

            next_state, reward, done, info = self.batched_env[batch_index].step(extracted_action)
            if DEBUG:
                print(f"[id={completions.id},turn={turn}] Env stepping {next_state}")
            batch_traj[batch_index]['state'].append(next_state)
            batch_traj[batch_index]['reward'].append(reward)
            batch_traj[batch_index]['terminate'].append(done)

            #=========================== Check Termination ============================================#
            if turn == max_turns:
                batch_conversations[batch_index].append(
                    {"role": self.conversation_prefix,
                     "content": self.fail_prompt.format(state=next_state, fail_reason="because reaching maximum turns")}
                )     # final env return
                batch_traj[batch_index]['success'].append(0.)
                batch_traj[batch_index]['end_reason'].append("reach max turn")
                if DEBUG:
                    print(f"[id={completions.id},turn={turn}] Reach max turns {max_turns}, done!")
                return

            if batch_token_n[batch_index] >= self.total_response_max_length:
                done = True
                batch_traj[batch_index]['end_reason'].append("reach max response length")

            if done:
                # check success
                env_success = self.batched_env[batch_index].check_success()
                end_info = self.success_prompt.format(state=next_state) if env_success else self.fail_prompt.format(state=next_state, fail_reason="because failling into holes")
                if batch_token_n[batch_index] >= self.total_response_max_length and not env_success:
                    end_info = self.fail_prompt.format(state=next_state, fail_reason="because reaching maximum response length")

                batch_conversations[batch_index].append({"role": self.conversation_prefix, "content": end_info.format(state=next_state)})     # final env return
                batch_traj[batch_index]['success'].append(float(env_success))
                batch_traj[batch_index]['end_reason'].append('game terminate')
                if DEBUG:
                    print(f"[id={completions.id},turn={turn}] Env terminate, done!")
                return

            batch_conversations[batch_index].append(
                {"role": self.conversation_prefix,
                 "content": self.query_prompt.format(state=next_state, available_move='1,2,3,4', turn_idx=turn, turn_left=max_turns-turn)
                 }
            )
            if DEBUG:
                print(f"[id={completions.id},turn={turn}] Env continue ...")

            #=========================== Submit chat completions ========================================#
            extra_headers = {"x-request-id": completions.id}
            await self.submit_chat_completions(
                callback=callback,
                callback_additional_info={
                    "batch_conversations": batch_conversations,
                    "batch_traj": batch_traj,
                    "batch_token_n": batch_token_n,
                    "batch_index": batch_index,
                    "turn": turn + 1,
                },
                model=self.model_name,
                messages=batch_conversations[batch_index],
                extra_headers=extra_headers,
                **kwargs,
            )

        # repeat batch to parallelize envs
        gen_batch = batch.repeat(repeat_times=repeat_num, interleave=True)
        tasks, batch_conversations, batch_traj = [], [None] * len(gen_batch), [defaultdict(list) for _ in range(len(gen_batch))]
        batch_token_n = [0 for _ in range(len(gen_batch))]

        # build local batched env
        self.build_env(n=len(batch), repeat=repeat_num, interleave=True)        # for each original samples, repeat

        # replace raw prompt with sys_prompt + initial_state query TODO[yan]: can we do it outside the scheduler?
        new_raw_prompt = []
        for batch_index, conversation in enumerate(gen_batch.non_tensor_batch["raw_prompt"]):
            sys_p = {"role": "system", "content": self.system_prompt}
            conversation[0]['content'] += self.batched_state_query[batch_index]
            new_raw_prompt.append([sys_p, conversation[0]])
        gen_batch.non_tensor_batch["processed_prompt"] = np.array(new_raw_prompt)       # TODO[yan]: no need to replace raw_prompt ?

        for batch_index, conversation in enumerate(gen_batch.non_tensor_batch["processed_prompt"]):
            # raw_prompt: [{"role": "user", "content": ""}, ["role": "assistant", "content"], ...]
            batch_conversations[batch_index] = conversation.tolist()        # need to contain complete prompt and query
            batch_traj[batch_index]['state'].append(self.batched_state[batch_index])

            tasks.append(
                asyncio.create_task(
                    self.submit_chat_completions(
                        callback=callback,
                        callback_additional_info={
                            "batch_conversations": batch_conversations,
                            "batch_traj": batch_traj,
                            "batch_token_n": batch_token_n,
                            "batch_index": batch_index,
                            "turn": 1,
                        },
                        model=self.model_name,
                        messages=batch_conversations[batch_index],
                        **kwargs,
                    )
                )
            )

        await asyncio.gather(*tasks)
        if DEBUG:
            print("[EnvChatChatCompletionScheduler] generate_sequences done")
            print(f"\nTraj = {batch_traj}\n")

        # _postprocess assumes n>=1
        batch_conversations = [[conversation] for conversation in batch_conversations]
        return self._postprocess(gen_batch, batch_conversations, repeat_num, batch_traj)


    def _postprocess(self, batch: DataProto, batch_conversations: List[List[Dict[str, str]]], n: int, traj) -> DataProto:
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


        #=========================== Processing Response =============================#
        # sequences = [self.tokenizer.apply_chat_template(conversation, add_generation_prompt=False, tokenize=False) for conversation in batch_conversations]
        response_input_ids = []
        response_attention_mask = []
        response_loss_mask = []
        response_reward = []
        raw_env_reward = [i["reward"] for i in traj]
        for idx, conversation in enumerate(batch_conversations):
            ret = self.customize_chat_template(conversation=conversation,
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
        non_tensor_batch['success_rate'] = np.array(success_rate)
        non_tensor_batch['action_valid_rate'] = np.stack(action_valid_rate)

        ret = DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
        ret.check_consistency()

        print(f"========")
        print(f"Avg success rate = {np.mean(success_rate)}")
        print(f"========")

        return ret

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
                    if self.config.multi_turn.post_process.process_reward:
                        assert len(env_reward_lst) == turn_reward_idx, print(f"env reward list number {len(env_reward_lst)} does not align with reward turn counts {turn_reward_idx}")

                    assert (torch.tensor(assistant_masks)*torch.tensor(assistant_reward_lst)).sum() == sum(env_reward_lst), print(f"masked reward does not align with raw env reward")

                    if return_tensors:
                        out.convert_to_tensors(tensor_type=return_tensors)

                return out
            else:
                return out["input_ids"]
        else:
            return rendered_chat