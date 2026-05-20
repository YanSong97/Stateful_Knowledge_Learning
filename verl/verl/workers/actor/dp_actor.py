# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import logging
import os
from typing import Optional, Any

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, get_policy_loss_fn, kl_penalty
from verl.utils.device import get_device_name, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs, slice_input_tensor
from verl.workers.actor import BasePPOActor

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def calculate_sum_pi_squared_from_logits(logits: torch.Tensor):
    """
    Compute exact sum of squared probabilities from logits.
    Formula: Σπ² = exp(logsumexp(2*logits) - 2*logsumexp(logits))

    Used for optimal baseline variance reduction as described in
    "What Matters for Model Merging at Scale?" (arXiv:2410.03617)

    Args:
        logits: Logits tensor (..., vocab_size).

    Returns:
        Sum of squared probabilities tensor (...).
    """
    return torch.exp(torch.logsumexp(2.0 * logits, dim=-1) - 2.0 * torch.logsumexp(logits, dim=-1))


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None, sft_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.sft_optimizer = sft_optimizer

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()

        # Sum of squared probabilities computation (for optimal_token_baseline)
        # Only initialize if calculate_sum_pi_squared config is enabled
        if self.config.get("calculate_sum_pi_squared", False):
            self.calculate_sum_pi_squared_from_logits = (
                torch.compile(calculate_sum_pi_squared_from_logits, dynamic=True)
                if self.config.get("use_torch_compile", True)
                else calculate_sum_pi_squared_from_logits
            )
            assert not (self.use_fused_kernels), (
                "calculate_sum_pi_squared is not supported with "
                f"{self.use_fused_kernels=} for now."
            )

    def _forward_micro_batch(
        self, 
        micro_batch, 
        temperature, 
        calculate_entropy=False,
        return_all_logps: bool = False,
        distill_topk: Optional[int] = None,
        topk_indices: Optional[torch.Tensor] = None,
        debug: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        or a dict if return_all_logps is True
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            if "image_bound" in micro_batch["multi_modal_inputs"][0]:  # minicpm-o logic
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = [inputs[key] for inputs in micro_batch["multi_modal_inputs"]]
            else:
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = torch.cat(
                        [inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0
                    )

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)
                if debug:
                    import pdb; pdb.set_trace()

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch.keys()
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            
            outputs = {"log_probs": log_probs}
            if calculate_entropy:
                outputs['entropys'] = entropy

            return outputs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm
    
    def _sft_optimizer_step(self):
        assert self.config.sft_loss.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.sft_loss.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.sft_loss.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.sft_loss.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN (SFT): rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.sft_optimizer.zero_grad()
        else:
            self.sft_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                outputs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(outputs["log_probs"])
            if calculate_entropy:
                entropy_lst.append(outputs["entropys"])

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        current_global_step = data.meta_info["global_steps"]
        distill_per_global_step = self.config.distill_per_global_step
        if self.config.policy_loss.get("loss_mode", "vanilla") == "distillation":
            assert distill_per_global_step == 1, print(f"In distillation only mode, distill per global step must be 1, but got {distill_per_global_step}")
        
        update_this_epoch = (current_global_step % distill_per_global_step == 0)
        print(f"## current_global_step: {current_global_step}, distill_per_global_step: {distill_per_global_step}, update_this_epoch: {update_this_epoch}")

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        
        if self.config.include_distillation_loss:
            distillation_select_keys = {
                # teacher
                "distill_teacher_input_ids",
                "distill_teacher_attention_mask",
                "distill_teacher_position_ids",
                "distill_teacher_response",
                "distill_teacher_response_mask",
                # student
                "distill_student_input_ids",
                "distill_student_attention_mask",
                "distill_student_position_ids",
                "distill_student_response",
                "distill_student_response_mask",
            }
            assert distillation_select_keys.issubset(set(data.batch.keys())), f"Missing required keys: {distillation_select_keys - set(data.batch.keys())}"

            select_keys.extend(distillation_select_keys)

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for ppo_epoch in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()
                for idx, micro_batch in enumerate(micro_batches):
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = (
                        self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    )
                    clip_ratio_high = (
                        self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    )
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode
                            
                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")

                    if loss_mode == 'distillation':    # this mean pure distillation, skip policy gradient computation
                        assert self.config.include_distillation_loss, "Distillation loss is not included"
                        log_prob = None
                    else:
                        # calculate log_prob and entropy with gradients, all return: (bsz, response_length)
                        calculate_entropy = False
                        if entropy_coeff != 0:
                            calculate_entropy = True
                        outputs = self._forward_micro_batch(
                            model_inputs, 
                            temperature=temperature, 
                            calculate_entropy=calculate_entropy,
                            debug=False,
                        )
                        log_prob = outputs['log_probs']
                        if calculate_entropy:
                            entropy = outputs['entropys']
                        
                        if self.config.policy_loss.loss_mode == "vanilla":
                            # for NLRL-RL Loss
                            pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                                old_log_prob=old_log_prob,
                                log_prob=log_prob,
                                advantages=advantages,
                                response_mask=response_mask,
                                cliprange=clip_ratio,
                                cliprange_low=clip_ratio_low,
                                cliprange_high=clip_ratio_high,
                                clip_ratio_c=clip_ratio_c,
                                loss_agg_mode=loss_agg_mode,
                            )
                        else:
                            policy_loss_fn = get_policy_loss_fn(loss_mode)
                            pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                                old_log_prob=old_log_prob,
                                log_prob=log_prob,
                                advantages=advantages,
                                response_mask=response_mask,
                                loss_agg_mode=loss_agg_mode,
                                config=self.config,
                            )


                        if entropy_coeff != 0:
                            entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                            # compute policy loss
                            policy_loss = pg_loss - entropy_loss * entropy_coeff
                        else:
                            policy_loss = pg_loss

                        if self.config.use_kl_loss:
                            ref_log_prob = model_inputs["ref_log_prob"]
                            # compute kl loss
                            kld = kl_penalty(
                                logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                            )
                            kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                            policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                            micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item()
                            micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef
                        
                        # print(f"policy_loss: {policy_loss}")

                        micro_batch_metrics.update(
                            {
                                "actor/pg_loss": pg_loss.detach().item(),
                                "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                                "actor/ppo_kl": ppo_kl.detach().item(),
                                "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                            }
                        )

                    if self.config.include_distillation_loss and update_this_epoch:
                        if self.config.distillation_mode == "supervised-sd":
                            distillation_loss, distillation_metrics = self._compute_ssd_distillation_loss(
                                inputs=model_inputs,
                                temperature=temperature,
                                input_log_prob=log_prob.detach() if log_prob is not None else log_prob,
                                old_log_prob=old_log_prob.detach(),   # also need to correct offpolicyness of old_log_prob
                            )
                        elif self.config.distillation_mode == "op-sd":
                            distillation_loss, distillation_metrics = self._compute_op_sd_distillation_loss(
                                inputs=model_inputs,
                                temperature=temperature,
                                input_log_prob=log_prob if log_prob is not None else log_prob,  # here no need to detach
                                old_log_prob=old_log_prob.detach(),   # also need to correct offpolicyness of old_log_prob
                            )
                        else:
                            raise ValueError(f"Invalid distillation mode: {self.config.distillation_mode}")

                        distillation_loss = distillation_loss.float() * self.config.distillation_loss_coef
                        # distill_student_response_mask = model_inputs['distill_student_response_mask']

                        micro_batch_metrics["actor/distillation_loss"] = distillation_loss.detach().item()
                        micro_batch_metrics["actor/distillation_loss_coef"] = self.config.distillation_loss_coef
                        micro_batch_metrics.update(distillation_metrics)

                        auxiliary_loss = distillation_loss * self.config.distillation_loss_coef
    
                    append_to_dict(metrics, micro_batch_metrics)

                    if loss_mode == "distillation":
                        # only the distillation loss
                        integrated_loss = auxiliary_loss
                    elif loss_mode == "vanilla":
                        integrated_loss = policy_loss
                        if self.config.include_distillation_loss and update_this_epoch:
                            integrated_loss = integrated_loss + auxiliary_loss
                    else:
                        raise ValueError(f"Invalid loss mode: {loss_mode}")
                    
                    if self.config.use_dynamic_bsz:
                        loss = integrated_loss * (response_mask.shape[0] / self.config.ppo_mini_batch_size)
                    else:
                        loss = integrated_loss / self.gradient_accumulation
                    
                    loss.backward()

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
                del grad_norm
                torch.cuda.empty_cache()
        self.actor_optimizer.zero_grad()
        return metrics
    
    def _compute_ssd_distillation_loss(self, inputs, temperature, input_log_prob, old_log_prob):

        assert self.config.include_distillation_loss, "Distillation loss is not included"
        self_distillation_cfg = getattr(self.config, "self_distillation", None)
        return_all_logps = self_distillation_cfg.full_logit_distillation and not self_distillation_cfg.distillation_topk
        distill_topk = self_distillation_cfg.distillation_topk if self_distillation_cfg.full_logit_distillation else None

        ####### For auxiliary distillation loss
        distill_teacher_response_mask = inputs['distill_teacher_response_mask']
        distill_student_response_mask = inputs['distill_student_response_mask']

        # forward pass on transformed student traj
        student_inputs = {
            "responses": inputs['distill_student_response'],
            "input_ids": inputs['distill_student_input_ids'],
            "attention_mask": inputs['distill_student_attention_mask'],
            "position_ids": inputs['distill_student_position_ids'],
        }

        student_outputs = self._forward_micro_batch_v2(
            student_inputs, 
            temperature=temperature, 
            calculate_entropy=False,
            return_all_logps=return_all_logps,
            distill_topk=distill_topk,
            debug=False
        )
        student_log_probs = student_outputs['log_probs']

        student_all_logps = student_outputs.get("all_logps") if return_all_logps else None
        student_topk_logps = student_outputs.get("topk_logps") if distill_topk is not None else None
        student_topk_indices = student_outputs.get("topk_indices") if distill_topk is not None else None

        if input_log_prob is None:     
            teacher_inputs = {
                "responses": inputs['responses'],
                "input_ids": inputs['input_ids'],
                "attention_mask": inputs['attention_mask'],
                "position_ids": inputs['position_ids'],
            }
            with torch.no_grad():
                teacher_outputs = self._forward_micro_batch_v2(
                    teacher_inputs, 
                    temperature=temperature, 
                    calculate_entropy=False,
                    return_all_logps=return_all_logps,
                    distill_topk=distill_topk,
                    topk_indices=student_topk_indices,
                    disable_gradient=False
                )
            teacher_log_probs = teacher_outputs['log_probs']
            teacher_all_logps = teacher_outputs.get("all_logps") if return_all_logps else None
            teacher_topk_logps = teacher_outputs.get("topk_logps") if distill_topk is not None else None
        else:
            # re-use it
            teacher_log_probs = input_log_prob
            teacher_all_logps = None
            teacher_topk_logps = None
        
        # compute distillation loss
        distill_kl_loss, distillation_metrics = self._ssd_distillation_loss(
            student_log_probs=student_log_probs,
            teacher_log_probs=teacher_log_probs,
            self_distillation_config=self_distillation_cfg,
            student_all_log_probs=student_all_logps,
            teacher_all_log_probs=teacher_all_logps,
            student_topk_log_probs=student_topk_logps,
            teacher_topk_log_probs=teacher_topk_logps,
            teacher_response_mask=distill_teacher_response_mask,
            student_response_mask=distill_student_response_mask,
            old_log_prob=old_log_prob,
        )
        return distill_kl_loss, distillation_metrics


    def _ssd_distillation_loss(
        self,
        student_log_probs: torch.Tensor,
        teacher_log_probs: torch.Tensor,
        self_distillation_config: Any,
        student_all_log_probs: Optional[torch.Tensor] = None,
        teacher_all_log_probs: Optional[torch.Tensor] = None,
        student_topk_log_probs: Optional[torch.Tensor] = None,
        teacher_topk_log_probs: Optional[torch.Tensor] = None,
        teacher_response_mask: Optional[torch.Tensor] = None,
        student_response_mask: Optional[torch.Tensor] = None,
        old_log_prob: Optional[torch.Tensor] = None,
    ):
        metrics = {}

        if self_distillation_config.full_logit_distillation:
            use_topk = self_distillation_config.distillation_topk is not None
            if use_topk:
                if student_topk_log_probs is None or teacher_topk_log_probs is None:
                    raise ValueError("top-k distillation requires student_topk_log_probs and teacher_topk_log_probs")
                def add_tail(log_probs: torch.Tensor) -> torch.Tensor:
                    # Compute tail log-probability using logsumexp for numerical stability
                    # log(1 - sum(p_i)) = log(1 - exp(log_sum_exp(log(p_i))))
                    log_s = torch.logsumexp(log_probs, dim=-1, keepdim=True)
                    log_s = torch.clamp(log_s, max=-1e-7)  # Clamp to avoid log_s >= 0 (which implies sum(probs) >= 1)
                    tail_log = torch.log(-torch.expm1(log_s))  # We use the identity: 1 - exp(x) = -(exp(x) - 1); torch.expm1(x) computes (e^x - 1) with high precision for small x.
                    return torch.cat([log_probs, tail_log], dim=-1)

                def renorm_topk_log_probs(logp: torch.Tensor) -> torch.Tensor:
                    logZ = torch.logsumexp(logp, dim=-1, keepdim=True)
                    return logp - logZ
                
                student_distill_log_probs = student_topk_log_probs
                teacher_distill_log_probs = teacher_topk_log_probs
                if self_distillation_config.distillation_add_tail:
                    student_distill_log_probs = add_tail(student_distill_log_probs)
                    teacher_distill_log_probs = add_tail(teacher_distill_log_probs)
                else:
                    student_distill_log_probs = renorm_topk_log_probs(student_distill_log_probs)
                    teacher_distill_log_probs = renorm_topk_log_probs(teacher_distill_log_probs)    # [bz, seq_len, topk]
            else:
                if student_all_log_probs is None or teacher_all_log_probs is None:
                    raise ValueError("full_logit_distillation requires student_all_log_probs and teacher_all_log_probs.")
                student_distill_log_probs = student_all_log_probs
                teacher_distill_log_probs = teacher_all_log_probs   # [bz, seq_len, vocab_size]

        else:
            teacher_distill_log_probs = teacher_log_probs.unsqueeze(-1) # expand the vocab dimension
            student_distill_log_probs = student_log_probs.unsqueeze(-1)

        # masking
        masked_teacher_distill_log_probs = teacher_distill_log_probs * teacher_response_mask.unsqueeze(-1)
        masked_student_distill_log_probs = student_distill_log_probs * student_response_mask.unsqueeze(-1)

        # kl_loss = F.kl_div(teacher_distill_log_probs, student_distill_log_probs, reduction="none", log_target=True)
        mask_teacher = (teacher_response_mask != 0)
        mask_student = (student_response_mask != 0)
        compact_teacher_logp = masked_teacher_distill_log_probs[mask_teacher]
        compact_student_logp = masked_student_distill_log_probs[mask_student]
        # compact_kl_loss = (torch.exp(compact_teacher_logp) * (compact_teacher_logp - compact_student_logp)).sum(-1)
        if self_distillation_config.full_logit_distillation:
            # use F.kl_div
            if self_distillation_config.alpha == 0.:
                compact_kl_loss = F.kl_div(compact_student_logp, compact_teacher_logp, reduction="none", log_target=True)
            elif self_distillation_config.alpha == 1.:
                compact_kl_loss = F.kl_div(compact_teacher_logp, compact_student_logp, reduction="none", log_target=True)
            else:
                raise ValueError(f"Invalid alpha: {self_distillation_config.alpha}")
            
            compact_kl_loss = compact_kl_loss.sum(-1)
        else:
            # reverse KL
            ratio = compact_student_logp - compact_teacher_logp
            compact_kl_loss = (compact_student_logp * ratio.detach()).sum(-1)
            metrics["actor/distill_ratio_mean"] = ratio.detach().mean().item()
            metrics["actor/distill_ratio_max"] = ratio.detach().max().item()
            metrics["actor/distill_ratio_min"] = ratio.detach().min().item()
        
        compact_mask = torch.ones_like(compact_kl_loss, dtype=teacher_response_mask.dtype, device=compact_kl_loss.device)
        # offpolicyness correction? not doing for supervised-sd, as the student traj need re-compute old_log_prob in ray_trainer

        compact_kl_loss = agg_loss(loss_mat=compact_kl_loss, loss_mask=compact_mask, loss_agg_mode=self.config.loss_agg_mode)

        return compact_kl_loss, metrics



    def _compute_op_sd_distillation_loss(self, inputs, temperature, input_log_prob, old_log_prob):
        self_distillation_cfg = getattr(self.config, "self_distillation", None)
        return_all_logps = self_distillation_cfg.full_logit_distillation and not self_distillation_cfg.distillation_topk
        distill_topk = self_distillation_cfg.distillation_topk if self_distillation_cfg.full_logit_distillation else None

        # for op-sd mode, 

        # if input_log_prob is None:
        if return_all_logps or distill_topk is not None:   # need to recomute student logits
            # forward pass on original student traj
            student_inputs = {
                "responses": inputs['responses'],
                "input_ids": inputs['input_ids'],
                "attention_mask": inputs['attention_mask'],
                "position_ids": inputs['position_ids'],
            }
            student_outputs = self._forward_micro_batch_v2(
                student_inputs,
                temperature=temperature,
                calculate_entropy=False,
                return_all_logps=return_all_logps,
                distill_topk=distill_topk,
                debug=False
            )
            student_log_probs = student_outputs['log_probs']
            student_all_logps = student_outputs.get("all_logps") if return_all_logps else None
            student_topk_logps = student_outputs.get("topk_logps") if distill_topk is not None else None
        else:
            student_log_probs = input_log_prob
            student_all_logps = None   # here we might still need to do full_logits computation is return_all_logits is True
            student_topk_logps = None
        
        # compute teacher logp
        teacher_inputs = {
            "responses": inputs['distill_teacher_response'],
            "input_ids": inputs['distill_teacher_input_ids'],
            "attention_mask": inputs['distill_teacher_attention_mask'],
            "position_ids": inputs['distill_teacher_position_ids'],
        }
        with torch.no_grad():
            teacher_outputs = self._forward_micro_batch_v2(
                teacher_inputs,
                temperature=temperature,
                calculate_entropy=False,
                return_all_logps=return_all_logps,
                distill_topk=distill_topk,
                debug=False
            )
        teacher_log_probs = teacher_outputs['log_probs']
        teacher_all_logps = teacher_outputs.get("all_logps") if return_all_logps else None
        teacher_topk_logps = teacher_outputs.get("topk_logps") if distill_topk is not None else None

        if sum(inputs['distill_teacher_response_mask'][0]) != sum(inputs['distill_student_response_mask'][0]):
            raise ValueError("The number of valid teacher and student response tokens are not the same")

        # compute distillation loss
        distill_kl_loss, distillation_metrics = self._op_sd_distillation_loss(
            student_log_probs=student_log_probs,
            teacher_log_probs=teacher_log_probs,
            self_distillation_config=self_distillation_cfg,
            student_all_log_probs=student_all_logps,
            teacher_all_log_probs=teacher_all_logps,
            student_topk_log_probs=student_topk_logps,
            teacher_topk_log_probs=teacher_topk_logps,
            teacher_response_mask=inputs['distill_teacher_response_mask'],
            student_response_mask=inputs['distill_student_response_mask'],
            old_log_prob=old_log_prob,
        )
        return distill_kl_loss, distillation_metrics

    def _op_sd_distillation_loss(
        self, 
        student_log_probs: torch.Tensor,
        teacher_log_probs: torch.Tensor,
        self_distillation_config: Any,
        student_all_log_probs: Optional[torch.Tensor] = None,
        teacher_all_log_probs: Optional[torch.Tensor] = None,
        student_topk_log_probs: Optional[torch.Tensor] = None,
        teacher_topk_log_probs: Optional[torch.Tensor] = None,
        teacher_response_mask: Optional[torch.Tensor] = None,
        student_response_mask: Optional[torch.Tensor] = None,
        old_log_prob: Optional[torch.Tensor] = None,
    ):
        metrics = {}

        if self_distillation_config.full_logit_distillation:
            use_topk = self_distillation_config.distillation_topk is not None
            if use_topk:
                if student_topk_log_probs is None or teacher_topk_log_probs is None:
                    raise ValueError("top-k distillation requires student_topk_log_probs and teacher_topk_log_probs")

                def add_tail(log_probs: torch.Tensor) -> torch.Tensor:
                    # Compute tail log-probability using logsumexp for numerical stability
                    # log(1 - sum(p_i)) = log(1 - exp(log_sum_exp(log(p_i))))
                    log_s = torch.logsumexp(log_probs, dim=-1, keepdim=True)
                    log_s = torch.clamp(log_s, max=-1e-7)  # Clamp to avoid log_s >= 0 (which implies sum(probs) >= 1)
                    tail_log = torch.log(-torch.expm1(log_s))  # We use the identity: 1 - exp(x) = -(exp(x) - 1); torch.expm1(x) computes (e^x - 1) with high precision for small x.
                    return torch.cat([log_probs, tail_log], dim=-1)

                def renorm_topk_log_probs(logp: torch.Tensor) -> torch.Tensor:
                    logZ = torch.logsumexp(logp, dim=-1, keepdim=True)
                    return logp - logZ
                
                student_distill_log_probs = student_topk_log_probs
                teacher_distill_log_probs = teacher_topk_log_probs
                if self_distillation_config.distillation_add_tail:
                    student_distill_log_probs = add_tail(student_distill_log_probs)
                    teacher_distill_log_probs = add_tail(teacher_distill_log_probs)
                else:
                    student_distill_log_probs = renorm_topk_log_probs(student_distill_log_probs)
                    teacher_distill_log_probs = renorm_topk_log_probs(teacher_distill_log_probs)
            else:
                if student_all_log_probs is None or teacher_all_log_probs is None:
                    raise ValueError("full_logit_distillation requires student_all_log_probs and teacher_all_log_probs.")
                student_distill_log_probs = student_all_log_probs
                teacher_distill_log_probs = teacher_all_log_probs
        else:
            student_distill_log_probs = student_log_probs.unsqueeze(-1)
            teacher_distill_log_probs = teacher_log_probs.unsqueeze(-1)

        # masking
        masked_teacher_distill_log_probs = teacher_distill_log_probs * teacher_response_mask.unsqueeze(-1)
        masked_student_distill_log_probs = student_distill_log_probs * student_response_mask.unsqueeze(-1)

        # unpad
        mask_teacher = (teacher_response_mask != 0)
        mask_student = (student_response_mask != 0)
        compact_teacher_logp = masked_teacher_distill_log_probs[mask_teacher]
        compact_student_logp = masked_student_distill_log_probs[mask_student]

        # compute KL is full logits
        if self_distillation_config.full_logit_distillation:
            # use F.kl_div
            if self_distillation_config.alpha == 0.:
                compact_kl_loss = F.kl_div(compact_student_logp, compact_teacher_logp, reduction="none", log_target=True)
            elif self_distillation_config.alpha == 1.:
                compact_kl_loss = F.kl_div(compact_teacher_logp, compact_student_logp, reduction="none", log_target=True)
            else:
                raise ValueError(f"Invalid alpha: {self_distillation_config.alpha}")

            compact_kl_loss = compact_kl_loss.sum(-1)
        else:
            # only reverse KL is supported
            ratio = compact_student_logp - compact_teacher_logp
            compact_kl_loss = (compact_student_logp * ratio.detach())
            metrics["actor/distill_ratio_mean"] = ratio.detach().mean().item()
            metrics["actor/distill_ratio_max"] = ratio.detach().max().item()
            metrics["actor/distill_ratio_min"] = ratio.detach().min().item()

        # weighted by old_log_probs?
        is_clip = self_distillation_config.is_clip
        if is_clip is not None:
            assert old_log_prob is not None
            assert old_log_prob.shape == student_log_probs.shape

            compact_old_log_prob = old_log_prob[mask_student]
            compact_student_log_prob = student_log_probs[mask_student]
            negative_approx_kl = (compact_student_log_prob - compact_old_log_prob).detach()
            negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
            is_ratio = torch.exp(negative_approx_kl).clamp(max=is_clip)
            compact_kl_loss = compact_kl_loss * is_ratio
            metrics["actor/is_ratio_mean"] = is_ratio.detach().mean().item()
        
        compact_mask = torch.ones_like(compact_kl_loss, dtype=teacher_response_mask.dtype, device=compact_kl_loss.device)
        compact_kl_loss = agg_loss(loss_mat=compact_kl_loss, loss_mask=compact_mask, loss_agg_mode=self.config.loss_agg_mode)

        return compact_kl_loss, metrics


    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_sft(self, data: DataProto):
        """Update the actor model using SFT (Supervised Fine-Tuning) loss.
        
        Args:
            data (DataProto): A DataProto containing:
                - input_ids: tensor of shape [batch_size, sequence_length]
                - attention_mask: tensor of shape [batch_size, sequence_length]
                - position_ids: tensor of shape [batch_size, sequence_length]
                - loss_mask: tensor of shape [batch_size, sequence_length] indicating which tokens to compute loss on
                - multi_modal_inputs (optional): dict containing multi-modal inputs
        
        Returns:
            dict: A dictionary containing training metrics such as loss, grad_norm, etc.
        """
        # make sure we are in training mode
        self.actor_module.train()
        sft_config = self.config.sft_loss

        select_keys = [
            "sft_input_ids",
            "sft_attention_mask",
            "sft_position_ids",
            "sft_loss_mask",
        ]

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # verl's SFT only has micro batch size
        sft_micro_batch_size = sft_config.get("sft_micro_batch_size_per_gpu", self.config.get("ppo_micro_batch_size_per_gpu", 1))
        # balance_dp_token = sft_config.get("balance_dp_token", False)

        use_remove_padding = sft_config.get("use_remove_padding", False)
        ulysses_sequence_parallel_size = sft_config.get("ulysses_sequence_parallel_size", self.ulysses_sequence_parallel_size)
        use_ulysses_sp = ulysses_sequence_parallel_size > 1

        # Split to make minibatch iterator for updating the actor
        micro_batches = data.split(sft_micro_batch_size)
        n_micro_batches = len(micro_batches)

        metrics = {}
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        self.sft_optimizer.zero_grad()
        
        for batch_idx, micro_batch in enumerate(micro_batches):
            micro_batch_metrics = {}
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            # actor calling
            with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
                input_ids = model_inputs["sft_input_ids"]
                attention_mask = model_inputs["sft_attention_mask"]
                position_ids = model_inputs["sft_position_ids"]
                loss_mask = model_inputs["sft_loss_mask"][:, :-1].reshape(-1)  # Remove last token and flatten
                batch_size, seqlen = input_ids.shape

                if use_remove_padding:
                    input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                        input_ids.unsqueeze(-1), attention_mask
                    )  # input_ids_rmpad (total_nnz, ...)
                    input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                    if position_ids.dim() == 3:
                        position_ids_rmpad = (
                            index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                            .transpose(0, 1)
                            .unsqueeze(1)
                        )
                    else:
                        position_ids_rmpad = index_first_axis(
                            rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                        ).transpose(0, 1)

                    input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                    if use_ulysses_sp:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=ulysses_sequence_parallel_size,
                        )

                        input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad_rolled,
                            position_ids_rmpad=None,
                            sp_size=ulysses_sequence_parallel_size,
                        )
                    
                    input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                    output = self.actor_module(
                        input_ids=input_ids_rmpad,
                        attention_mask=None,
                        position_ids=position_ids_rmpad,
                        use_cache=False,
                    )

                    # Compute loss locally then aggregate
                    logits_rmpad = output.logits.squeeze(0)
                    # input_ids_rmpad_rolled = input_ids_rmpad_rolled.to(logits_rmpad.device)
                    loss = loss_fct(logits_rmpad, input_ids_rmpad_rolled)

                    if use_ulysses_sp:
                        loss = gather_outputs_and_unpad(loss, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                        full_loss = pad_input(hidden_states=loss.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)
                        full_loss = full_loss.squeeze(-1)[:, :-1]  # Remove last token's loss
                        full_loss = full_loss.reshape(-1)
                        loss_mask = loss_mask.to(full_loss.device)
                        loss = full_loss * loss_mask
                else:
                    raise ValueError("use_remove_padding must be True")
                
                # Compute average loss over valid tokens
                valid_token_this_rank = torch.sum(loss_mask)
                dp_size = 1 # no dp balance

                sft_loss = torch.sum(loss) / (valid_token_this_rank + 1e-8) * dp_size

                loss_scaled = sft_loss / n_micro_batches
                
                loss_scaled.backward()

                micro_batch_metrics.update(
                        {
                            "actor/sft_loss": sft_loss.detach().item(),
                            "actor/sft_valid_tokens": valid_token_this_rank.detach().item(),
                        }
                )
                append_to_dict(metrics, micro_batch_metrics)
                
            grad_norm = self._sft_optimizer_step()
            mini_batch_metrics = {"actor/sft_grad_norm": grad_norm.detach().item()}
            append_to_dict(metrics, mini_batch_metrics)

        self.sft_optimizer.zero_grad()
        return metrics
    
    def _forward_micro_batch_v2(
        self, 
        micro_batch, 
        temperature, 
        calculate_entropy=False,
        return_all_logps: bool = False,
        distill_topk: Optional[int] = None,
        topk_indices: Optional[torch.Tensor] = None,
        disable_gradient: bool = False,
        debug: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        or a dict if return_all_logps is True
        """
        calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False)
        sum_pi_squared_checkpointing = self.config.get("sum_pi_squared_checkpointing", False)
        use_topk = distill_topk is not None or topk_indices is not None
        compute_all_logps = return_all_logps and not use_topk
        return_topk_indices = use_topk and topk_indices is not None
        if (return_all_logps or use_topk) and self.use_fused_kernels:
            raise ValueError("Logit distillation requires disabling fused kernels.")


        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            if "image_bound" in micro_batch["multi_modal_inputs"][0]:  # minicpm-o logic
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = [inputs[key] for inputs in micro_batch["multi_modal_inputs"]]
            else:
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = torch.cat(
                        [inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0
                    )

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask);  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)
                if debug:
                    import pdb; pdb.set_trace()
                # print(f"input_nonpad = {input_ids.shape[1] - sum(input_ids[0]==151643)}, attention sum = {attention_mask.sum()}, indices shape = {indices.shape}, batch_size = {batch_size}, seqlen = {seqlen}, input_ids_rmpad shape = {input_ids_rmpad.shape}, indices shape = {indices.shape}, cu_seqlens = {cu_seqlens}")

                if input_ids_rmpad.shape[1] != indices.shape[0]:
                    import pdb; pdb.set_trace()
                # print(f"input_ids shape = {input_ids.shape}, attention_mask shape = {attention_mask.shape}")
                
                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch.keys()
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)
                    all_lops_rmpad = torch.log_softmax(logits_rmpad, dim=-1) if compute_all_logps else None

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )
                    
                    if use_topk:
                        if topk_indices is None:
                            topk = min(distill_topk, logits_rmpad.shape[-1])
                            topk_logits_rmpad, topk_indices_rmpad = torch.topk(logits_rmpad, topk, dim=-1)
                        else:
                            topk = topk_indices.size(-1)
                            full_topk_indices = torch.zeros(
                                batch_size,
                                seqlen,
                                topk,
                                device=topk_indices.device,
                                dtype=topk_indices.dtype,
                            )
                            full_topk_indices[:, -response_length - 1 : -1, :] = topk_indices
                            topk_indices_rmpad = index_first_axis(
                                rearrange(full_topk_indices, "b s k -> (b s) k"), indices
                            )
                            if self.use_ulysses_sp:
                                topk_indices_rmpad = slice_input_tensor(
                                    topk_indices_rmpad.unsqueeze(0), dim=1, padding=True
                                ).squeeze(0)
                            # topk_logits_rmpad = torch.gather(logits_rmpad, dim=-1, index=topk_indices_rmpad)
                        # logsumexp_rmpad = torch.logsumexp(logits_rmpad, dim=-1, keepdim=True)
                        # topk_logps_rmpad = topk_logits_rmpad - logsumexp_rmpad
                        # topk_logps_rmpad = torch.gather(
                            # all_lops_rmpad, dim=-1, index=topk_indices_rmpad
                        # )

                        # Compute top-k log-probs directly, without full log_softmax if compute_all_logps is False
                        if compute_all_logps:
                            topk_logps_rmpad = torch.gather(
                                all_lops_rmpad, dim=-1, index=topk_indices_rmpad
                            )
                        else:
                            # Memory-efficient: compute logsumexp per row and subtract
                            if disable_gradient:
                                logits_rmpad = logits_rmpad.detach().cpu()
                                logsumexp_rmpad = torch.logsumexp(logits_rmpad, dim=-1, keepdim=True).to(topk_indices_rmpad.device)
                                topk_logits_rmpad = torch.gather(logits_rmpad, dim=-1, index=topk_indices_rmpad.detach().cpu()).to(topk_indices_rmpad.device)
                            else:
                                logsumexp_rmpad = torch.logsumexp(logits_rmpad, dim=-1, keepdim=True)
                                topk_logits_rmpad = torch.gather(logits_rmpad, dim=-1, index=topk_indices_rmpad)
                            
                            topk_logps_rmpad = topk_logits_rmpad - logsumexp_rmpad

                    # Compute sum_pi_squared if requested (for optimal_token_baseline)
                    if calculate_sum_pi_squared:
                        sum_pi_squared_rmpad = (
                            self.calculate_sum_pi_squared_from_logits(logits_rmpad)
                            if not sum_pi_squared_checkpointing
                            else torch.utils.checkpoint.checkpoint(
                                self.calculate_sum_pi_squared_from_logits, logits_rmpad
                            )
                        )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                    if use_topk:
                        topk_logps_rmpad = gather_outputs_and_unpad(
                            topk_logps_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                        if return_topk_indices:
                            topk_indices_rmpad = gather_outputs_and_unpad(
                                topk_indices_rmpad,
                                gather_dim=0,
                                unpad_dim=0,
                                padding_size=pad_size,
                            )
                    if calculate_sum_pi_squared:
                        sum_pi_squared_rmpad = gather_outputs_and_unpad(
                            sum_pi_squared_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                    if compute_all_logps:
                        all_lops_rmpad = gather_outputs_and_unpad(
                            all_lops_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                
                # if is_mask_all_zero: not implemented yet

                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                if calculate_sum_pi_squared:
                    full_sum_pi_squared = pad_input(
                        hidden_states=sum_pi_squared_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                if compute_all_logps:
                    full_all_logps = pad_input(
                        hidden_states=all_lops_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )

                if use_topk:
                    full_topk_logps = pad_input(
                        hidden_states=topk_logps_rmpad,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    if return_topk_indices:
                        full_topk_indices = pad_input(
                            hidden_states=topk_indices_rmpad,
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen,
                        )
                
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                if calculate_sum_pi_squared:
                    # (bsz, response_length)
                    sum_pi_squared = full_sum_pi_squared.squeeze(-1)[:, -response_length - 1 : -1]
                
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

                if compute_all_logps:
                    all_logps = full_all_logps[:, -response_length - 1 : -1, :]
                if use_topk:
                    topk_logps = full_topk_logps[:, -response_length - 1 : -1, :]
                    if return_topk_indices:
                        topk_indices = full_topk_indices[:, -response_length - 1 : -1, :]

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if compute_all_logps:
                        all_logps = torch.log_softmax(logits, dim=-1)
                    if use_topk:
                        if topk_indices is None:
                            topk = min(distill_topk, logits.size(-1))
                            topk_logits, topk_indices = torch.topk(logits, topk, dim=-1)
                        else:
                            topk_logits = torch.gather(logits, dim=-1, index=topk_indices)
                        logsumexp = torch.logsumexp(logits, dim=-1, keepdim=True)
                        topk_logps = topk_logits - logsumexp
                    
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

                    # Compute sum_pi_squared if requested (for optimal_token_baseline)
                    if calculate_sum_pi_squared:
                        sum_pi_squared = (
                            self.calculate_sum_pi_squared_from_logits(logits)
                            if not sum_pi_squared_checkpointing
                            else torch.utils.checkpoint.checkpoint(self.calculate_sum_pi_squared_from_logits, logits)
                        )
            outputs = {"log_probs": log_probs}
            if calculate_entropy:
                outputs['entropys'] = entropy
            if calculate_sum_pi_squared:
                outputs['sum_pi_squared'] = sum_pi_squared
            if compute_all_logps:
                outputs['all_logps'] = all_logps
            if use_topk:
                outputs['topk_logps'] = topk_logps
                if return_topk_indices:
                    outputs['topk_indices'] = topk_indices
            return outputs