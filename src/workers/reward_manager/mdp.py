from collections import defaultdict

import torch
import numpy as np

from verl import DataProto


class MDPRewardManager:
    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine

        self.reward_fn_key = reward_fn_key

    def __call__(self, data: DataProto, return_dict=False):
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        # process reward extra info
        reward_extra_info = {}

        env_success = data.meta_info['others']['success_rate']
        # print(f"response shape = {data.batch['responses'].shape}")
        # print(f"env_success shape = {env_success}")
        # assert len(env_success) == data.batch['responses'].shape[0], print(f"mismatch dimension between env_success and response {len(env_success)} != {data.batch['responses'].shape}")
        if len(env_success) > data.batch['responses'].shape[0]:     # unpad_dataproto not applied to metrics data
            env_success = env_success[:-data.batch['responses'].shape[0]]
        elif len(env_success) == data.batch['responses'].shape[0]:
            pass
        else:
            assert len(env_success) == data.batch['responses'].shape[0], print(
                f"mismatch dimension between env_success and response {len(env_success)} != {data.batch['responses'].shape}")


        reward_extra_info["success_rate"] = [int(i) for i in env_success]


        # process reward
        if "reward_tensor" in data.batch.keys():
            reward_tensor = data.batch["reward_tensor"]
        else:
            reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

            for i in range(len(data)):
                data_item = data[i]
                prompt_ids = data_item.batch["prompts"]
                prompt_length = prompt_ids.shape[-1]

                # valid_p rompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
                # valid_prompt_ids = prompt_ids[-valid_prompt_length:]

                response_ids = data_item.batch["responses"]
                valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                # valid_response_ids = response_ids[:valid_response_length]

                reward_tensor[i, valid_response_length - 1] = float(env_success[i])


        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        else:
            return reward_tensor



