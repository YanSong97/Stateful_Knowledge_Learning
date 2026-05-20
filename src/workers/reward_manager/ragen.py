from collections import defaultdict

import torch

from verl import DataProto


class RAGENRewardManager:
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

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        all_scores = []
        already_print_data_sources = {}
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            score = data_item.non_tensor_batch['reward']
            score = float(score)

            reward_tensor[i, valid_response_length - 1] = score
            all_scores.append(score)

            # Get data_source from data_item if available, otherwise use a default value
            data_source = data_item.non_tensor_batch.get('data_source', 'default')

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)

        print(f"[DEBUG] all_scores: {all_scores}")
        print(f"[DEBUG] all_scores shape: {np.array(all_scores).shape}")
        print(f"[DEBUG] all_scores mean: {np.mean(all_scores)}")
        print(f"[DEBUG] all_scores max: {np.max(all_scores)}")
        print(f"[DEBUG] all_scores min: {np.min(all_scores)}")
        print(f"[DEBUG] all_scores std: {np.std(all_scores)}")


        reward_extra_info = {}
        reward_extra_info["reward"] = reward_tensor
        reward_extra_info["action_valid_rate"] = data.non_tensor_batch["action_valid_rate"]

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        else:
            return reward_tensor
