# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from collections import defaultdict
from tqdm import tqdm
import torch
import numpy as np
import ray
from typing import List, Dict, Any, Optional

from verl import DataProto
from verl.workers.reward_manager import register


@ray.remote
class RayRewardWorker:
    """Ray worker for distributed reward computation."""
    
    def __init__(self, tokenizer, compute_score, reward_fn_key="data_source", custom_reward_config=None):
        self.tokenizer = tokenizer
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        self.custom_reward_config = custom_reward_config
        
        # Load custom reward function if config is provided or if compute_score is None
        if custom_reward_config or compute_score is None:
            self._load_custom_reward_function()
    
    def _load_custom_reward_function(self):
        """Load custom reward function in the Ray worker process."""
        import importlib.util
        import sys
        
        # If we have custom_reward_config, load the custom function
        if self.custom_reward_config:
            file_path = self.custom_reward_config.get("path")
            function_name = self.custom_reward_config.get("name")
            
            if file_path and function_name:
                try:
                    spec = importlib.util.spec_from_file_location("custom_module", file_path)
                    module = importlib.util.module_from_spec(spec)
                    sys.modules["custom_module"] = module
                    spec.loader.exec_module(module)
                    
                    if hasattr(module, function_name):
                        self.compute_score = getattr(module, function_name)
                        return
                except Exception as e:
                    print(f"Warning: Failed to load custom reward function from {file_path}: {e}")
        
        # If no custom config or loading failed, and compute_score is None, load default
        if self.compute_score is None:
            try:
                from verl.utils.reward_score import default_compute_score
                self.compute_score = default_compute_score
            except ImportError:
                print("Warning: Failed to load default compute_score function")
    
    def process_batch(self, batch_data: List[Dict[str, Any]]):
        """Process a batch of data items and return results."""
        results = []
        judge_responses = []
        
        for item_data in batch_data:
            # Extract data from the serialized item
            prompt_ids = item_data["prompt_ids"]
            response_ids = item_data["response_ids"]
            attention_mask = item_data["attention_mask"]
            golden_answers = item_data["golden_answers"]
            question = item_data["question"]
            data_source = item_data["data_source"]
            extra_info = item_data.get("extra_info", {})
            num_turns = item_data.get("num_turns", None)
            extra_info["num_turns"] = num_turns
            
            # Process the data similar to the original loop
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = attention_mask[:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            
            valid_response_length = attention_mask[prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            
            # Decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            
            ground_truth = golden_answers
            
            # Compute score
            score, evaluation_response, extract_answer = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                question=question,
                model_config=None,
                return_dict=True,
                extra_info=extra_info,
            )
            
            if isinstance(score, dict):
                reward = score["score"]
                result = {
                    "reward": reward,
                    "score_dict": score,
                    "prompt_str": prompt_str,
                    "response_str": response_str,
                    "ground_truth": ground_truth,
                    "data_source": data_source,
                    "valid_response_length": valid_response_length
                }
            else:
                reward = score
                result = {
                    "reward": reward,
                    "score_dict": {"score": reward},
                    "prompt_str": prompt_str,
                    "response_str": response_str,
                    "ground_truth": ground_truth,
                    "data_source": data_source,
                    "valid_response_length": valid_response_length
                }
            
            results.append(result)
            judge_responses.append(
                {
                    "response": evaluation_response,
                    "ground_truth": ground_truth,
                    "extracted_answer": extract_answer

                }
            )
        
        return results, judge_responses


@register("generative_distributed")
class GenerativeDistributedRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", 
                 use_ray_distributed=True, num_ray_workers=4, custom_reward_config=None) -> None:
        """
        Initialize the GenerativeRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
            use_ray_distributed: Whether to use Ray for distributed processing. Defaults to False.
            num_ray_workers: Number of Ray workers to use for distributed processing. Defaults to 4.
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        assert compute_score is not None
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source
        assert use_ray_distributed
        self.use_ray_distributed = use_ray_distributed
        self.num_ray_workers = num_ray_workers
        self.custom_reward_config = custom_reward_config


    def _prepare_batch_data(self, data: DataProto) -> List[List[Dict[str, Any]]]:
        """Prepare data for Ray distributed processing by creating batches."""
        all_items = []
        
        for i in range(len(data)):
            data_item = data[i]
            
            # Extract and serialize the data
            item_data = {
                "prompt_ids": data_item.batch["prompts"].cpu().numpy(),
                "response_ids": data_item.batch["responses"].cpu().numpy(),
                "attention_mask": data_item.batch["attention_mask"].cpu().numpy(),
                "golden_answers": data_item.non_tensor_batch['golden_answers'],
                "question": data_item.non_tensor_batch['question'],
                "data_source": data_item.non_tensor_batch[self.reward_fn_key],
                "extra_info": data_item.non_tensor_batch.get("extra_info", {}),
                "num_turns": data_item.non_tensor_batch.get("__num_turns__", None),
            }
            all_items.append(item_data)
        
        # Split into batches for Ray workers
        batch_size = max(1, len(all_items) // self.num_ray_workers)
        batches = []
        for i in range(0, len(all_items), batch_size):
            batches.append(all_items[i:i + batch_size])
        
        return batches

    def _distributed_evaluate(self, data: DataProto) -> tuple:
        """Evaluate using Ray distributed processing."""
        if not ray.is_initialized():
            ray.init()
        
        # Prepare batches
        batches = self._prepare_batch_data(data)
        
        # Create Ray workers
        # Pass None as compute_score if we have custom_reward_config, let workers load it themselves
        compute_score_to_pass = None if self.custom_reward_config else self.compute_score
        workers = [RayRewardWorker.remote(self.tokenizer, compute_score_to_pass, self.reward_fn_key, self.custom_reward_config) 
                  for _ in range(self.num_ray_workers)]
        
        # Distribute work
        futures = []
        for i, batch in enumerate(batches):
            worker_idx = i % len(workers)
            future = workers[worker_idx].process_batch.remote(batch)
            futures.append(future)
        
        # Collect results
        all_results = []
        all_judge_responses = []
        for future in tqdm(ray.get(futures), desc="Collecting Ray results"):
            all_results.extend(future[0])
            all_judge_responses.extend(future[1])
        
        # Convert results back to tensor format
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print_data_sources = {}
        
        for i, result in enumerate(all_results):
            reward = result["reward"]
            valid_response_length = result["valid_response_length"]
            reward_tensor[i, valid_response_length - 1] = reward
            
            # Store extra info
            score_dict = result["score_dict"]
            for key, value in score_dict.items():
                reward_extra_info[key].append(value)
            
            # Handle printing for examination
            data_source = result["data_source"]
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", result["prompt_str"])
                print("[response]", result["response_str"])
                print("[ground_truth]", result["ground_truth"])
                for key, value in score_dict.items():
                    print(f"[{key}]", value)

        judge_info = defaultdict(list)
        for i, result in enumerate(all_judge_responses):
            judge_response = result['response']
            gt = result['ground_truth']
            extracted_answer = result['extracted_answer']

            judge_info["judge_response"].append(judge_response)
            judge_info['gt'].append(gt)
            judge_info['extracted_answer'].append(extracted_answer)
        
        return reward_tensor, reward_extra_info, judge_info

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        # Use Ray distributed processing if enabled

        reward_tensor, reward_extra_info, judge_info = self._distributed_evaluate(data)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
                "judge_info": judge_info
            }
        else:
            return reward_tensor



    def offline_evaluate_single(self, data: DataProto, return_dict=False):
        """
        Offline evaluation for the reward manager. No need to decode the response.
        """
        reward_extra_info = defaultdict(list)

        extra_info = data.non_tensor_batch['extra_info'][0]

        response_str = extra_info['output']
        question = extra_info['question']
        ground_truth = extra_info["ground_truth"]
        data_source = data.non_tensor_batch['data_source'][0]

        score = self.compute_score(
            data_source=data_source,
            solution_str=response_str,
            ground_truth=ground_truth,
            question=question,
            model_config=None,
            return_dict=True,
            extra_info=extra_info,
        )

        if isinstance(score, dict):
            reward = score["score"]
            # Store the information including original reward
            for key, value in score.items():
                reward_extra_info[key].append(value)
        else:
            reward = score

        if return_dict:
            return {
                "reward_tensor": np.array(reward),
                "reward_extra_info": reward_extra_info,
            }
        else:
            return np.array(reward)

    def offline_evaluate_distributed(self, data_list: List[DataProto], return_dict=False):
        """
        Distributed offline evaluation for multiple DataProto instances.
        """
        if not ray.is_initialized():
            ray.init()
        
        # Create Ray worker for offline evaluation
        @ray.remote
        class OfflineEvaluationWorker:
            def __init__(self, compute_score):
                self.compute_score = compute_score
            
            def evaluate_single(self, data_item: Dict[str, Any]) -> Dict[str, Any]:
                """Evaluate a single data item."""
                extra_info = data_item['extra_info']
                response_str = extra_info['output']
                question = extra_info['question']
                ground_truth = extra_info["ground_truth"]
                data_source = data_item['data_source']
                
                score = self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    question=question,
                    model_config=None,
                    return_dict=True,
                    extra_info=extra_info,
                )
                
                if isinstance(score, dict):
                    reward = score["score"]
                    result = {
                        "reward": reward,
                        "score_dict": score,
                    }
                else:
                    reward = score
                    result = {
                        "reward": reward,
                        "score_dict": {"score": reward},
                    }
                
                return result
        
        # Prepare data for Ray processing
        data_items = []
        for data in data_list:
            item_data = {
                "extra_info": data.non_tensor_batch['extra_info'][0],
                "data_source": data.non_tensor_batch['data_source'][0],
            }
            data_items.append(item_data)
        
        # Create workers
        workers = [OfflineEvaluationWorker.remote(self.compute_score) 
                  for _ in range(self.num_ray_workers)]
        
        # Distribute work
        futures = []
        for i, item in enumerate(data_items):
            worker_idx = i % len(workers)
            future = workers[worker_idx].evaluate_single.remote(item)
            futures.append(future)
        
        # Collect results
        all_results = []
        for future in tqdm(ray.get(futures), desc="Collecting offline evaluation results"):
            all_results.append(future)
        
        # Aggregate results
        reward_extra_info = defaultdict(list)
        total_reward = 0.0
        
        for result in all_results:
            reward = result["reward"]
            total_reward += reward
            
            # Store extra info
            score_dict = result["score_dict"]
            for key, value in score_dict.items():
                reward_extra_info[key].append(value)
        
        avg_reward = total_reward / len(all_results)
        
        if return_dict:
            return {
                "reward_tensor": np.array(avg_reward),
                "reward_extra_info": reward_extra_info,
            }
        else:
            return np.array(avg_reward)