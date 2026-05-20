#!/usr/bin/env python3
"""
Script to evaluate JSON files containing LLM outputs and ground truth using GenerativeRewardManager.

This script reads a JSON file with LLM outputs and ground truth, then uses the GenerativeRewardManager
to evaluate each response using a remote xverify model and computes average results.

python src/workers/reward_manager/evaluate_json.py \
    --json_file logs/eval/nq_search/async_rag_loop/output_tool_mcp_rag_threshold2000_turn5_xverify__20250903_032515/1.jsonl \
    --eval_mode "xverify"


python src/workers/reward_manager/evaluate_json.py \
    --json_file logs/eval/gpt_oss/logs.jsonl \
    --eval_mode "xverify"

python src/workers/reward_manager/evaluate_json.py \
    --json_file logs/eval/nq_search/async_rag_loop/output_tool_mcp_rag_turn3_em__20250901_074610/1.jsonl \
    --eval_mode "llm_judge_distributed"

"""
import time
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import torch
from tqdm import tqdm
import numpy as np
from tensordict import TensorDict

# Import necessary modules
from verl import DataProto
from src.workers.reward_manager.generative import GenerativeRewardManager
from src.workers.reward_manager import GenerativeDistributedRewardManager
from src.utils.reward_score.xverify_reward_fn import xverify_compute_score
from src.utils.reward_score.llm_judge_reward_fn import llm_judge_compute_score


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    """Load JSON data from file - supports both JSON and JSONL formats"""
    data = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        # Try to load as regular JSON first
        try:
            f.seek(0)  # Reset file pointer
            json_data = json.load(f)
            
            # Handle both list and single object formats
            if isinstance(json_data, dict):
                data = [json_data]
            elif isinstance(json_data, list):
                data = json_data
            else:
                raise ValueError(f"Invalid JSON format: expected dict or list, got {type(json_data)}")
                
        except json.JSONDecodeError as e:
            # If regular JSON fails, try JSONL format
            f.seek(0)  # Reset file pointer
            data = []
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        item = json.loads(line)
                        data.append(item)
                    except json.JSONDecodeError as line_error:
                        raise ValueError(f"Invalid JSON at line {line_num}: {line_error}")
    
    return data


def create_dummy_tokenizer():
    """Create a dummy tokenizer for the reward manager"""
    class DummyTokenizer:
        def decode(self, token_ids, skip_special_tokens=True):
            # This is a dummy implementation - the actual tokenization is not needed
            # since we're working with pre-decoded text
            return "dummy"
    
    return DummyTokenizer()


def create_data_proto_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Create a DataProto item from JSON data"""
    # Extract the required fields
    input_text = item.get("input", "")
    output_text = item.get("output", "")
    ground_truth = item.get("groundtruth", [])
    
    # Handle different ground truth formats
    if isinstance(ground_truth, list):
        if len(ground_truth) > 0:
            ground_truth = ground_truth[0]  # Take the first ground truth
        else:
            ground_truth = ""
    elif isinstance(ground_truth, str):
        pass
    else:
        ground_truth = str(ground_truth)
    
    # Create dummy tensors (not actually used for evaluation)
    dummy_tensor = torch.tensor([[1]], dtype=torch.long)
    
    # Create TensorDict for batch data
    batch_tensors = {
        "prompts": dummy_tensor,
        "responses": dummy_tensor,
        "attention_mask": dummy_tensor,
    }
    batch = TensorDict(source=batch_tensors, batch_size=[1])
    
    # Create non_tensor_batch with proper numpy arrays
    # All values must be numpy arrays with dtype=object and shape[0] == batch_size (1)
    non_tensor_batch = {
        "golden_answers": np.array([ground_truth], dtype=object),
        "data_source": np.array(["xverify"], dtype=object),  # Use xverify as data source
        "question": np.array([input_text], dtype=object),
        "extra_info": np.array([{
            "question": input_text,
            "output": output_text,
            "ground_truth": ground_truth
        }], dtype=object)
    }

        # Create the data item structure
    data_item = {
        "batch": batch,
        "non_tensor_batch": non_tensor_batch
    }
    
    return data_item


def evaluate_single_item(reward_manager: GenerativeRewardManager, 
                        item: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate a single JSON item"""
    # Create data proto item
    data_item = create_data_proto_item(item)
    
    # Create a minimal DataProto with one item
    # Extract batch and non_tensor_batch from the data_item
    batch = data_item["batch"]
    non_tensor_batch = data_item["non_tensor_batch"]

    # Create DataProto directly with the data
    data_proto = DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
    

    
    # Evaluate using the reward manager
    # try:
    result = reward_manager.offline_evaluate_single(data_proto, return_dict=True)
    result['golden_answers'] = non_tensor_batch['golden_answers']
    # except Exception as e:
    #     print(f"Error in offline_evaluate_single: {e}")
    #     # Return a default result with error information
    #     result = {
    #         "reward_tensor": np.array(0.0),
    #         "reward_extra_info": defaultdict(list, {
    #             "score": [0.0],
    #             "is_correct": [0.0],
    #             "evaluation_response": [f"Error: {str(e)}"],
    #             "error": [str(e)]
    #         }),
    #         "golden_answers": non_tensor_batch['golden_answers']
    #     }

    return result


def evaluate_json_file(json_file_path: str, 
                      url: str = "http://127.0.0.1:8808/v1",
                      eval_mode: str = "llm_judge",
                      verbose: bool = False) -> Dict[str, Any]:
    """
    Evaluate a JSON file containing LLM outputs and ground truth.
    
    Args:
        json_file_path: Path to the JSON file
        url: URL of the xverify model server
        verbose: Whether to enable verbose logging
    
    Returns:
        Dictionary containing evaluation results and statistics
    """
    setup_logging(verbose)
    logger = logging.getLogger(__name__)
    
    # Load JSON data
    logger.info(f"Loading JSON data from {json_file_path}")
    data = load_json_data(json_file_path)
    logger.info(f"Loaded {len(data)} items for evaluation")
    
    # Create reward manager with custom xverify config
    tokenizer = create_dummy_tokenizer()

    if eval_mode == "llm_judge":
        com_fn = llm_judge_compute_score
        r_manager = GenerativeRewardManager
        reward_manager = r_manager(
            tokenizer=tokenizer,
            num_examine=0,  # Don't print examples
            compute_score=com_fn,
            reward_fn_key="data_source"
        )

    elif eval_mode == 'xverify':
        com_fn = xverify_compute_score
        r_manager = GenerativeRewardManager
        reward_manager = r_manager(
            tokenizer=tokenizer,
            num_examine=0,  # Don't print examples
            compute_score=com_fn,
            reward_fn_key="data_source"
        )

    elif eval_mode == 'llm_judge_distributed':
        r_manager = GenerativeDistributedRewardManager
        com_fn = llm_judge_compute_score
        reward_manager = r_manager(
            tokenizer=tokenizer,
            num_examine=0,  # Don't print examples
            compute_score=com_fn,
            reward_fn_key="data_source",
            use_ray_distributed=True,
            num_ray_workers=32  # Adjust based on your system
        )

    else:
        raise NotImplementedError(f"Eval mode {eval_mode} is not supported")

    
    # Update xverify config with provided URL
    # reward_manager.xverify_config["url"] = [url]
    
    # Log xverify configuration for debugging
    # logger.info(f"Xverify config: {reward_manager.xverify_config}")
    
    # Evaluate each item
    results = []
    scores = []
    extra_info = defaultdict(list)
    
    logger.info("Starting evaluation...")
    
    if eval_mode == 'llm_judge_distributed':
        # Use distributed evaluation
        logger.info("Using distributed evaluation with Ray")
        
        # Create list of DataProto instances for distributed evaluation
        data_proto_list = []
        for item in data:
            data_item = create_data_proto_item(item)
            batch = data_item["batch"]
            non_tensor_batch = data_item["non_tensor_batch"]
            data_proto = DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
            data_proto_list.append(data_proto)
        
        # Run distributed evaluation
        distributed_result = reward_manager.offline_evaluate_distributed(data_proto_list, return_dict=True)
        
        # Process distributed results
        avg_score = float(distributed_result["reward_tensor"].item())
        
        # Extract extra info from distributed results
        if "reward_extra_info" in distributed_result:
            for key, values in distributed_result["reward_extra_info"].items():
                if values:  # Only add if there are values
                    extra_info[key].extend(values)
        
        # Create individual results for each item (using the average score)
        for i, item in enumerate(data):
            item["evaluation_score"] = avg_score
            item["evaluation_result"] = {
                "reward_tensor": [avg_score],
                "reward_extra_info": distributed_result.get("reward_extra_info", {})
            }
            results.append({
                "reward_tensor": np.array(avg_score),
                "reward_extra_info": distributed_result.get("reward_extra_info", {}),
                "golden_answers": data_proto_list[i].non_tensor_batch['golden_answers']
            })
            scores.append(avg_score)
        
    else:
        # Use sequential evaluation (original method)
        for i, item in enumerate(tqdm(data, desc="Evaluating")):
            # try:
            result = evaluate_single_item(reward_manager, item)
            results.append(result)

            # Extract scores and extra info
            score = 0.0  # Default score
            if "reward_tensor" in result:
                # Get the score from reward tensor
                reward_tensor = result["reward_tensor"]
                if isinstance(reward_tensor, np.ndarray):
                    score = float(reward_tensor.item())
                elif isinstance(reward_tensor, (list, tuple)) and len(reward_tensor) > 0:
                    score = float(reward_tensor[-1])
                elif hasattr(reward_tensor, 'item'):
                    score = float(reward_tensor.item())
                else:
                    score = float(reward_tensor)

            scores.append(score)

            if "reward_extra_info" in result:
                for key, values in result["reward_extra_info"].items():
                    if values:  # Only add if there are values
                        extra_info[key].extend(values)

            # Add original item data for reference
            item["evaluation_score"] = score
            # Convert evaluation result to JSON-serializable format
            serializable_result = {}
            for key, value in result.items():
                if isinstance(value, np.ndarray):
                    serializable_result[key] = value.tolist()
                elif isinstance(value, np.integer):
                    serializable_result[key] = int(value)
                elif isinstance(value, np.floating):
                    serializable_result[key] = float(value)
                else:
                    serializable_result[key] = value
            item["evaluation_result"] = serializable_result
            #
            # except Exception as e:
            #     logger.error(f"Error evaluating item {i}: {e}")
            #     scores.append(0.0)
            #     item["evaluation_error"] = str(e)
    
    # Compute statistics
    if scores:
        avg_score = sum(scores) / len(scores)
        correct_count = sum(1 for score in scores if score > 0.5)
        accuracy = correct_count / len(scores)
    else:
        avg_score = 0.0
        accuracy = 0.0
        correct_count = 0
    
    # Compute extra info statistics
    extra_stats = {}
    for key, values in extra_info.items():
        if values:
            if isinstance(values[0], (int, float)):
                extra_stats[f"avg_{key}"] = sum(values) / len(values)
                extra_stats[f"sum_{key}"] = sum(values)
            extra_stats[f"count_{key}"] = len(values)
    
    # Prepare final results
    evaluation_results = {
        "total_items": len(data),
        "evaluated_items": len(scores),
        "average_score": avg_score,
        "accuracy": accuracy,
        "correct_count": correct_count,
        "incorrect_count": len(scores) - correct_count,
        "extra_statistics": extra_stats,
        "detailed_results": data
    }
    
    logger.info(f"Evaluation completed:")
    logger.info(f"  Total items: {evaluation_results['total_items']}")
    logger.info(f"  Evaluated items: {evaluation_results['evaluated_items']}")
    logger.info(f"  Average score: {avg_score:.4f}")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Correct: {correct_count}")
    logger.info(f"  Incorrect: {evaluation_results['incorrect_count']}")
    
    return evaluation_results


def save_results(results: Dict[str, Any], output_file: str):
    """Save evaluation results to a JSON file"""
    def convert_numpy(obj):
        """Convert numpy arrays and other non-serializable objects to JSON-serializable types"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    # Convert numpy arrays to JSON-serializable types
    serializable_results = convert_numpy(results)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {output_file}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Evaluate JSON files containing LLM outputs and ground truth"
    )
    parser.add_argument(
        "--json_file",
        help="Path to the JSON file containing LLM outputs and ground truth"
    )
    parser.add_argument("--eval_mode", default="llm_judge", help="llm_judge, xverify")

    parser.add_argument(
        "--url",
        required=True,
        help="URL of the xverify or LLM judge model server."
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file path for evaluation results (default: evaluation_results.json)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.json_file).exists():
        print(f"Error: File {args.json_file} does not exist")
        return 1
    
    # Set default output file if not provided
    if not args.output:
        input_path = Path(args.json_file)
        args.output = input_path.parent / f"{input_path.stem}_{args.eval_mode}_evaluation_results.json"
    
    # try:
        # Run evaluation
    results = evaluate_json_file(
        json_file_path=args.json_file,
        url=args.url,
        verbose=args.verbose,
        eval_mode=args.eval_mode
    )
    
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Total items: {results['total_items']}")
    print(f"Evaluated items: {results['evaluated_items']}")
    print(f"Average score: {results['average_score']:.4f}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Correct: {results['correct_count']}")
    print(f"Incorrect: {results['incorrect_count']}")
    
    if results['extra_statistics']:
        print("\nExtra Statistics:")
        for key, value in results['extra_statistics'].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    
        # Save results
    save_results(results, args.output)
    return 0
        
    # except Exception as e:
    #     print(f"Error during evaluation: {e}")
    #     if args.verbose:
    #         import traceback
    #         traceback.print_exc()
    #     return 1


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
