
"""
each data instance contain a chesspuzzle state, and the query is to evaluate the state and provide the best action
"""

import os
import asyncio
import datasets
from dataclasses import dataclass, field
from typing import Optional, List, Any
from tqdm import tqdm

from verl.utils.hdfs_io import copy, makedirs
import argparse


@dataclass
class StockfishRewardConfig:
    """Configuration for Stockfish reward settings"""
    enable: bool = False  # use stockfish evaluator to give as reward in (external) env
    scaling_factor: float = 0.001
    use_as_planning_reward: bool = False  # use as reward in internal planning env
    use_as_feedback: bool = False  # also return state value as environment feedback
    use_as_planning_feedback: bool = False  # return state value as each planning feedback


@dataclass
class RolloutConfig:
    """Configuration for rollout settings"""
    n: Any = None  # the group number of envs (for GRPO and GiGPO). -1 means disable env grouping.
    train_seed: List[int] = field(default_factory=lambda: [0, 10000])  # eval(train_seed)
    val_seed: List[int] = field(default_factory=lambda: [4096, 8192])
    train_val_ratio: float = 0.2
    conversation_prefix: str = "user"
    max_plan_traj_n: int = 2
    max_plan_horizon: int = 2


@dataclass
class ChessEnvConfig:
    """Configuration for Chess Environment"""
    env_name: str = "ChessPuzzles"
    game_mode: str = "puzzles"  # puzzle or full
    full_game_n: Optional[int] = None
    use_stockfish: bool = True  # whether to initialize stockfish engine
    stockfish_url: str = "http://127.0.0.1:8080"  # remote stockfish engine url
    opponent_mode: str = "random"
    max_steps: int = 10
    rating_range: List[int] = field(default_factory=lambda: [0, 1000])  # selecting difficulty of puzzles
    puzzle_scope: Optional[Any] = None  # getting the first 100 samples
    move_format: str = "san"
    board_format: str = "fen"
    stockfish_reward: StockfishRewardConfig = field(default_factory=StockfishRewardConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    prompt_set: str = "tasks.classic_games.chess.prompt.QWEN_SIMPLEST_PLAN_PROMPT_SET"
    template_set: str = "src.models.qwen2.tokenizer.QWEN25_TEMPLATE"  # refresh model chat template, e.g. Qwen2.5's
    action_processor: str = "tasks.classic_games.chess.processor.chess_extract_best_move"
    max_legal_action_n: int = 30
    act_action_format_penality: float = 0.02
    plan_action_format_penality: float = 0.0  # if 0, plan action format penalty is not used
    plan_action_reward_scale: float = 0.0  # if 0, env reward during planning is not used
    act_reward_scale: float = 1.0  # acting phase reward scaling
    use_stockfish_move_matching_reward: bool = False  # whether to use stockfish move matching reward
    stockfish_move_matching_reward: float = 0.5
    stockfish_move_topk: int = 1  # the top k moves to match with the planned action


def get_env(async_mode):
    if async_mode:
        from tasks.classic_games.chess.chess_puzzle_env import AsyncChessPuzzleEnv_Wrapper
        return AsyncChessPuzzleEnv_Wrapper(env_config=ChessEnvConfig())
    else:
        from tasks.classic_games.chess.chess_puzzle_env import ChessPuzzleEnv_Wrapper
        return ChessPuzzleEnv_Wrapper(env_config=ChessEnvConfig())



async def get_state_topk_action(game_idx, top_k_action):
    env = get_env(async_mode=True)

    initial_state = env.reset(specified_game_idx=game_idx)
    puzzle_id = env.puzzle_id
    rating = env.puzzle_rating
    legal_moves = env.legal_moves_list

    analysis_result = await env.analyse_position(num_moves=top_k_action)
    best_action = analysis_result[0]

    ret = {
        "state": initial_state,
        "legal_moves": legal_moves,
        "best_action": best_action,
        "top_k_actions": analysis_result,
        "puzzle_id": puzzle_id,
        "rating": rating,
    }

    return ret




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./datasets/chess')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--template_type', type=str, default='base')
    parser.add_argument('--top_k_action', type=int, default=3)
    parser.add_argument('--max_concurrent', type=int, default=32, help='Maximum number of concurrent async tasks')

    args = parser.parse_args()

    data_source = 'chess_state_eval'
    cfg = ChessEnvConfig()


    sync_env = get_env(async_mode=False)
    num_games = sync_env.num_game

    # train test split

    test_size = int(num_games * cfg.rollout.train_val_ratio)
    train_size = num_games - test_size
    test_indices = list(range(test_size))
    train_indices = list(range(test_size, num_games))
    print(f"test_indices: {test_indices}, train_indices: {train_indices}")
    train_data_dict = {"state": [], "legal_moves": [], "best_action": [], "top_k_actions": []}
    test_data_dict = {"state": [], "legal_moves": [], "best_action": [], "top_k_actions": []}

    # Helper async function to run tasks with concurrency limit and progress bar
    async def run_tasks_with_limit(tasks, description, max_concurrent):
        semaphore = asyncio.Semaphore(max_concurrent)
        results = [None] * len(tasks)
        
        # Create progress bar
        pbar = tqdm(total=len(tasks), desc=description, unit="task", dynamic_ncols=True)
        
        async def bounded_task_with_update(task, idx):
            async with semaphore:
                try:
                    result = await task
                    # Update progress bar from within the task
                    pbar.update(1)
                    return (idx, result)
                except Exception as e:
                    pbar.update(1)
                    raise e
        
        # Create tasks with indices
        async_tasks = [asyncio.create_task(bounded_task_with_update(task, i)) 
                      for i, task in enumerate(tasks)]
        
        # Wait for all tasks to complete
        try:
            for coro in asyncio.as_completed(async_tasks):
                try:
                    idx, result = await coro
                    results[idx] = result
                except Exception as e:
                    raise e
        finally:
            pbar.close()
        
        return results

    # Run async calls concurrently for train indices
    print(f"\nProcessing {len(train_indices)} train samples...")
    train_tasks = [get_state_topk_action(i, args.top_k_action) for i in train_indices]
    train_results = asyncio.run(run_tasks_with_limit(train_tasks, "Train", args.max_concurrent))
    
    for state_topk_action in train_results:
        train_data_dict["state"].append(state_topk_action["state"])
        train_data_dict["legal_moves"].append(state_topk_action["legal_moves"])
        train_data_dict["best_action"].append(state_topk_action["best_action"])
        train_data_dict["top_k_actions"].append(state_topk_action["top_k_actions"])
    
    # Run async calls concurrently for test indices
    print(f"\nProcessing {len(test_indices)} test samples...")
    test_tasks = [get_state_topk_action(j, args.top_k_action) for j in test_indices]
    test_results = asyncio.run(run_tasks_with_limit(test_tasks, "Test", args.max_concurrent))
    
    for state_topk_action in test_results:
        test_data_dict["state"].append(state_topk_action["state"])
        test_data_dict["legal_moves"].append(state_topk_action["legal_moves"])
        test_data_dict["best_action"].append(state_topk_action["best_action"])
        test_data_dict["top_k_actions"].append(state_topk_action["top_k_actions"])

    train_dataset = datasets.Dataset.from_dict(train_data_dict)
    test_dataset = datasets.Dataset.from_dict(test_data_dict)


    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):

            question = example['query']
            solution = {
                "target": example['gt'],
            }


            data = {
                "id": f"{split}_{idx}",
                "question": question,
                "golden_answers": example['gt'],
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                # "prompt": example['prompt'],
                "ability": "game-playing",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                # "reward_model": example["reward_model"],
                "extra_info": {
                    'split': split,
                    'index': idx,
                },
                # "extra_info": example['extra_info']
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)