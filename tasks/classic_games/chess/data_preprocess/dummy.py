import re
import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./datasets/chess')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--template_type', type=str, default='base')

    args = parser.parse_args()

    data_source = 'chess'

    # Create dummy dataset with 128 examples
    dummy_train_data = {
        'query': [f"Placeholder for game training data {i}" for i in range(10240)],
        'gt': [f"Answer {i}" for i in range(10240)]
    }
    dummy_test_data = {
        'query': [f"Placeholder for game testing data {i}" for i in range(10240)],
        'gt': [f"Answer {i}" for i in range(10240)]
    }
    
    # Create a Dataset from dictionary
    train_dataset = datasets.Dataset.from_dict(dummy_train_data)
    test_dataset = datasets.Dataset.from_dict(dummy_test_data)

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