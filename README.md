# Learning Stateful Predictive Knowledge from Experience

This is the codebase for paper "Learning Stateful Predictive Knowledge from Experience".

More details will be added soon.


## Setup

```bash
conda create -n SKL python==3.12
conda activate SKL

pip install -r requirements.txt

cd verl
pip install -e . --no-deps

```


## Running experiments

```bash

export MODEL_PATH=xxx
export PYTHONPATH=./

# create dummy datasets
python tasks/classic_games/chess/data_preprocess/dummy.py

# Run GRPO training
bash experiments/chesspuzzles/run_train_GRPO.sh

# Run NLTS-BFS training
bash experiments/chesspuzzles/run_train_NLTS_BFS_GRPO.sh
```