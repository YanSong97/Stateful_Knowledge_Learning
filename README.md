<h1 align="center">Learning Stateful Predictive Knowledge from Experience</h1>

<p align="center">
  Yan Song, Xidong Feng, Bo Liu, Xinyu Cui, Zichen Liu, Haotian Fu,
  Mengyue Yang, Cheng Deng, Jian Zhao, Jun Wang
</p>

<p align="center">
  <a href="">
    <img src="https://img.shields.io/badge/Notion-Blog-000000?style=flat-square&logo=notion&logoColor=white" alt="Notion Blog">
  </a>
  <a href="">
    <img src="https://img.shields.io/badge/Paper-PDF-B31B1B?style=flat-square&logo=adobeacrobatreader&logoColor=white" alt="PDF Paper">
  </a>
  <a href="TODO_ARXIV_URL">
    <img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?style=flat-square&logo=arxiv&logoColor=white" alt="arXiv">
  </a>
</p>


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

# Create dummy datasets
python tasks/classic_games/chess/data_preprocess/dummy.py

# Run GRPO training
bash experiments/chesspuzzles/run_train_GRPO.sh

# Run NLTS-BFS training
bash experiments/chesspuzzles/run_train_NLTS_BFS_GRPO.sh
```

## Citation

```bibtex
@inproceedings{songlearning,
  title={Learning Stateful Predictive Knowledge From Experience},
  author={Song, Yan and Feng, Xidong and Liu, Bo and Cui, Xinyu and Liu, Zichen and Fu, Haotian and Yang, Mengyue and Deng, Cheng and Zhao, Jian and Wang, Jun},
  booktitle={Second Workshop on Agents in the Wild: Safety, Security, and Beyond}
}
```
