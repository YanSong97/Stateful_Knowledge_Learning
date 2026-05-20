set -x

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(realpath "$SCRIPT_DIR/../..")"
echo "root path: ${ROOT_DIR}"

export export VLLM_USE_V1=1
export CUDA_VISIBLE_DEVICES=0
GPU_NUM=1
export WANDB_MODE=offline   # offline mode for wandb
export SWANLAB_MODE=local   # local mode for swanlab

CONFIG_NAME="value_infer_frozenlake_trainer.yaml"
CONFIG_PATH=$ROOT_DIR/src/config/frozenlake/$CONFIG_NAME
AGENT_TYPE="GRPO"    # MDP, LPM

MODEL_PATH=$MODEL_PATH
MODEL_TAG="Qwen3-8B"

TRAIN_DATA_PATH=$ROOT_DIR/datasets/chess/train.parquet
TEST_DATA_PATH=$ROOT_DIR/datasets/chess/test.parquet
DATA_TAG="frozenlake"

BATCH_SIZE=512
MAX_PROMPT_LENGTH=2400
MAX_RESPONSE_LENGTH=20000
EACH_RESPONSE_LENGTH=1024


LR=1e-6
MINI_BATCH_SIZE=128
MICRO_BATCH_SIZE=4

MODE=async

#============ Evaluation/Reward Manager ========================#
REWARD_FN_PATH=null
REWARD_FN_NAME="compute_score"
REWARD_MANAGER="mdp"

#==============================================================#

#============ Multi-Turn settings ========================#

MAX_STEPS=20
IS_SLIPPERY=True
GAME_SIZE=5
NUM_MAP=100

WANDB_PROJECT=NLTS
TIME_TAG=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_NAME="FrozenLake_GRPO_${MODEL_TAG}_turn${MAX_STEPS}_${DATA_TAG}_${TIME_TAG}"

GENERATION_SAVE_PATH=$ROOT_DIR/logs/train/${DATA_TAG}/${AGENT_TYPE}/${MODEL_TAG}_${TIME_TAG}

python -m src.train_chess \
    --config-path=./config/frozenlake \
    --config-name=$CONFIG_NAME \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_DATA_PATH \
    data.val_files=$TEST_DATA_PATH \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=4096 \
    data.train_batch_size=64 \
    data.val_batch_size=64 \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    data.prompt_key='prompt' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.rollout.response_length=$MAX_RESPONSE_LENGTH \
    actor_rollout_ref.rollout.each_response_length=$EACH_RESPONSE_LENGTH \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.mode=$MODE \
    actor_rollout_ref.rollout.multi_turn.envs.env_config.max_steps=$MAX_STEPS \
    actor_rollout_ref.rollout.multi_turn.envs.env_config.is_slippery=$IS_SLIPPERY \
    actor_rollout_ref.rollout.multi_turn.envs.env_config.size=$GAME_SIZE \
    actor_rollout_ref.rollout.multi_turn.envs.env_config.num_map=$NUM_MAP \
    actor_rollout_ref.rollout.multi_turn.envs.env_config.prompt_set="tasks.classic_games.chess.prompt.QWEN_TWO_STAGE_VALUE_PROMPT_SET_CLEANEST" \
    actor_rollout_ref.rollout.multi_turn.post_process.tool_masking=True \
    actor_rollout_ref.rollout.multi_turn.envs.env_config.act_action_format_penality=0.02 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    custom_reward_function.name=$REWARD_FN_NAME \
    reward_model.reward_manager=$REWARD_MANAGER \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.log_val_generations=0 \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.logger=['console','wandb'] \
    trainer.validation_data_dir=$GENERATION_SAVE_PATH \
    trainer.n_gpus_per_node=$GPU_NUM \
    trainer.project_name=$WANDB_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.nnodes=1 \
    trainer.save_freq=200 \
    trainer.test_freq=10 \
    trainer.total_epochs=30 $@