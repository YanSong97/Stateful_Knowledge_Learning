set -x

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(realpath "$SCRIPT_DIR/../..")"
echo "root path: ${ROOT_DIR}"

export export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export CUDA_VISIBLE_DEVICES=0
GPU_NUM=1
export WANDB_MODE=offline
export SWANLAB_MODE=local

CONFIG_NAME="value_lpm_BFS_distill_chess_loop_trainer.yaml"
CONFIG_PATH=$ROOT_DIR/src/config/chesspuzzle/$CONFIG_NAME
AGENT_TYPE="Distill_NLTS_BFS"    # MDP, LPM

MODEL_PATH=$MODEL_PATH
MODEL_TAG="Qwen2.5-3B"

DISTILL_MODE="op-sd"
POLICY_LOSS="distillation"

TRAIN_DATA_PATH=$ROOT_DIR/datasets/chess/train.parquet
TEST_DATA_PATH=$ROOT_DIR/datasets/chess/test.parquet
DATA_TAG="chess"

BATCH_SIZE=512
MAX_PROMPT_LENGTH=700
MAX_RESPONSE_LENGTH=56000
EACH_RESPONSE_LENGTH=1024
EACH_SUMMARY_RESPONSE_LENGTH=1024


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

MAX_STEPS=10
MAX_PLAN_TRAJ_N=3
MAX_PLAN_HORIZON=1

WANDB_PROJECT=NLTS
TIME_TAG=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_NAME="EXP_Distill_800_N3H1_RuleOppo_SingleGPU_LR1e-6_KL002_NewTrial_${AGENT_TYPE}_${MODEL_TAG}_turn${MAX_STEPS}_${DATA_TAG}_${TIME_TAG}"

GENERATION_SAVE_PATH=$ROOT_DIR/logs/train/${DATA_TAG}/${AGENT_TYPE}/${MODEL_TAG}_${TIME_TAG}
BUFFER_SAVE_PATH=$ROOT_DIR/logs/buffer/rollout_batch.pkl

python -m src.train_chess \
    --config-path=./config/chesspuzzle \
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
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.02 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.rollout.save_batch_to_path=$BUFFER_SAVE_PATH \
    actor_rollout_ref.rollout.response_length=$MAX_RESPONSE_LENGTH \
    actor_rollout_ref.rollout.each_response_length=$EACH_RESPONSE_LENGTH \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.mode=$MODE \
    actor_rollout_ref.rollout.agent.num_workers=32 \
    actor_rollout_ref.rollout.multi_turn.envs.env_config.rating_range=[0,800] \
    actor_rollout_ref.rollout.multi_turn.envs.env_config.max_steps=$MAX_STEPS \
    actor_rollout_ref.rollout.multi_turn.envs.env_config.opponent_mode=fixed \
    actor_rollout_ref.rollout.multi_turn.envs.env_config.prompt_set="tasks.classic_games.chess.prompt.QWEN_TWO_STAGE_VALUE_PARALLEL_PLAN_PROMPT_SET" \
    actor_rollout_ref.rollout.multi_turn.envs.env_config.move_format="san" \
    actor_rollout_ref.rollout.multi_turn.envs.env_config.board_format="fen" \
    actor_rollout_ref.rollout.multi_turn.envs.env_config.rollout.max_plan_traj_n=$MAX_PLAN_TRAJ_N \
    actor_rollout_ref.rollout.multi_turn.envs.env_config.rollout.max_plan_horizon=$MAX_PLAN_HORIZON \
    actor_rollout_ref.rollout.multi_turn.post_process.tool_masking=True \
    actor_rollout_ref.rollout.multi_turn.independent_summary=False \
    actor_rollout_ref.rollout.multi_turn.random_move_when_invalid_plan_expansion=False \
    actor_rollout_ref.rollout.multi_turn.random_move_when_invalid_plan_simulation=False \
    actor_rollout_ref.rollout.multi_turn.random_move_when_invalid_act=False \
    actor_rollout_ref.rollout.multi_turn.envs.env_config.act_action_format_penality=0.01 \
    actor_rollout_ref.rollout.multi_turn.envs.env_config.plan_action_format_penality=0. \
    actor_rollout_ref.rollout.multi_turn.envs.env_config.plan_action_reward_scale=0. \
    actor_rollout_ref.rollout.multi_turn.envs.env_config.act_reward_scale=1. \
    actor_rollout_ref.rollout.multi_turn.summary_response_length=$EACH_SUMMARY_RESPONSE_LENGTH \
    actor_rollout_ref.rollout.multi_turn.final_reward_readjustment=False \
    actor_rollout_ref.actor.include_distillation_loss=True \
    actor_rollout_ref.actor.policy_loss.loss_mode=$POLICY_LOSS \
    actor_rollout_ref.actor.distillation_mode=$DISTILL_MODE \
    actor_rollout_ref.actor.distillation_loss_coef=1. \
    actor_rollout_ref.actor.self_distillation.full_logit_distillation=False \
    actor_rollout_ref.actor.self_distillation.distillation_topk=20 \
    actor_rollout_ref.actor.self_distillation.distillation_add_tail=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    custom_reward_function.name=$REWARD_FN_NAME \
    reward_model.reward_manager=$REWARD_MANAGER \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.log_val_generations=0 \
    trainer.critic_warmup=0 \
    trainer.val_before_train=True \
    trainer.logger=['console','swanlab'] \
    trainer.validation_data_dir=$GENERATION_SAVE_PATH \
    trainer.n_gpus_per_node=$GPU_NUM \
    trainer.project_name=$WANDB_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.nnodes=1 \
    trainer.save_freq=200 \
    trainer.test_freq=10 \
    trainer.total_epochs=30 $@
