#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
export VLLM_ATTENTION_BACKEND=XFORMERS
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

if [ -z "$1" ]; then
    echo "Usage: $0 <category> <model_path>"
    exit 1
fi

CATEGORY=${1}
MODEL_PATH=${2}
TRAIN_FILE=data/toolbench_rl/${CATEGORY}.parquet
VAL_FILE=data/toolbench_test/${CATEGORY}.parquet

TOOLBENCH_URL=http://127.0.0.1:12345
REWARD_SERVER_URL=http://localhost:12346/evaluate_batch

# Reward权重
FORMAT_REWARD_WEIGHT=1
FUNCTION_CALL_REWARD_WEIGHT=1
FINISH_REWARD_WEIGHT=1
PASS_REWARD_WEIGHT=1

# GRPO特定配置
N_AGENT=4
LEARNING_RATE=1e-6
LR_WARMUP_RATIO=0
EPOCHS=10

# 运行训练
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo_toolbench \
    --config-name=grpo_toolbench_trainer \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    data.val_batch_size=1 \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr="$LEARNING_RATE" \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio="$LR_WARMUP_RATIO" \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.rollout.n_agent="$N_AGENT" \
    toolbench_url="$TOOLBENCH_URL" \
    reward_model.reward_server_url="$REWARD_SERVER_URL" \
    algorithm.adv_estimator=grpo \
    reward_model.format_reward_weight="$FORMAT_REWARD_WEIGHT" \
    reward_model.function_call_reward_weight="$FUNCTION_CALL_REWARD_WEIGHT" \
    reward_model.finish_reward_weight="$FINISH_REWARD_WEIGHT" \
    reward_model.pass_reward_weight="$PASS_REWARD_WEIGHT" \
    trainer.n_gpus_per_node="$NUM_GPUS" \
    +trainer.val_before_train=true \
    trainer.test_freq=10 \
    trainer.total_epochs="$EPOCHS" \
    trainer.save_freq=25 \
    trainer.default_local_dir=checkpoints/$CATEGORY/stage-rl \
    trainer.logger=['console'] \
    trainer.val_only=true
