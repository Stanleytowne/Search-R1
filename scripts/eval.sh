#!/bin/bash
set -e

export WANDB_API_KEY=
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
TRAIN_FILE=data/toolbench_rl/${CATEGORY}.parquet
VAL_FILE=data/toolbench_test/${CATEGORY}.parquet
MODEL_PATH=${2}

TOOLBENCH_URL=http://127.0.0.1:12345
REWARD_SERVER_URL=http://localhost:12346/evaluate_batch

WANDB_PROJECT=toolbench_eval
EXPERIMENT_NAME=${CATEGORY}

PASS_REWARD_WEIGHT=1

VAL_BATCH_SIZE=$(python3 -c "import pandas as pd; import sys; val_file='$VAL_FILE'; df=pd.read_parquet(val_file); print(len(df))")

# 运行训练
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo_toolbench \
    --config-name=grpo_toolbench_trainer \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    data.val_batch_size="$VAL_BATCH_SIZE" \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    toolbench_url="$TOOLBENCH_URL" \
    use_toolbench=true \
    reward_model.reward_server_url="$REWARD_SERVER_URL" \
    algorithm.adv_estimator=grpo \
    reward_model.pass_reward_weight="$PASS_REWARD_WEIGHT" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.project_name="$WANDB_PROJECT" \
    trainer.n_gpus_per_node="$NUM_GPUS" \
    +trainer.val_before_train=true \
    +trainer.val_only=true \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384