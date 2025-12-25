#!/bin/bash
# ToolBench GRPO训练脚本

set -e

if [ -f "wandb_api.json" ]; then
    WANDB_API_KEY=$(python3 -c "
import json
with open('wandb_api.json') as f:
    d = json.load(f)
print(d.get('WANDB_API_KEY', ''))
" )
    if [ ! -z "$WANDB_API_KEY" ]; then
        export WANDB_API_KEY="$WANDB_API_KEY"
        echo "Loaded WANDB_API_KEY from wandb_api.json"
    else
        echo "wandb_api.json found but WANDB_API_KEY is empty"
    fi
else
    echo "wandb_api.json not found, using existing WANDB_API_KEY if set"
fi


# 配置
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
export VLLM_ATTENTION_BACKEND=XFORMERS

# 根据CUDA_VISIBLE_DEVICES自动设置GPU数量
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

# 数据路径
TRAIN_FILE="${TRAIN_FILE:-data/toolbench/Sports.parquet}"
VAL_FILE="${VAL_FILE:-data/toolbench/Sports.parquet}"

# 模型路径
MODEL_PATH="${MODEL_PATH:-checkpoints/Sports/stage1/global_step_327}"


# ToolBench服务器
TOOLBENCH_URL="${TOOLBENCH_URL:-http://127.0.0.1:8080}"
REWARD_SERVER_URL="${REWARD_SERVER_URL:-http://localhost:1234/evaluate_batch}"

# 训练配置
CONFIG_FILE="${CONFIG_FILE:-verl/trainer/config/grpo_toolbench_trainer.yaml}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-toolbench_grpo_$(date +%Y%m%d_%H%M%S)}"
WANDB_PROJECT="${WANDB_PROJECT:-Search-R1}"

# Reward权重
FORMAT_REWARD_WEIGHT="${FORMAT_REWARD_WEIGHT:-1}"
FUNCTION_CALL_REWARD_WEIGHT="${FUNCTION_CALL_REWARD_WEIGHT:-1}"
FINISH_REWARD_WEIGHT="${FINISH_REWARD_WEIGHT:-1}"
PASS_REWARD_WEIGHT="${PASS_REWARD_WEIGHT:-1}"

# GRPO特定配置
N_AGENT="${N_AGENT:-4}"  # 每个prompt生成的响应数
LEARNING_RATE="${LEARNING_RATE:-5e-7}"
LR_WARMUP_RATIO="${LR_WARMUP_RATIO:-0.285}"
EPOCHS="${EPOCHS:-10}"

echo "Starting ToolBench GRPO training..."
echo "  Train file: $TRAIN_FILE"
echo "  Val file: $VAL_FILE"
echo "  Model: $MODEL_PATH"
echo "  ToolBench URL: $TOOLBENCH_URL"
echo "  Experiment: $EXPERIMENT_NAME"
echo "  GPUs: $NUM_GPUS (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "  N_AGENT: $N_AGENT (GRPO需要多个响应)"
echo ""

# 运行训练
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo_toolbench \
    --config-name=grpo_toolbench_trainer \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
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
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.project_name="$WANDB_PROJECT" \
    trainer.n_gpus_per_node="$NUM_GPUS" \
    +trainer.val_before_train=false \
    trainer.test_freq=-1 \
    trainer.total_epochs="$EPOCHS" \
    trainer.save_freq=100 \
    "$@"

echo ""
echo "Training completed!"
