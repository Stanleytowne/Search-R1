#!/bin/bash
# ToolBench GRPO训练脚本

set -e

# 配置
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
export VLLM_ATTENTION_BACKEND=XFORMERS

# 数据路径
DATA_DIR="${DATA_DIR:-data/toolbench}"
TRAIN_FILE="${TRAIN_FILE:-$DATA_DIR/train.parquet}"
VAL_FILE="${VAL_FILE:-$DATA_DIR/val.parquet}"

# 模型路径
MODEL_PATH="${MODEL_PATH:-ToolBench/ToolLLaMA-2-7b-v2}"

# ToolBench服务器
TOOLBENCH_URL="${TOOLBENCH_URL:-http://10.153.48.52:8080}"
TOOLBENCH_KEY="${TOOLBENCH_KEY:-}"

# 训练配置
CONFIG_FILE="${CONFIG_FILE:-verl/trainer/config/grpo_toolbench_trainer.yaml}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-toolbench_grpo_$(date +%Y%m%d_%H%M%S)}"
WANDB_PROJECT="${WANDB_PROJECT:-Search-R1}"

# Reward权重
FORMAT_REWARD_WEIGHT="${FORMAT_REWARD_WEIGHT:-0.1}"
FUNCTION_CALL_REWARD_WEIGHT="${FUNCTION_CALL_REWARD_WEIGHT:-0.2}"
FINISH_REWARD_WEIGHT="${FINISH_REWARD_WEIGHT:-0.3}"
ERROR_PENALTY="${ERROR_PENALTY:--0.5}"
FINISH_BONUS="${FINISH_BONUS:-0.5}"

# GRPO特定配置
N_AGENT="${N_AGENT:-5}"  # 每个prompt生成的响应数
LEARNING_RATE="${LEARNING_RATE:-5e-7}"
LR_WARMUP_RATIO="${LR_WARMUP_RATIO:-0.285}"

# 检查数据文件
if [ ! -f "$TRAIN_FILE" ]; then
    echo "Error: Training file not found: $TRAIN_FILE"
    echo "Please run the data conversion script first:"
    echo "  python scripts/convert_toolbench_to_verl.py --input <json_file> --output $TRAIN_FILE --split"
    exit 1
fi

if [ ! -f "$VAL_FILE" ]; then
    echo "Error: Validation file not found: $VAL_FILE"
    exit 1
fi

# 检查ToolBench服务器
echo "Checking ToolBench server at $TOOLBENCH_URL..."
if ! curl -s "$TOOLBENCH_URL/health" > /dev/null 2>&1; then
    echo "Warning: ToolBench server may not be running at $TOOLBENCH_URL"
    echo "Please start the server:"
    echo "  cd StableToolBench/server && bash start_server.sh"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "Starting ToolBench GRPO training..."
echo "  Train file: $TRAIN_FILE"
echo "  Val file: $VAL_FILE"
echo "  Model: $MODEL_PATH"
echo "  ToolBench URL: $TOOLBENCH_URL"
echo "  Experiment: $EXPERIMENT_NAME"
echo "  N_AGENT: $N_AGENT (GRPO需要多个响应)"
echo ""

# 运行训练
# 注意：不使用--config-name，直接覆盖配置项，避免Hydra路径问题
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo_toolbench \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    data.max_prompt_length=4096 \
    data.max_response_length=4096 \
    data.max_start_length=2048 \
    data.max_obs_length=500 \
    data.train_batch_size=64 \
    data.val_batch_size=64 \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size=4 \
    actor_rollout_ref.actor.optim.lr="$LEARNING_RATE" \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio="$LR_WARMUP_RATIO" \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.ref.log_prob_micro_batch_size=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.n_agent="$N_AGENT" \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    use_toolbench=true \
    toolbench_url="$TOOLBENCH_URL" \
    toolbench_key="$TOOLBENCH_KEY" \
    default_category="G1_category" \
    algorithm.adv_estimator=grpo \
    algorithm.no_think_rl=false \
    reward_model.format_reward_weight="$FORMAT_REWARD_WEIGHT" \
    reward_model.function_call_reward_weight="$FUNCTION_CALL_REWARD_WEIGHT" \
    reward_model.finish_reward_weight="$FINISH_REWARD_WEIGHT" \
    reward_model.error_penalty="$ERROR_PENALTY" \
    reward_model.finish_bonus="$FINISH_BONUS" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.project_name="$WANDB_PROJECT" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    max_turns=5 \
    "$@"

echo ""
echo "Training completed!"
