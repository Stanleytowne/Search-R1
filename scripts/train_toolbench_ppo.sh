#!/bin/bash
# ToolBench PPO训练脚本

set -e

# 配置
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN

# 数据路径
DATA_DIR="${DATA_DIR:-~/data/toolbench}"
TRAIN_FILE="${TRAIN_FILE:-$DATA_DIR/train.parquet}"
VAL_FILE="${VAL_FILE:-$DATA_DIR/val.parquet}"

# 模型路径
MODEL_PATH="${MODEL_PATH:-~/models/deepseek-llm-7b-chat}"

# ToolBench服务器
TOOLBENCH_URL="${TOOLBENCH_URL:-http://127.0.0.1:8000}"
TOOLBENCH_KEY="${TOOLBENCH_KEY:-}"

# 训练配置
CONFIG_FILE="${CONFIG_FILE:-verl/trainer/config/ppo_toolbench_trainer.yaml}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-toolbench_ppo_$(date +%Y%m%d_%H%M%S)}"

# Reward权重
FORMAT_REWARD_WEIGHT="${FORMAT_REWARD_WEIGHT:-0.1}"
FUNCTION_CALL_REWARD_WEIGHT="${FUNCTION_CALL_REWARD_WEIGHT:-0.2}"
FINISH_REWARD_WEIGHT="${FINISH_REWARD_WEIGHT:-0.3}"
ERROR_PENALTY="${ERROR_PENALTY:--0.5}"
FINISH_BONUS="${FINISH_BONUS:-0.5}"

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

echo "Starting ToolBench PPO training..."
echo "  Train file: $TRAIN_FILE"
echo "  Val file: $VAL_FILE"
echo "  Model: $MODEL_PATH"
echo "  ToolBench URL: $TOOLBENCH_URL"
echo "  Experiment: $EXPERIMENT_NAME"
echo ""

# 运行训练
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo_toolbench \
    --config-name=ppo_toolbench_trainer \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    critic.model.path="$MODEL_PATH" \
    use_toolbench=true \
    toolbench_url="$TOOLBENCH_URL" \
    toolbench_key="$TOOLBENCH_KEY" \
    default_category="G1_category" \
    reward_model.format_reward_weight="$FORMAT_REWARD_WEIGHT" \
    reward_model.function_call_reward_weight="$FUNCTION_CALL_REWARD_WEIGHT" \
    reward_model.finish_reward_weight="$FINISH_REWARD_WEIGHT" \
    reward_model.error_penalty="$ERROR_PENALTY" \
    reward_model.finish_bonus="$FINISH_BONUS" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    "$@"

echo ""
echo "Training completed!"
