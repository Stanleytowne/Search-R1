#!/bin/bash
# SFT训练脚本

set -e

# 配置
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
export VLLM_ATTENTION_BACKEND=XFORMERS

# 数据路径
TRAIN_FILE=/scratch/mrm2vx/tpz/Search-R1/data/toolbench_stage2/Sports.parquet
VAL_FILE=/scratch/mrm2vx/tpz/Search-R1/data/toolbench_stage2/Sports.parquet
SYSTEM_PROMPT_KEY=system

# 模型路径
MODEL_PATH=Qwen/Qwen2.5-7B

# 训练配置
CONFIG_FILE=verl/trainer/config/sft_trainer.yaml
EXPERIMENT_NAME=toolbench_stage2_$(date +%Y%m%d_%H%M%S)
WANDB_PROJECT=Search-R1-stage2
TOTAL_EPOCHS=1
NUM_GPUS=4
LR=1e-5
BATCH_SIZE=64
MICRO_BATCH_SIZE=4
MAX_LENGTH=4096

# 运行训练
torchrun --nproc_per_node=4 verl/trainer/fsdp_sft_trainer.py \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    data.prompt_key=prompt \
    data.response_key=response \
    data.system_prompt_key="$SYSTEM_PROMPT_KEY" \
    data.max_length="$MAX_LENGTH" \
    data.train_batch_size="$BATCH_SIZE" \
    data.micro_batch_size="$MICRO_BATCH_SIZE" \
    optim.lr="$LR" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.project_name="$WANDB_PROJECT" \
    trainer.n_gpus_per_node="$NUM_GPUS" \
    trainer.total_epochs="$TOTAL_EPOCHS" \
    trainer.default_local_dir="checkpoints/$EXPERIMENT_NAME" \
    "$@"