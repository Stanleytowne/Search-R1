#!/bin/bash
set -x
set -e

cd ../Long-Digestor-Experiments
export WANDB_API_KEY=

category=${1:-Search}

model_path=/ceph/home/muhan01/huggingfacemodels/Qwen2.5-7B-Instruct
train_data_path=../Search-R1/data/toolbench_stage1/${category}.parquet
val_data_path=../Search-R1/data/toolbench_stage1/${category}.parquet
nproc_per_node=8
save_path=../Search-R1/checkpoints/${category}/stage1
epochs=3
lr=5e-5
batch_size=64
micro_batch_size_per_gpu=2

project_name=toolbench-sft-stage1
experiment_name=${category}

echo "=========================================="
echo "Starting Stage 1 Training"
echo "=========================================="

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_batch_size=$batch_size \
    data.micro_batch_size_per_gpu=$micro_batch_size_per_gpu \
    data.train_files=$train_data_path \
    data.val_files=$val_data_path \
    data.prompt_key=prompt \
    data.response_key=response \
    data.max_length=8192 \
    optim.lr=$lr \
    model.partial_pretrain=$model_path \
    model.fsdp_config.model_dtype=bf16 \
    model.use_liger=True \
    model.strategy=fsdp2 \
    trainer.default_local_dir=$save_path \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.total_epochs=$epochs \
    trainer.logger=['console','wandb'] \
    trainer.default_hdfs_dir=null


echo "=========================================="
echo "Stage 1 Training Completed"
echo "=========================================="

# 获取最新的 checkpoint
echo "Finding latest checkpoint..."
latest_step=$(ls "$save_path" | grep "global_step_" | sed 's/global_step_//' | sort -n | tail -1)

if [ -z "$latest_step" ]; then
    echo "Error: No global_step directories found in $save_path"
    exit 1
fi

latest_checkpoint="$save_path/global_step_$latest_step"
echo "Latest checkpoint: $latest_checkpoint"

# Stage 2 配置
train_data_path=../Search-R1/data/toolbench_stage2/${category}.parquet
val_data_path=../Search-R1/data/toolbench_stage2/${category}.parquet
nproc_per_node=8
save_path=../Search-R1/checkpoints/${category}/stage2
epochs=3
lr=2e-5
batch_size=32
micro_batch_size_per_gpu=2

project_name=toolbench-sft-stage2

# Stage 2 训练
echo "=========================================="
echo "Starting Stage 2 Training"
echo "=========================================="

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_batch_size=$batch_size \
    data.micro_batch_size_per_gpu=$micro_batch_size_per_gpu \
    data.train_files=$train_data_path \
    data.val_files=$val_data_path \
    data.prompt_key=prompt \
    data.response_key=response \
    data.system_prompt_key=system \
    data.max_length=8192 \
    optim.lr=$lr \
    model.partial_pretrain=$latest_checkpoint \
    model.fsdp_config.model_dtype=bf16 \
    model.use_liger=True \
    trainer.default_local_dir=$save_path \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.total_epochs=$epochs \
    trainer.logger=['console','wandb'] \
    trainer.default_hdfs_dir=null

echo "=========================================="
echo "Stage 2 Training Completed!"
echo "=========================================="
