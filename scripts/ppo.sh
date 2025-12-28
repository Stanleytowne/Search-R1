#!/bin/bash
set -e

export WANDB_API_KEY=
export CUDA_VISIBLE_DEVICES=0,1,2,3
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

WANDB_PROJECT=toolbench_ppo
EXPERIMENT_NAME=${CATEGORY}

ACTOR_LEARNING_RATE=1e-6
CRITIC_LEARNING_RATE=1e-5
ACTOR_LR_WARMUP_RATIO=0
CRITIC_LR_WARMUP_RATIO=0.015
EPOCHS=10
TOTAL_TRAINING_STEPS=100
SAVE_FREQ=15

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    --config-name=grpo_toolbench_trainer \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    data.val_batch_size=2 \
    algorithm.adv_estimator=gae \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr="$ACTOR_LEARNING_RATE" \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio="$ACTOR_LR_WARMUP_RATIO" \
    actor_rollout_ref.actor.state_masking=true \
    actor_rollout_ref.rollout.n_agent=1 \
    critic.optim.lr="$CRITIC_LEARNING_RATE" \
    critic.optim.lr_warmup_steps_ratio="$CRITIC_LR_WARMUP_RATIO" \
    critic.model.path=$MODEL_PATH \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console', 'wandb'] \
    +trainer.val_before_train=true \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=10 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=$EPOCHS \
    trainer.total_training_steps=$TOTAL_TRAINING_STEPS \
    trainer.default_local_dir=checkpoints/$CATEGORY/stage-ppo \
    toolbench_url="$TOOLBENCH_URL" \
    reward_model.reward_server_url="$REWARD_SERVER_URL"
