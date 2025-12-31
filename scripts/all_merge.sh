#!/bin/bash
set -e
set -x

steps=(40 80 120 160)
base_dir=checkpoints/Movies/stage-ppo/actor/global_step_
base_model=checkpoints/Movies/stage2/global_step_90
target_dir=checkpoints/Search/stage2/global_step_108
output_dirs=()
for step in ${steps[@]}; do
    dir=$base_dir$steps
    output_dir=checkpoints/Search/inherited-$step
    output_dirs+=("$output_dir")
    python inherit_weight.py --grpo_model $dir --base_model $base_model --target_dir $target_dir --output_dirs $output_dir
done

for model in ${output_dirs[@]}; do
    echo "Evaluating $model"
    python eval.py --model_path $model --category Search --num_runs 3 --batch_size 4
done