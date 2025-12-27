#!/bin/bash
set -e
output_name=${1}

for category in Mapping Media Movies Search; do
    echo "Starting eval for $category"
    save_path=checkpoints/$category/stage2
    latest_step=$(ls "$save_path" | grep "global_step_" | sed 's/global_step_//' | sort -n | tail -1)
    latest_checkpoint="$save_path/global_step_$latest_step"
    echo "evaluating stage2-ed model at $latest_checkpoint"
    python eval.py --model_path $latest_checkpoint --category $category --output_name $output_name
    echo "evaluating inherited model"
    python eval.py --model_path checkpoints/$category/inherited --category $category --output_name $output_name
done