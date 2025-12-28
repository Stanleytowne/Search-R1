category=(Mapping Media Movies Search)
save_name=gpt4_1_steps40_merged
grpo_model=checkpoints/Email/stage-rl/actor/global_step_40
base_model=checkpoints/Email/stage2/global_step_33


target_models=()
save_dirs=()

for category in ${category[@]}; do
    base_path=checkpoints/$category/stage2
    latest_step=$(ls "$base_path" | grep "global_step_" | sed 's/global_step_//' | sort -n | tail -1)
    target_models+=("$base_path/global_step_$latest_step")
    save_dirs+=("checkpoints/$category/$save_name")
done

python inherit_weight.py --grpo_model $grpo_model --base_model $base_model --target_models ${target_models[@]} --save_dirs ${save_dirs[@]}