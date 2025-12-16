python toolbench_data_process/get_all_api_infos.py \
    --input ../StableToolBench/data/instruction/G1_query.json

python toolbench_data_process/get_all_api_infos.py \
    --input ../StableToolBench/data/instruction/G2_query.json

python toolbench_data_process/gen_stage1_data.py

python toolbench_data_process/gen_stage2_data.py \
    --input ../StableToolBench/data/answer/G1_answer \
    --mapping ../StableToolBench/data/instruction/G1_query.json \
    --output ./data/toolbench_stage2

python toolbench_data_process/gen_stage2_data.py \
    --input ../StableToolBench/data/answer/G2_answer \
    --mapping ../StableToolBench/data/instruction/G2_query.json \
    --output ./data/toolbench_stage2

python toolbench_data_process/organize_instruction.py \
    --input ../StableToolBench/data/instruction \
    --output ./data/toolbench_instruction

python toolbench_data_process/gen_rl_data.py \
    --input ./data/toolbench_instruction/Sports.json \
    --output ./data/toolbench_rl/Sports.parquet \
    --split