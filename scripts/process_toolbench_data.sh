echo "[1] Collecting all APIs information and organize them into categories"

python toolbench_data_process/get_all_api_infos.py

echo "[2] Generating data for knowledge injection"

python toolbench_data_process/gen_stage1_data.py

echo "[3] Generating data for rl format warm-starting"

python toolbench_data_process/gen_stage2_data.py \
    --input ../StableToolBench/data/answer/G1_answer \
    --mapping ../StableToolBench/data/instruction/G1_query.json \
    --output ./data/toolbench_stage2

python toolbench_data_process/gen_stage2_data.py \
    --input ../StableToolBench/data/answer/G2_answer \
    --mapping ../StableToolBench/data/instruction/G2_query.json \
    --output ./data/toolbench_stage2

echo "[4] Organizing instruction data for rl training"

python toolbench_data_process/organize_instruction.py

echo "[5] Generating rl data for category Sports"

python toolbench_data_process/gen_rl_data.py \
    --input ./data/toolbench_instruction/Email.json \
    --output ./data/toolbench_rl/Email.parquet