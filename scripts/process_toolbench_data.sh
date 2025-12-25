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

echo "[4] Generating rl data"

python toolbench_data_process/organize_instruction.py \
    --input_dir ../StableToolBench/data/instruction \
    --output_dir ./data/toolbench_instruction \
    --files G1_query.json G2_query.json

python toolbench_data_process/gen_rl_data.py \
    --input_dir ./data/toolbench_instruction \
    --output_dir ./data/toolbench_rl

echo "[5] Generating test data"
python toolbench_data_process/organize_instruction.py \
    --input_dir ../StableToolBench/solvable_queries/test_instruction \
    --output_dir ./data/toolbench_test_instruction \
    --files G1_category.json G1_instruction.json G1_tool.json G2_category.json G2_instruction.json

python toolbench_data_process/gen_test_data.py \
    --input_dir ./data/toolbench_test_instruction \
    --output_dir ./data/toolbench_test
