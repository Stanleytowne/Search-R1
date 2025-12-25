import os
import glob
import pandas as pd
import json

json_files = glob.glob("data/api_infos/*.json")
for json_file in json_files:
    with open(json_file, 'r') as f:
        data = json.load(f)
    print(f"File: {json_file}, Number of rows: {len(data)}")

parquet_files = glob.glob("data/toolbench_stage1/*.parquet")

for parquet_file in parquet_files:
    df = pd.read_parquet(parquet_file)
    print(f"File: {parquet_file}, Number of rows: {len(df)}")

parquet_files = glob.glob("data/toolbench_stage2/*.parquet")

for parquet_file in parquet_files:
    df = pd.read_parquet(parquet_file)
    print(f"File: {parquet_file}, Number of rows: {len(df)}")

parquet_files = glob.glob("data/toolbench_rl/*.parquet")

for parquet_file in parquet_files:
    df = pd.read_parquet(parquet_file)
    print(f"File: {parquet_file}, Number of rows: {len(df)}")

parquet_files = glob.glob("data/toolbench_test/*.parquet")

for parquet_file in parquet_files:
    df = pd.read_parquet(parquet_file)
    print(f"File: {parquet_file}, Number of rows: {len(df)}")