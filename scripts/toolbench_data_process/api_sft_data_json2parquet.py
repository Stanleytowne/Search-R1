import pandas as pd
import json
import glob
import os

# 配置路径
INPUT_PROCESSED_PATH = "data/api_infos_processed/processed_Sports.json"  # vLLM 生成的 JSON 文件目录
OUTPUT_PARQUET_PATH = "data/api_sft_data/Sports.parquet" # 输出的 Parquet 文件路径

data_rows = []

with open(INPUT_PROCESSED_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

for item in data:
    # 假设我们用 original_json 作为 Prompt，让模型学习生成描述
    # 你可以自定义这里的 Prompt，比如加上 "Explain this API:" 前缀
    api_json_str = json.dumps(item['original_json'], ensure_ascii=False)
    
    prompt_text = ""

    # 我们有 3 个生成的模板，可以把它们变成 3 条独立的训练数据
    # 这样可以增加数据多样性
    for template_key, generated_text in item['generations'].items():
        data_rows.append({
            "prompt": prompt_text,
            "response": generated_text,
            "source": template_key # 可选：记录数据来源
        })

    data_rows.append({
        "prompt": prompt_text,
        "response": api_json_str,
        "source": "original_json"
    })

# 转换为 DataFrame
df = pd.DataFrame(data_rows)

# 确保输出目录存在
os.makedirs(os.path.dirname(OUTPUT_PARQUET_PATH), exist_ok=True)

# 保存为 Parquet
df.to_parquet(OUTPUT_PARQUET_PATH, index=False)

print(f"转换完成！数据已保存至 {OUTPUT_PARQUET_PATH}，共 {len(df)} 条样本。")
print("示例数据：")
print(df.head(1))