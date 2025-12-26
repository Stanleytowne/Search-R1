import os
import glob
import pandas as pd
import json
from collections import defaultdict

# 定义要扫描的文件夹和文件类型
folders_config = [
    ("data/api_infos", "*.json"),
    ("data/toolbench_stage1", "*.parquet"),
    ("data/toolbench_stage2", "*.parquet"),
    ("data/toolbench_rl", "*.parquet"),
    ("data/toolbench_test", "*.parquet"),
]

# 存储每个文件名（不含扩展名）对应的文件信息
file_groups = defaultdict(list)

# 收集所有文件并统计条目数
for folder, pattern in folders_config:
    files = glob.glob(os.path.join(folder, pattern))
    for file_path in files:
        # 获取文件名（不含扩展名）
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # 根据文件类型读取数据并统计条目数
        try:
            if file_path.endswith('.json'):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                row_count = len(data) if isinstance(data, list) else 1
            elif file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path)
                row_count = len(df)
            else:
                row_count = 0
            
            file_groups[base_name].append({
                'path': file_path,
                'count': row_count
            })
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

# 按文件名分组统计并打印结果
# 收集所有路径（文件夹）
all_folders = [folder for folder, _ in folders_config]

# 构建表格数据：每个文件名在各个路径下的条目数
table_data = {}
for base_name, files_info in file_groups.items():
    table_data[base_name] = {}
    # 初始化所有路径的计数为0
    for folder in all_folders:
        table_data[base_name][folder] = 0
    # 统计每个路径下的条目数
    for file_info in files_info:
        file_path = file_info['path']
        count = file_info['count']
        # 找到文件所属的路径
        for folder in all_folders:
            if file_path.startswith(folder):
                table_data[base_name][folder] += count
                break

# 生成表格
print("# 按文件名（不含扩展名）分组统计数据条目\n")
print("| 文件名 | " + " | ".join(all_folders) + " | 总计 |")
print("|--------|" + "|".join(["--------" for _ in all_folders]) + "|--------|")

for base_name in sorted(table_data.keys()):
    row_data = table_data[base_name]
    folder_counts = [str(row_data[folder]) for folder in all_folders]
    total = sum(row_data.values())
    print(f"| {base_name} | " + " | ".join(folder_counts) + f" | {total} |")

print(f"\n**总计**: {len(table_data)} 个不同的文件名组")