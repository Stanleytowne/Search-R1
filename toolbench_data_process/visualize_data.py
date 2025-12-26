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
print("=" * 80)
print("按文件名（不含扩展名）分组统计数据条目")
print("=" * 80)

for base_name in sorted(file_groups.keys()):
    files_info = file_groups[base_name]
    total_count = sum(f['count'] for f in files_info)
    
    print(f"\n文件名: {base_name}")
    print(f"  总条目数: {total_count}")
    print(f"  文件数量: {len(files_info)}")
    for file_info in files_info:
        print(f"    - {file_info['path']}: {file_info['count']} 条")

print("\n" + "=" * 80)
print(f"总计: {len(file_groups)} 个不同的文件名组")
print("=" * 80)