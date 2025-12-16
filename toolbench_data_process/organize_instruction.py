import json
import os
import re
from collections import defaultdict

def sanitize_filename(name):
    """
    清洗文件名，去掉非法字符（如 / \ : * ? " < > |），将空格转为下划线
    """
    if not name:
        return "unknown"
    # 替换非法字符为空
    name = re.sub(r'[\\/*?:"<>|]', '', name)
    # 替换空格为下划线
    name = name.replace(" ", "_")
    return name

def load_json(filepath):
    if not os.path.exists(filepath):
        print(f"Warning: 文件不存在 {filepath}，跳过。")
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_grouped_data(grouped_data, base_folder):
    """
    将字典形式的分组数据保存为单独的 JSON 文件
    grouped_data: { 'category_name': [list of items], ... }
    """
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    
    print(f"正在保存数据到: {base_folder} ...")
    
    count = 0
    for key, items in grouped_data.items():
        safe_name = sanitize_filename(key)
        file_path = os.path.join(base_folder, f"{safe_name}.json")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(items, f, indent=2, ensure_ascii=False)
        count += 1
    
    print(f"完成！共生成 {count} 个分类文件。")

def process_files(files):
    """
    处理 G2 系列文件：
    1. 按 Category Name 分类 (G2 是类内多工具，所有工具属于同一类)
    """
    by_category = defaultdict(list)
    
    all_items = []
    for file_name in files:
        path = os.path.join(DATA_DIR, file_name)
        data = load_json(path)
        all_items.extend(data)
        
    print(f"\nG2 总数据量: {len(all_items)} 条")

    for item in all_items:
        api_list = item.get('api_list', [])
        
        if not api_list:
            continue
            
        # G2 既然是 Intra-Category，所有工具理论上属于同一类
        # 我们取第一个有 category_name 的作为代表
        category_name = None
        for tool in api_list:
            if tool.get('category_name'):
                category_name = tool.get('category_name')
                break
        
        if category_name:
            by_category[category_name].append(item)

    # 保存
    save_grouped_data(by_category, OUTPUT_DIR)

def main():
    # 定义要处理的文件列表
    files = ['G1_query.json', 'G2_query.json']
    
    # 确保输出目录存在
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    process_files(files)

    print(f"\n全部完成！请检查文件夹: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--input", type=str, default="../StableToolBench/data/instruction",
                       help="Input JSON file path")
    parser.add_argument("--output", type=str, default="./data/toolbench_instruction",
                       help="Output file path")
    args = parser.parse_args()

    DATA_DIR = args.input
    OUTPUT_DIR = args.output

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    process_files(files)

    print(f"\n全部完成！请检查文件夹: {os.path.abspath(OUTPUT_DIR)}")

    main()