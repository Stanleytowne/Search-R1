import json
import os
import re
import argparse
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

def process_files(files, output_dir):
    by_category = defaultdict(list)
    
    all_items = []
    for file in files:
        data = load_json(file)
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
    save_grouped_data(by_category, output_dir)

def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="../StableToolBench/data/instruction",
                       help="Input directory")
    parser.add_argument("--output_dir", type=str, default="./data/toolbench_instruction",
                       help="Output directory")
    parser.add_argument("--files", type=str, nargs='+', default=['G1_query.json', 'G2_query.json'],
                       help="Files to process")
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    files = []
    for file in args.files:
        files.append(os.path.join(args.input_dir, file))
    process_files(files, args.output_dir)

    print(f"\n全部完成！请检查文件夹: {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    main()