import json
import os
import re
import argparse
from collections import defaultdict

def sanitize_filename(name):
    """
    sanitize the filename, remove illegal characters (e.g. / \ : * ? " < > |) and replace spaces with underscores
    """
    if not name:
        return "unknown"
    # replace illegal characters with empty
    name = re.sub(r'[\\/*?:"<>|]', '', name)
    # replace spaces with underscores
    name = name.replace(" ", "_")
    return name

def load_json(filepath):
    if not os.path.exists(filepath):
        print(f"Warning: file not found {filepath}, skip.")
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_grouped_data(grouped_data, base_folder):
    """
    save the grouped data in dictionary format to separate JSON files
    grouped_data: { 'category_name': [list of items], ... }
    """
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    
    print(f"saving data to: {base_folder} ...")
    
    count = 0
    for key, items in grouped_data.items():
        safe_name = sanitize_filename(key)
        file_path = os.path.join(base_folder, f"{safe_name}.json")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(items, f, indent=2, ensure_ascii=False)
        count += 1
    
    print(f"done! {count} category files generated.")

def process_files(files, output_dir):
    by_category = defaultdict(list)
    
    all_items = []
    for file in files:
        data = load_json(file)
        all_items.extend(data)
        
    print(f"\ntotal number of data: {len(all_items)}")

    for item in all_items:
        api_list = item.get('api_list', [])
        
        if not api_list:
            continue
            
        # since G2 is Intra-Category, all tools should belong to the same category
        # we take the first tool with category_name as the representative
        category_name = None
        for tool in api_list:
            if tool.get('category_name'):
                category_name = tool.get('category_name')
                break
        
        if category_name:
            by_category[category_name].append(item)

    # save
    save_grouped_data(by_category, output_dir)

def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="data-toolbench/instruction",
                       help="Input directory")
    parser.add_argument("--output_dir", type=str, default="data/toolbench_instruction",
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

    print(f"\nall done. please check the folder: {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    main()