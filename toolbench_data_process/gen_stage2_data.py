import json
import os
import glob
import pandas as pd
import re
import argparse

from system_prompt import SYSTEM_PROMPT

def process_data_to_parquet(json_folder_path, mapping_file_path, api_infos_folder_path, output_folder):
    # ---------------------------------------------------------
    # step 1: build id to category mapping
    # ---------------------------------------------------------
    print(f"reading mapping file: {mapping_file_path} ...")
    id_to_category = {}
    
    with open(mapping_file_path, 'r', encoding='utf-8') as f:
        mapping_data = json.load(f)
        
    for item in mapping_data:
        q_id = item.get('query_id')
        api_list = item.get('api_list', [])
        
        if api_list and len(api_list) > 0:
            category = api_list[0].get('category_name', 'Uncategorized')
        else:
            category = 'Uncategorized'
        
        id_to_category[q_id] = category
        
    print(f"mapping table built, containing {len(id_to_category)} ids.")

    # ---------------------------------------------------------
    # step 2: iterate over the api infos folder and extract data
    # ---------------------------------------------------------
    api_names_in_category = {}
    api_infos_files = glob.glob(os.path.join(api_infos_folder_path, "*.json"))
    for api_info_file in api_infos_files:
        category = os.path.basename(api_info_file).split('.')[0]
        with open(api_info_file, 'r', encoding='utf-8') as f:
            api_infos = json.load(f)
        for api_name, _ in api_infos.items():
            if category not in api_names_in_category:
                api_names_in_category[category] = []
            api_names_in_category[category].append(api_name)

    api_names_in_category = {category: ", ".join(api_names) for category, api_names in api_names_in_category.items()}

    # ---------------------------------------------------------
    # step 3: iterate over the data folder and extract data
    # ---------------------------------------------------------
    # file name pattern: [id]_ChatGPT_DFS_woFilter_w2.json
    file_pattern = os.path.join(json_folder_path, "*_ChatGPT_DFS_woFilter_w2.json")
    files = glob.glob(file_pattern)
    
    print(f"found {len(files)} data files, starting to process...")
    
    # temporary buffer for storing data: { "Logistics": [ {row1}, {row2} ], "Finance": [...] }
    data_buffer = {}

    for file_path in files:
        filename = os.path.basename(file_path)
        
        # 1. extract id from file name
        # assume the file name starts with id, e.g. "100_ChatGPT_..." -> extract "100"
        file_id_str = filename.split('_')[0]
        file_id = int(file_id_str)

        # 2. get category
        category = id_to_category.get(file_id, "Unknown_Category")
        
        # 3. read and parse json content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = json.load(f)
        
        # locate answer_generation block
        ans_gen = content.get('answer_generation', {})
        
        # get prompt (query)
        prompt = ans_gen.get('query', '')
        if type(prompt) == list:
            prompt = prompt[0]
        
        # get system prompt and response
        # rule: the first list in train_messages (message chain)
        train_msgs_groups = ans_gen.get('train_messages', [])
        
        if not train_msgs_groups:
            continue # data is incomplete, skip

        first_msg_chain = train_msgs_groups[0] # get the first message chain
        
        assistant_response = ""
        valid = True
        
        # iterate over the message chain to find system and the first assistant
        for msg in first_msg_chain:
            role = msg.get('role')
            
            if role == 'assistant':
                thought = msg['content']
                if 'error' in thought.lower():
                    # print(thought)
                    valid = False
                    break

                function_call = msg.get('function_call')
                if function_call:
                    action = function_call['name']
                    action_input = function_call['arguments']
                    try:
                        action_input = json.loads(action_input)
                    except Exception as e:
                        print(f"parsing action input error for query id {file_id}")
                        break
                else:
                    continue
                
                text = f"Thought: {thought}\nAction: {action}\nAction Input: {json.dumps(action_input)}"
                assistant_response = text
            
            if role == 'function' and assistant_response != "":
                function_response = json.loads(msg['content'])
                if function_response['error'] != "":
                    valid = False
                break
        
        if valid:
            row = {
                'system': SYSTEM_PROMPT.replace("{api_names}", api_names_in_category[category]),
                'prompt': prompt,
                'response': assistant_response,
                'source_id': file_id
            }
            
            if category not in data_buffer:
                data_buffer[category] = []
            
            data_buffer[category].append(row)

    # ---------------------------------------------------------
    # step 4: save data to parquet file
    # ---------------------------------------------------------
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print("saving data to parquet files...")

    for category, rows in data_buffer.items():
        # clean category name, make it suitable for file name (remove spaces and illegal characters)
        safe_filename = re.sub(r'[\\/*?:"<>| ]', '_', category)
        output_path = os.path.join(output_folder, f"{safe_filename}.parquet")

        df = pd.DataFrame(rows)

        # if the parquet file already exists, read the old data, merge with new data and save
        if os.path.exists(output_path):
            try:
                old_df = pd.read_parquet(output_path)
                df = pd.concat([old_df, df], ignore_index=True)
            except Exception as e:
                print(f"error reading existing parquet file {output_path}: {e}")
                print("will only write new data.")

        try:
            df.to_parquet(output_path, index=False)
        except Exception as e:
            print(f"error writing file {output_path}: {e}")
            prompts = df['prompt'].to_list()
            for idx, prompt in enumerate(prompts):
                if type(prompt) != str:
                    print(f"the prompt of the {idx}th data is not a string: {prompt}")
                    print(f"source idx: {df.iloc[idx].get('source_id', 'unknown')}")
            breakpoint()
        print(f"-> saved: {output_path} (containing {len(df)} rows)")

    print("all done.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--input", type=str, default="data-toolbench/answer/G1_answer",
                       help="Input answer file path")
    parser.add_argument("--mapping", type=str, default="data-toolbench/instruction/G1_query.json",
                       help="Input mapping file path")
    parser.add_argument("--api_infos", type=str, default="data/api_infos",
                       help="Input api infos file path")
    parser.add_argument("--output", type=str, default="data/toolbench_stage2",
                       help="Output file path")
    args = parser.parse_args()

    process_data_to_parquet(args.input, args.mapping, args.api_infos, args.output)