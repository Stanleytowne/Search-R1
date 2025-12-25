import json
import os
import glob
import pandas as pd
import re
import argparse

from system_prompt import SYSTEM_PROMPT

def process_data_to_parquet(json_folder_path, mapping_file_path, output_folder):
    """
    将指定文件夹下的JSON数据根据category分类并保存为parquet文件
    """
    
    # ---------------------------------------------------------
    # 第一步：构建 ID 到 Category 的映射表
    # ---------------------------------------------------------
    print(f"正在读取映射文件: {mapping_file_path} ...")
    id_to_category = {}
    
    try:
        with open(mapping_file_path, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)
            
        for item in mapping_data:
            q_id = item.get('query_id')
            api_list = item.get('api_list', [])
            
            # 取出第一个 api 的 category_name，如果没有则标记为 Uncategorized
            if api_list and len(api_list) > 0:
                category = api_list[0].get('category_name', 'Uncategorized')
            else:
                category = 'Uncategorized'
            
            # 确保 ID 是 int 类型 (假设文件名里的ID能转成int)
            id_to_category[q_id] = category
            
        print(f"映射表构建完成，共包含 {len(id_to_category)} 个 ID。")
        
    except Exception as e:
        print(f"读取映射文件失败: {e}")
        return

    # ---------------------------------------------------------
    # 第二步：遍历数据文件夹并提取数据
    # ---------------------------------------------------------
    # 匹配文件名格式: [id]_ChatGPT_DFS_woFilter_w2.json
    file_pattern = os.path.join(json_folder_path, "*_ChatGPT_DFS_woFilter_w2.json")
    files = glob.glob(file_pattern)
    
    print(f"找到 {len(files)} 个数据文件，开始处理...")
    
    # 用于临时存储数据： { "Logistics": [ {row1}, {row2} ], "Finance": [...] }
    data_buffer = {}

    for file_path in files:
        filename = os.path.basename(file_path)
        
        # 1. 从文件名提取 ID
        # 假设文件名开头就是 ID，例如 "100_ChatGPT_..." -> 提取 "100"
        try:
            file_id_str = filename.split('_')[0]
            file_id = int(file_id_str)
        except ValueError:
            print(f"跳过文件 {filename}: 无法从文件名提取整数 ID")
            continue

        # 2. 获取 Category
        category = id_to_category.get(file_id, "Unknown_Category")
        
        # 3. 读取并解析 JSON 内容
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
            
            # 定位 answer_generation 块
            ans_gen = content.get('answer_generation', {})
            
            # 获取 Prompt (Query)
            prompt = ans_gen.get('query', '')
            if type(prompt) == list:
                prompt = prompt[0]
            
            # 获取 System Prompt 和 Response
            # 规则：train_messages 中的第一条 list (message chain)
            train_msgs_groups = ans_gen.get('train_messages', [])
            
            if not train_msgs_groups:
                continue # 数据不完整，跳过

            first_msg_chain = train_msgs_groups[0] # 取第一组对话
            
            assistant_response = ""
            valid = True
            
            # 遍历对话链找到 system 和第一个 assistant
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
                    'system': SYSTEM_PROMPT,
                    'prompt': prompt,
                    'response': assistant_response,
                    'source_id': file_id
                }
                
                if category not in data_buffer:
                    data_buffer[category] = []
                
                data_buffer[category].append(row)

        except Exception as e:
            print(f"处理文件 {filename} 时出错: {e}")

    # ---------------------------------------------------------
    # 第三步：保存为 Parquet 文件
    # ---------------------------------------------------------
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print("开始写入 Parquet 文件...")

    for category, rows in data_buffer.items():
        # 清理 category 名称，使其适合作为文件名 (去除空格和非法字符)
        safe_filename = re.sub(r'[\\/*?:"<>| ]', '_', category)
        output_path = os.path.join(output_folder, f"{safe_filename}.parquet")

        df = pd.DataFrame(rows)

        # 如果 parquet 文件已经存在，则读取旧数据，与新数据合并后保存
        if os.path.exists(output_path):
            try:
                old_df = pd.read_parquet(output_path)
                df = pd.concat([old_df, df], ignore_index=True)
            except Exception as e:
                print(f"读取已存在的 Parquet 文件 {output_path} 时出错: {e}")
                print("将只写入新的数据。")

        try:
            df.to_parquet(output_path, index=False)
        except Exception as e:
            print(f"写入文件 {output_path} 时出错: {e}")
            prompts = df['prompt'].to_list()
            for idx, prompt in enumerate(prompts):
                if type(prompt) != str:
                    print(f"第 {idx} 条数据的 prompt 不是字符串: {prompt}")
                    print(f"source idx: {df.iloc[idx].get('source_id', '未知')}")
            breakpoint()
        print(f"-> 已保存: {output_path} (包含 {len(df)} 条数据)")

    print("全部处理完成。")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--input", type=str, default="../StableToolBench/data/answer/G1_answer",
                       help="Input answer file path")
    parser.add_argument("--mapping", type=str, default="../StableToolBench/data/instruction/G1_query.json",
                       help="Input mapping file path")
    parser.add_argument("--output", type=str, default="./data/toolbench_stage2",
                       help="Output file path")
    args = parser.parse_args()

    process_data_to_parquet(args.input, args.mapping, args.output)