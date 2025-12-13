#!/usr/bin/env python
"""
将StableToolBench的数据格式转换为verl训练所需的parquet格式

输入格式：StableToolBench JSON文件
- 每个样本包含id和conversations数组
- conversations包含system, user, assistant, function消息

输出格式：parquet文件
- prompt: chat格式（列表字典，每个元素有from和value）
- data_source: 数据来源标识
- reward_model: reward相关信息（可选）
- extra_info: 额外信息（包含index等）
"""

import json
import pandas as pd
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import os
import random

SYSTEM_PROMPT_TEMPLATE = """You are AutoGPT, you can use many tools(functions) to do the following task.

First you will be given the task description, and then your task starts.
At each step, you need to give your thought to analyze the status now and what to do next, with a function call to actually excute your step. Your output should follow this format:

Thought:
Action
Action Input:

After the call, you will get the call result, and you are now in a new state.
Then you will analyze your status now, then decide what to do next...
After several (Thought-call) pairs, you finally perform the task, then you can give your finial answer.
Remember: 
1.the state change is irreversible, you can't go back to one of the former state, if you want to restart the task, say "I give up and restart".
2.All the thought is short, at most in 5 sentence.
3.You can do more then one trys, so if your plan is to continusly try some conditions, you can do one of the conditions per try.

Let's Begin!

Task description: You should use functions to help handle the real time user querys. Remember:
1.ALWAYS call "Finish" function at the end of the task. And the final answer should contain enough information to show to the user,If you can't handle the task, or you find that function calls always fail(the function is not valid now), use function Finish->give_up_and_restart.
2.Do not use origin tool names, use only subfunctions' names.

You have access of the following tools:
{tool_descriptions}

Specifically, you have access to the following APIs: {api_list}"""


def convert_api_list_to_system_format(api_list: List[Dict]) -> tuple[str, List[Dict]]:
    """
    将G1_query.json格式的api_list转换为system message中需要的格式
    
    Args:
        api_list: G1_query.json格式的API列表，每个元素包含：
            - category_name: category名称
            - tool_name: 工具名称
            - api_name: API名称
            - api_description: API描述
            - required_parameters: 必需参数
            - optional_parameters: 可选参数
            - method: HTTP方法
    
    Returns:
        (tool_descriptions, api_list_formatted): 
        - tool_descriptions: 工具描述字符串
        - api_list_formatted: 格式化后的API列表（用于JSON序列化）
    """
    # 收集工具描述
    tool_descriptions = []
    seen_tools = set()
    
    # 格式化API列表
    api_list_formatted = []
    
    for api in api_list:
        tool_name = api.get('tool_name', '')
        api_name = api.get('api_name', '')
        category_name = api.get('category_name', '')
        api_description = api.get('api_description', '').strip()
        
        # 构建API名称（格式：api_name_for_tool_name）
        formatted_api_name = f"{api_name}_for_{tool_name.lower().replace(' ', '_')}"
        
        # 构建参数信息
        required_params = api.get('required_parameters', [])
        optional_params = api.get('optional_parameters', [])
        
        # 如果参数是字符串，尝试解析为列表
        if isinstance(required_params, str):
            try:
                required_params = json.loads(required_params) if required_params else []
            except:
                required_params = []
        if isinstance(optional_params, str):
            try:
                optional_params = json.loads(optional_params) if optional_params else []
            except:
                optional_params = []
        
        # 构建parameters字典
        properties = {}
        required = []
        
        for param in required_params:
            if isinstance(param, dict):
                param_name = param.get('name', '')
                if param_name:
                    properties[param_name] = {
                        'type': param.get('type', 'string'),
                        'description': param.get('description', '')
                    }
                    required.append(param_name)
        
        for param in optional_params:
            if isinstance(param, dict):
                param_name = param.get('name', '')
                if param_name and param_name not in properties:
                    properties[param_name] = {
                        'type': param.get('type', 'string'),
                        'description': param.get('description', '')
                    }
        
        parameters = {
            'type': 'object',
            'properties': properties,
            'required': required,
            'optional': []
        }
        
        # 构建API信息
        api_info = {
            'name': formatted_api_name,
            'description': f'This is the subfunction for tool "{tool_name}", you can use this tool. The description of this function is: "{api_description}"',
            'parameters': parameters
        }
        api_list_formatted.append(api_info)
        
        # 收集工具描述
        if tool_name and tool_name not in seen_tools:
            seen_tools.add(tool_name)
            # 这里可以添加工具描述，如果没有可以从其他地方获取
            tool_descriptions.append(f"{tool_name}: {category_name} tool")
    
    # 添加Finish函数
    finish_api = {
        'name': 'Finish',
        'description': 'If you believe that you have obtained a result that can answer the task, please call this function to provide the final answer. Alternatively, if you recognize that you are unable to proceed with the task in the current state, call this function to restart. Remember: you must ALWAYS call this function at the end of your attempt, and the only part that will be shown to the user is the final answer, so it should contain sufficient information.',
        'parameters': {
            'type': 'object',
            'properties': {
                'return_type': {
                    'type': 'string',
                    'enum': ['give_answer', 'give_up_and_restart']
                },
                'final_answer': {
                    'type': 'string',
                    'description': 'The final answer you want to give the user. You should have this field if "return_type"=="give_answer"'
                }
            },
            'required': ['return_type'],
            'optional': []
        }
    }
    api_list_formatted.append(finish_api)
    
    tool_descriptions_str = '\n'.join([f"{i+1}.{desc}" for i, desc in enumerate(tool_descriptions)])
    
    return tool_descriptions_str, api_list_formatted


def convert_g1_query_to_conversations(sample: Dict) -> List[Dict]:
    """
    将G1_query.json格式的样本转换为conversations格式
    
    Args:
        sample: G1_query.json格式的样本，包含：
            - api_list: API列表
            - query: 用户查询
    
    Returns:
        conversations: 对话列表，格式类似toolllama_G123_dfs_eval.json
    """
    api_list = sample.get('api_list', [])
    query = sample.get('query', '')
    
    # 转换API列表格式
    tool_descriptions, api_list_formatted = convert_api_list_to_system_format(api_list)
    
    # 构建system message
    api_list_json = json.dumps(api_list_formatted, ensure_ascii=False)
    system_message = SYSTEM_PROMPT_TEMPLATE.format(
        tool_descriptions=tool_descriptions,
        api_list=api_list_json
    )
    
    # 构建conversations
    conversations = [
        {
            "from": "system",
            "value": system_message
        },
        {
            "from": "user",
            "value": f"\n{query}\nBegin!\n"
        }
    ]
    
    return conversations


def convert_conversations_to_chat(conversations: List[Dict]) -> List[Dict]:
    """
    将StableToolBench的conversations转换为chat格式
    
    StableToolBench格式：
    - {"from": "system", "value": "..."}
    - {"from": "user", "value": "..."}
    - {"from": "assistant", "value": "..."}
    - {"from": "function", "value": "..."}
    
    verl需要的格式：
    - 相同的格式，但需要确保格式正确
    """
    chat = []
    for conv in conversations:
        role = conv.get("from", "")
        value = conv.get("value", "")
        
        # 确保role和value都存在
        if role and value:
            chat.append({
                "role": role,  # verl可能使用role而不是from
                "from": role,  # 保留from以兼容
                "content": value  # 某些tokenizer使用content
            })
    
    return chat


def extract_category_from_conversations(conversations: List[Dict], tools_folder: str = None) -> str:
    """
    从conversations的system message中提取category信息
    
    方法：
    1. 从system message中解析API列表（格式：Specifically, you have access to the following APIs: [...]）
    2. 从API名称中提取tool_name（格式：api_name_for_tool_name）
    3. 根据tool_name在tools目录中查找对应的category
    
    Args:
        conversations: 对话列表
        tools_folder: tools目录路径（可选，用于查找category）
    
    Returns:
        category名称，如果找不到则返回默认值
    """
    import re
    import os
    import json
    
    # 默认category
    default_category = "G1_category"
    
    # 查找system message
    system_msg = None
    for conv in conversations:
        if conv.get("from") == "system" or conv.get("role") == "system":
            system_msg = conv.get("value", "") or conv.get("content", "")
            break
    
    if not system_msg:
        return default_category
    
    # 尝试从system message中提取API列表
    try:
        # 查找API列表：Specifically, you have access to the following APIs: [...]
        api_match = re.search(r'Specifically, you have access to the following APIs:\s*(\[.*?\])', system_msg, re.DOTALL)
        if api_match:
            api_list_str = api_match.group(1)
            api_list = json.loads(api_list_str)
            
            if len(api_list) > 0:
                # 从第一个API中提取tool_name（跳过Finish函数）
                for api_info in api_list:
                    api_name = api_info.get('name', '')
                    if api_name == 'Finish':
                        continue
                    if '_for_' in api_name:
                        tool_name = api_name.rsplit('_for_', 1)[-1]
                        
                        # 如果提供了tools_folder，尝试查找category
                        if tools_folder and os.path.exists(tools_folder):
                            # 遍历所有category目录，查找包含该tool_name的JSON文件
                            for category_dir in os.listdir(tools_folder):
                                category_path = os.path.join(tools_folder, category_dir)
                                if os.path.isdir(category_path):
                                    # 查找tool_name.json文件
                                    tool_file = os.path.join(category_path, f"{tool_name}.json")
                                    if os.path.exists(tool_file):
                                        return category_dir  # 返回找到的category
                            
                            # 如果找不到精确匹配，尝试标准化tool_name
                            # 注意：这里不能直接import utils，因为可能不在同一路径
                            # 使用简单的标准化：转小写，替换空格为下划线
                            standardized_tool_name = tool_name.lower().replace(' ', '_')
                            for category_dir in os.listdir(tools_folder):
                                category_path = os.path.join(tools_folder, category_dir)
                                if os.path.isdir(category_path):
                                    for json_file in os.listdir(category_path):
                                        if json_file.endswith('.json'):
                                            file_base = os.path.splitext(json_file)[0]
                                            if file_base.lower().replace(' ', '_') == standardized_tool_name:
                                                return category_dir
                        # 如果找不到，返回默认值
                        break
    except Exception as e:
        print(f"Warning: Failed to extract category from conversations: {e}")
    
    return default_category


def process_toolbench_json(input_file: str = None, output_file: str = None, 
                          max_samples: int = None, data: List[Dict] = None):
    """
    处理StableToolBench JSON文件并转换为parquet格式
    
    Args:
        input_file: 输入的JSON文件路径（如果data为None）
        output_file: 输出的parquet文件路径
        max_samples: 最大处理样本数（用于测试）
        data: 直接提供的数据（如果input_file为None）
    """
    if data is None:
        if input_file is None:
            raise ValueError("Either input_file or data must be provided")
        print(f"Reading {input_file}...")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        print(f"Processing provided data...")
    
    print(f"Found {len(data)} samples")
    
    # 限制样本数（用于测试）
    if max_samples and max_samples > 0:
        data = data[:max_samples]
        print(f"Processing first {len(data)} samples")
    
    records = []
    
    for idx, sample in enumerate(data):
        # 检查数据格式：如果是G1_query.json格式（有api_list和query），需要转换
        if 'api_list' in sample and 'query' in sample:
            # G1_query.json格式：需要构造conversations
            sample_id = sample.get("query_id", f"sample_{idx}")
            conversations = convert_g1_query_to_conversations(sample)
            
            # 从api_list中提取category（取第一个API的category_name）
            category = "G1_category"  # 默认值
            api_list = sample.get('api_list', [])
            if api_list and len(api_list) > 0:
                category = api_list[0].get('category_name', 'G1_category')
        else:
            # toolllama_G123_dfs_eval.json格式：已经有conversations
            sample_id = sample.get("id", f"sample_{idx}")
            conversations = sample.get("conversations", [])
            
            if not conversations:
                print(f"Warning: Sample {idx} has no conversations, skipping")
                continue
            
            # 提取category（用于ToolBench API调用）
            # 尝试从conversations的system message中提取category
            tools_folder = os.environ.get('TOOLS_FOLDER', None)
            if not tools_folder:
                # 尝试从相对路径查找
                possible_paths = [
                    '../../StableToolBench/data/toolenv/tools',
                    '../StableToolBench/data/toolenv/tools',
                    'StableToolBench/data/toolenv/tools'
                ]
                for path in possible_paths:
                    abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), path))
                    if os.path.exists(abs_path):
                        tools_folder = abs_path
                        break
            
            category = extract_category_from_conversations(conversations, tools_folder)
        
        # 转换为chat格式
        chat = convert_conversations_to_chat(conversations)
        
        if not chat:
            print(f"Warning: Sample {idx} has empty chat, skipping")
            continue
        
        # 构建记录
        record = {
            "prompt": chat,  # verl需要的prompt列
            "data_source": "toolbench",  # 数据来源标识
            "reward_model": {  # reward相关信息
                "style": "function",  # 使用function-based reward
                "ground_truth": None  # ToolBench不需要ground truth
            },
            "extra_info": {
                "index": idx,
                "sample_id": sample_id,
                "category": category
            }
        }
        
        records.append(record)
        
        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1} samples...")
    
    # 创建DataFrame
    df = pd.DataFrame(records)
    
    print(f"\nTotal records: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    # 保存为parquet
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving to {output_file}...")
    df.to_parquet(output_file, index=False, engine='pyarrow')
    
    print(f"✓ Successfully converted {len(df)} samples to {output_file}")
    
    # 打印示例
    if len(df) > 0:
        print("\nExample record:")
        example = df.iloc[0]
        print(f"  Sample ID: {example['extra_info']['sample_id']}")
        print(f"  Chat length: {len(example['prompt'])} messages")
        print(f"  First message role: {example['prompt'][0].get('role', example['prompt'][0].get('from', 'unknown'))}")


def split_train_val(input_file: str, train_output: str, val_output: str, 
                    train_ratio: float = 0.9, max_samples: int = None):
    """
    将数据分割为训练集和验证集
    
    Args:
        input_file: 输入的JSON文件路径
        train_output: 训练集输出路径
        val_output: 验证集输出路径
        train_ratio: 训练集比例
        max_samples: 最大处理样本数
    """
    print(f"Reading {input_file}...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Found {len(data)} samples")
    
    # 限制样本数
    if max_samples and max_samples > 0:
        data = data[:max_samples]
        print(f"Processing first {len(data)} samples")
    
    # 随机打乱（可选，但推荐）
    random.seed(42)
    random.shuffle(data)
    
    # 分割数据
    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    print(f"\nSplitting: {len(train_data)} train, {len(val_data)} val")
    
    # 处理训练集
    print("\nProcessing training set...")
    process_toolbench_json(
        input_file=None,
        output_file=train_output,
        max_samples=None,
        data=train_data
    )
    
    # 处理验证集
    print("\nProcessing validation set...")
    process_toolbench_json(
        input_file=None,
        output_file=val_output,
        max_samples=None,
        data=val_data
    )


def main():
    parser = argparse.ArgumentParser(description="Convert StableToolBench data to verl format")
    parser.add_argument("--input", type=str, default="../StableToolBench/data/instruction/G1_query.json",
                       help="Input JSON file path")
    parser.add_argument("--output", type=str,
                       help="Output parquet file path")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to process (for testing)")
    parser.add_argument("--split", action="store_true",
                       help="Split into train/val sets")
    parser.add_argument("--train_ratio", type=float, default=0.9,
                       help="Ratio of training data (when using --split)")
    parser.add_argument("--train_output", type=str, default=None,
                       help="Training set output path (when using --split)")
    parser.add_argument("--val_output", type=str, default=None,
                       help="Validation set output path (when using --split)")
    
    args = parser.parse_args()
    
    if args.split:
        # 分割模式
        train_output = args.train_output or args.output.replace(".parquet", "_train.parquet")
        val_output = args.val_output or args.output.replace(".parquet", "_val.parquet")
        
        split_train_val(
            input_file=args.input,
            train_output=train_output,
            val_output=val_output,
            train_ratio=args.train_ratio,
            max_samples=args.max_samples
        )
    else:
        # 单文件模式
        process_toolbench_json(
            input_file=args.input,
            output_file=args.output,
            max_samples=args.max_samples
        )


if __name__ == "__main__":
    main()
