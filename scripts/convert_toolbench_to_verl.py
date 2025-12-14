#!/usr/bin/env python
"""
Convert StableToolBench G1_query.json format to verl training parquet format.

Input format: G1_query.json
- Each sample contains api_list and query
- api_list contains API information (category_name, tool_name, api_name, etc.)

Output format: parquet file
- prompt: chat format (list of dicts, each with from and value)
- data_source: data source identifier
- reward_model: reward related information
- extra_info: additional information (index, category, api_list for validation)
  Note: api_list in extra_info only contains APIs from this sample, not the entire category
"""

import json
import pandas as pd
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import os
import random

def normalize_api_name(api_name: str) -> str:
    """
    Normalize API name: convert to lowercase and replace spaces with underscores.
    Special case: "Finish" is not normalized (it's a special function name).
    
    Args:
        api_name: Original API name (e.g., "Get Company Data by LinkedIn URL_for_fresh_linkedin_profile_data")
    
    Returns:
        Normalized API name (e.g., "get_company_data_by_linkedin_url_for_fresh_linkedin_profile_data")
    """
    if not api_name:
        return api_name
    # Don't normalize "Finish" - it's a special function name
    if api_name.strip() == "Finish":
        return "Finish"
    return api_name.lower().replace(' ', '_')

SYSTEM_PROMPT_TEMPLATE = """You are an intelligent agent designed to handle real-time user queries using a variety of tools.

First, you will receive a task description. Then, you will enter a loop of reasoning and acting to complete the task.

At each step, follow this process:
1. **Thought**: Analyze the current status and determine the next logical step.
2. **Action**: Select the appropriate tool to execute that step and output the function name directly.
3. **Action Input**: Provide the arguments for the tool as a STRICT valid JSON object.

Output Format:
Thought: <your reasoning>
Action: <function_name>
Action Input: <function_arguments_as_a_valid_JSON_object>

After the action is executed, you will receive the result (Observation). Based on the new state, continue the loop until the task is complete.

Constraints & Rules:
1. **Action Field**: The "Action" output must be the EXACT name of the function. Do NOT include parentheses `()`, words like "call" or "use", or any punctuation.
2. **Finishing**: You MUST call the "Finish" function to submit your final answer. 
3. **Failure**: If you cannot complete the task or verify that a tool is broken after retries, call "Finish" with "return_type" as "give_up".

Available Tools:
{api_list}
"""


def convert_api_list_to_system_format(api_list: List[Dict]) -> tuple[str, List[Dict]]:
    """
    Convert G1_query.json format api_list to system message format.
    
    Args:
        api_list: G1_query.json format API list, each element contains:
            - category_name: category name
            - tool_name: tool name
            - api_name: API name
            - api_description: API description
            - required_parameters: required parameters
            - optional_parameters: optional parameters
            - method: HTTP method
    
    Returns:
        (tool_descriptions, api_list_formatted): 
        - tool_descriptions: tool description string
        - api_list_formatted: formatted API list (for JSON serialization)
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
        
        # Build API name (format: api_name_for_tool_name)
        # Normalize both api_name and tool_name
        normalized_api_name = normalize_api_name(api_name)
        normalized_tool_name = normalize_api_name(tool_name)
        formatted_api_name = f"{normalized_api_name}_for_{normalized_tool_name}"
        
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
        'description': 'If you believe that you have obtained a result that can answer the task, please call this function to provide the final answer. Alternatively, if you recognize that you are unable to proceed with the task in the current state, call this function to give up. Remember: you must ALWAYS call this function at the end of your attempt, and the only part that will be shown to the user is the final answer, so it should contain sufficient information.',
        'parameters': {
            'type': 'object',
            'properties': {
                'return_type': {
                    'type': 'string',
                    'enum': ['give_answer', 'give_up']
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
    Convert G1_query.json format sample to conversations format.
    
    Args:
        sample: G1_query.json format sample, contains:
            - api_list: API list (only APIs for this sample)
            - query: user query
    
    Returns:
        conversations: conversation list, similar to toolllama_G123_dfs_eval.json format
    """
    api_list = sample['api_list']
    query = sample['query']
    
    # 转换API列表格式
    _, api_list_formatted = convert_api_list_to_system_format(api_list)
    
    # 构建system message
    api_list_json = json.dumps(api_list_formatted, ensure_ascii=False)
    system_message = SYSTEM_PROMPT_TEMPLATE.format(
        api_list=api_list_json
    )
    
    # 构建conversations
    conversations = [
        {
            "from": "system",
            "content": system_message
        },
        {
            "from": "user",
            "content": f"\n{query}\n"
        }
    ]
    
    return conversations, api_list_formatted




def process_toolbench_json(input_file: str = None, output_file: str = None, 
                          max_samples: int = None, data: List[Dict] = None):
    """
    Process G1_query.json format and convert to parquet format.
    
    Args:
        input_file: Input G1_query.json file path (if data is None)
        output_file: Output parquet file path
        max_samples: Maximum number of samples to process (for testing)
        data: Directly provided data (if input_file is None)
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
    
    if max_samples and max_samples > 0:
        data = data[:max_samples]
        print(f"Processing first {len(data)} samples")
    
    records = []
    
    for idx, sample in enumerate(data):        
        # Get api_list from this sample only (not the entire category)
        api_list = sample['api_list']
        
        # Construct conversations
        sample_id = sample["query_id"]
        conversations, api_list_formatted = convert_g1_query_to_conversations(sample)
        
        # Extract category from api_list (use first API's category_name)
        category = api_list[0]['category_name']
        
        # Build API validation info in simplified format for parquet storage
        # Format: {"api": "api1,api2,api3", "n_required_param": "0,1,2", "n_optional_param": "0,2,1"}
        api_names = []
        n_required_list = []
        n_optional_list = []
        
        for api in api_list_formatted:
            api_name = api.get('name', '')
            if api_name == 'Finish':
                continue  # Skip Finish function
            
            api_names.append(api_name)
            params = api.get('parameters', {})
            required = params.get('required', [])
            optional = params.get('optional', [])
            n_required_list.append(str(len(required)))
            n_optional_list.append(str(len(optional)))
        
        # Build record
        record = {
            "prompt": conversations,  # verl required prompt column
            "data_source": "toolbench",  # data source identifier
            "reward_model": {  # reward related information
                "style": "function",  # use function-based reward
                "ground_truth": None  # ToolBench doesn't need ground truth
            },
            "extra_info": {
                "index": idx,
                "sample_id": sample_id,
                "category": category,
                "api": ",".join(api_names),  # Comma-separated API names
                "n_required_param": ",".join(n_required_list),  # Comma-separated required param counts
                "n_optional_param": ",".join(n_optional_list)   # Comma-separated optional param counts
            }
        }
        
        records.append(record)
        
        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1} samples...")
    
    df = pd.DataFrame(records)
    
    print(f"\nTotal records: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving to {output_file}...")
    df.to_parquet(output_file, index=False, engine='pyarrow')
    
    print(f"✓ Successfully converted {len(df)} samples to {output_file}")


def split_train_val(input_file: str, train_output: str, val_output: str, 
                    train_ratio: float = 0.9, max_samples: int = None):
    print(f"Reading {input_file}...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Found {len(data)} samples")
    
    if max_samples and max_samples > 0:
        data = data[:max_samples]
        print(f"Processing first {len(data)} samples")
    
    random.seed(42)
    random.shuffle(data)
    
    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    print(f"\nSplitting: {len(train_data)} train, {len(val_data)} val")
    
    print("\nProcessing training set...")
    process_toolbench_json(
        input_file=None,
        output_file=train_output,
        max_samples=None,
        data=train_data
    )
    
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
    parser.add_argument("--train_output", type=str, default='data/toolbench/train.parquet',
                       help="Training set output path (when using --split)")
    parser.add_argument("--val_output", type=str, default='data/toolbench/val.parquet',
                       help="Validation set output path (when using --split)")
    
    args = parser.parse_args()
    
    if args.split:
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
        process_toolbench_json(
            input_file=args.input,
            output_file=args.output,
            max_samples=args.max_samples
        )


if __name__ == "__main__":
    main()
