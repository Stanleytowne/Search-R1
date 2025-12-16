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
import argparse

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

def convert_api_list_to_system_format(api_list: List[Dict]) -> List[Dict]:
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
        - api_list_formatted: formatted API list (for JSON serialization)
    """
    
    # 格式化API列表
    api_list_formatted = []
    
    for api in api_list:
        tool_name = api.get('tool_name', '')
        api_name = api.get('api_name', '')
        api_description = api.get('api_description', '').strip()
        template_response = api.get('template_response', {})
        
        # Build API name (format: api_name_for_tool_name)
        # Normalize both api_name and tool_name
        normalized_api_name = normalize_api_name(api_name)
        normalized_tool_name = normalize_api_name(tool_name)
        formatted_api_name = f"{normalized_api_name}_for_{normalized_tool_name}"
        
        # 构建参数信息
        required_params = api.get('required_parameters', [])
        optional_params = api.get('optional_parameters', [])
        
        # 构建parameters字典
        properties = {}
        required = []
        optional = []
        
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
                    optional.append(param_name)

        parameters = {
            'properties': properties,
            'required': required,
            'optional': optional
        }
        
        # 构建API信息
        api_info = {
            'name': formatted_api_name,
            'description': api_description,
            'parameters': parameters,
            'template_response': template_response
        }
        api_list_formatted.append(api_info)
    
    return api_list_formatted

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
    
    records = {}
    
    for idx, sample in enumerate(data):        
        # Get api_list from this sample only (not the entire category)
        api_list = sample['api_list']
        
        # Construct conversations
        sample_id = sample["query_id"]
        api_list_formatted = convert_api_list_to_system_format(api_list)
        
        # Extract category from api_list (use first API's category_name)
        category = api_list[0]['category_name']
        
        # 为每个category构建去重的API信息集合（使用API的'name'作为唯一标识）
        if category not in records:
            records[category] = {}
        for api in api_list_formatted:
            api_name = api['name']
            if api_name not in records[category]:
                records[category][api_name] = api

    # 创建输出目录
    output_dir = Path("data/api_infos")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for category, api_list in records.items():
        print(f"Category: {category} with {len(api_list)} APIs")

        output_file = output_dir / f"{category}.json"
        # 如果输出文件已存在，则读取已有内容，加上新的不重复的部分
        if output_file.exists():
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    original_data = json.load(f)
            except Exception as e:
                print(f"Warning: Could not read existing {output_file}: {e}")
                original_data = {}
        else:
            original_data = {}

        # 注意：records[category]为dict，original_data假定也是dict结构。合并两个dict（保留原有的不重复内容）
        merged_data = dict(original_data)
        print(f"Original data length: {len(original_data)}")
        for api_name, api_info in api_list.items():
            if api_name not in merged_data:
                merged_data[api_name] = api_info
        print(f"Merged data length: {len(merged_data)}")
        # 保存合并结果
        with open(output_file, "w", encoding='utf-8') as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=4)

def main():
    parser = argparse.ArgumentParser(description="Convert StableToolBench data to verl format")
    parser.add_argument("--input", type=str, default="../StableToolBench/data/instruction/G1_query.json",
                       help="Input JSON file path")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to process (for testing)")
    
    args = parser.parse_args()
    
    process_toolbench_json(
        input_file=args.input,
        max_samples=args.max_samples
    )


if __name__ == "__main__":
    main()
