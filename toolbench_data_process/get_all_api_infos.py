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

def process_toolbench_json(input_file: str = None, output_dir: str = None):
    """
    Process G1_query.json format and convert to parquet format.
    
    Args:
        input_file: Input G1_query.json file path (if data is None)
        output_dir: Output directory
    """
    print(f"Reading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Found {len(data)} samples")
    
    records = {}
    
    for sample in data:        
        # Get api_list from this sample only (not the entire category)
        api_list = sample['api_list']
        
        # Construct conversations
        api_list_formatted = convert_api_list_to_system_format(api_list)
        
        # Extract category from api_list (use first API's category_name)
        category = api_list[0]['category_name']
        
        # build unique API information set for each category (use API's 'name' as the unique identifier)
        if category not in records:
            records[category] = {}
        for api in api_list_formatted:
            api_name = api['name']
            if api_name not in records[category]:
                records[category][api_name] = api

    # create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    for category, api_list in records.items():
        print(f"Category: {category} with {len(api_list)} APIs")

        output_file = os.path.join(output_dir, f"{category}.json")
        # if the output file exists, read the existing content, and add the new non-duplicate part
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    original_data = json.load(f)
            except Exception as e:
                print(f"Warning: Could not read existing {output_file}: {e}")
                original_data = {}
        else:
            original_data = {}

        # Note: records[category] is a dict, original_data is assumed to be a dict structure. Merge two dicts (keep the original non-duplicate content)
        merged_data = dict(original_data)
        print(f"Original data length: {len(original_data)}")
        for api_name, api_info in api_list.items():
            if api_name not in merged_data:
                merged_data[api_name] = api_info
        print(f"Merged data length: {len(merged_data)}")
        # save merged result
        with open(output_file, "w", encoding='utf-8') as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=4)

def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--input_files", nargs='+', type=str, default=['data-toolbench/instruction/G1_query.json', 'data-toolbench/instruction/G2_query.json', 'data-toolbench/solvable_queries/test_instruction/G1_instruction.json', 'data-toolbench/solvable_queries/test_instruction/G1_category.json', 'data-toolbench/solvable_queries/test_instruction/G1_tool.json', 'data-toolbench/solvable_queries/test_instruction/G2_instruction.json', 'data-toolbench/solvable_queries/test_instruction/G2_category.json'],
                       help="Input files")
    parser.add_argument("--output_dir", type=str, default="data/api_infos",
                       help="Output directory")
    args = parser.parse_args()

    for input_file in args.input_files:
        process_toolbench_json(
            input_file=input_file,
            output_dir=args.output_dir
        )

if __name__ == "__main__":
    main()
