#!/usr/bin/env python
import json
import pandas as pd
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import os
import random

from system_prompt import SYSTEM_PROMPT

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
    
    # format API list
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
        
        # build parameters information
        required_params = api.get('required_parameters', [])
        optional_params = api.get('optional_parameters', [])
        
        # build parameters dictionary
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
        
        # build API information
        api_info = {
            'name': formatted_api_name,
            'description': api_description,
            'parameters': parameters,
            'template_response': template_response
        }
        api_list_formatted.append(api_info)
    
    return api_list_formatted

def convert_toolbench_to_conversations(sample: Dict, api_names_str: str) -> List[Dict]:
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
    
    # convert API list format
    api_list_formatted = convert_api_list_to_system_format(api_list)
    
    # build conversations
    conversations = [
        {
            "from": "system",
            "role": "system",
            "content": SYSTEM_PROMPT.replace("{api_names}", api_names_str)
        },
        {
            "from": "user",
            "role": "user",
            "content": f"\n{query}\n"
        }
    ]
    
    return conversations, api_list_formatted

def process_toolbench_json(input_file: str, output_file: str, 
                          api_infos_file: str):
    """
    Process G1_query.json format and convert to parquet format.
    
    Args:
        input_file: Input G1_query.json file path (if data is None)
        output_file: Output parquet file path
        api_infos_file: Input api infos file path
    """
    print(f"Reading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Found {len(data)} samples")

    with open(api_infos_file, 'r', encoding='utf-8') as f:
        api_infos = json.load(f)
    api_names = list(api_infos.keys())
    api_names_str = ", ".join(api_names)
    
    records = []
    
    for idx, sample in enumerate(data):        
        # Get api_list from this sample only (not the entire category)
        api_list = sample['api_list']
        
        # Construct conversations
        sample_id = sample["query_id"]
        conversations, api_list_formatted = convert_toolbench_to_conversations(sample, api_names_str)
        
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
            "data_source": "toolbench-eval",  # data source identifier
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
    
    df = pd.DataFrame(records)
    
    print(f"\nTotal records: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving to {output_file}...")
    df.to_parquet(output_file, index=False, engine='pyarrow')
    
    print(f"✓ Successfully converted {len(df)} samples to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Convert StableToolBench data to verl format")
    parser.add_argument("--input_dir", type=str, default="data/toolbench_test_instruction",
                       help="Input directory")
    parser.add_argument("--api_infos", type=str, default="data/api_infos",
                       help="Input api infos file path")
    parser.add_argument("--output_dir", type=str, default='data/toolbench_test',
                       help="Output directory")
    
    args = parser.parse_args()

    input_files = os.listdir(args.input_dir)
    for input_file in input_files:
        input_file_path = os.path.join(args.input_dir, input_file)
        output_file_path = os.path.join(args.output_dir, input_file.replace('.json', '.parquet'))
        api_infos_file_path = os.path.join(args.api_infos, input_file.replace('.json', '.json'))
        process_toolbench_json(
            input_file=input_file_path,
            output_file=output_file_path,
            api_infos_file=api_infos_file_path,
        )


if __name__ == "__main__":
    main()
