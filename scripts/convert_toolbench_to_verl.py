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


def extract_category_from_id(sample_id: str) -> str:
    """
    从sample id中提取category信息
    例如："Step 9: ..." -> "G1_category"
    """
    # 可以根据实际数据格式调整
    if "Step" in sample_id:
        return "G1_category"
    return "G1_category"  # 默认值


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
        sample_id = sample.get("id", f"sample_{idx}")
        conversations = sample.get("conversations", [])
        
        if not conversations:
            print(f"Warning: Sample {idx} has no conversations, skipping")
            continue
        
        # 转换为chat格式
        chat = convert_conversations_to_chat(conversations)
        
        if not chat:
            print(f"Warning: Sample {idx} has empty chat, skipping")
            continue
        
        # 提取category（用于ToolBench API调用）
        category = extract_category_from_id(sample_id)
        
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
    parser.add_argument("--input", type=str, required=True,
                       help="Input JSON file path")
    parser.add_argument("--output", type=str, required=True,
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
