#!/usr/bin/env python
"""
过滤parquet数据中过长的条目

支持多种过滤方式：
1. 按字符数过滤（prompt的字符串长度）
2. 按token数过滤（需要tokenizer）
3. 按conversations数量过滤
"""

import pandas as pd
import argparse
from pathlib import Path
from typing import Optional
import json


def count_tokens(text: str, tokenizer=None) -> int:
    """计算文本的token数量"""
    if tokenizer is None:
        # 如果没有tokenizer，使用简单的字符数估算（1 token ≈ 4 characters）
        return len(text) // 4
    else:
        return len(tokenizer.encode(text))


def get_prompt_length(sample: dict, method: str = 'chars', tokenizer=None) -> int:
    """
    获取prompt的长度
    
    Args:
        sample: 数据样本
        method: 计算方法 ('chars', 'tokens', 'messages')
        tokenizer: tokenizer对象（当method='tokens'时需要）
    
    Returns:
        长度值
    """
    prompt = sample.get('prompt', [])
    
    if method == 'chars':
        # 计算所有消息的字符总数
        total_chars = 0
        if isinstance(prompt, list):
            for msg in prompt:
                content = msg.get('value', '') or msg.get('content', '')
                total_chars += len(str(content))
        else:
            total_chars = len(str(prompt))
        return total_chars
    
    elif method == 'tokens':
        # 计算所有消息的token总数
        total_tokens = 0
        if isinstance(prompt, list):
            for msg in prompt:
                content = msg.get('value', '') or msg.get('content', '')
                total_tokens += count_tokens(str(content), tokenizer)
        else:
            total_tokens = count_tokens(str(prompt), tokenizer)
        return total_tokens
    
    elif method == 'messages':
        # 计算消息数量
        if isinstance(prompt, list):
            return len(prompt)
        else:
            return 1
    
    else:
        raise ValueError(f"Unknown method: {method}")


def filter_long_samples(
    input_file: str,
    output_file: str,
    max_length: int,
    method: str = 'chars',
    tokenizer_path: Optional[str] = None,
    verbose: bool = True
):
    """
    过滤掉过长的样本
    
    Args:
        input_file: 输入的parquet文件路径
        output_file: 输出的parquet文件路径
        max_length: 最大长度阈值
        method: 长度计算方法 ('chars', 'tokens', 'messages')
        tokenizer_path: tokenizer路径（当method='tokens'时需要）
        verbose: 是否打印详细信息
    """
    # 加载tokenizer（如果需要）
    tokenizer = None
    if method == 'tokens' and tokenizer_path:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            if verbose:
                print(f"Loaded tokenizer from {tokenizer_path}")
        except Exception as e:
            print(f"Warning: Failed to load tokenizer: {e}")
            print("Falling back to character-based estimation")
            method = 'chars'
    
    # 读取数据
    if verbose:
        print(f"Reading {input_file}...")
    df = pd.read_parquet(input_file, engine='pyarrow')
    
    original_count = len(df)
    if verbose:
        print(f"Original samples: {original_count}")
    
    # 计算每个样本的长度
    if verbose:
        print(f"Calculating lengths using method: {method}...")
    
    lengths = df.apply(
        lambda row: get_prompt_length(row.to_dict(), method=method, tokenizer=tokenizer),
        axis=1
    )
    
    # 打印统计信息
    if verbose:
        print(f"\nLength statistics ({method}):")
        print(f"  Min: {lengths.min()}")
        print(f"  Max: {lengths.max()}")
        print(f"  Mean: {lengths.mean():.2f}")
        print(f"  Median: {lengths.median():.2f}")
        print(f"  95th percentile: {lengths.quantile(0.95):.2f}")
        print(f"  99th percentile: {lengths.quantile(0.99):.2f}")
        print(f"\nFiltering with max_length={max_length}...")
    
    # 过滤
    mask = lengths <= max_length
    filtered_df = df[mask]
    
    filtered_count = len(filtered_df)
    removed_count = original_count - filtered_count
    
    if verbose:
        print(f"\nFiltered samples: {filtered_count}")
        print(f"Removed samples: {removed_count} ({removed_count/original_count*100:.2f}%)")
        print(f"Remaining samples: {filtered_count/original_count*100:.2f}%")
    
    # 保存结果
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"\nSaving to {output_file}...")
    filtered_df.to_parquet(output_file, index=False, engine='pyarrow')
    
    if verbose:
        print(f"✓ Successfully saved {filtered_count} samples to {output_file}")
    
    return filtered_df


def main():
    parser = argparse.ArgumentParser(
        description="Filter out long samples from parquet dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Filter by character count (default)
  python filter_long_samples.py --input data/train.parquet --output data/train_filtered.parquet --max_length 10000
  
  # Filter by token count (requires tokenizer)
  python filter_long_samples.py --input data/train.parquet --output data/train_filtered.parquet \\
      --max_length 2048 --method tokens --tokenizer ToolBench/ToolLLaMA-2-7b-v2
  
  # Filter by number of messages
  python filter_long_samples.py --input data/train.parquet --output data/train_filtered.parquet \\
      --max_length 20 --method messages
        """
    )
    
    parser.add_argument("--input", type=str, default='data/toolbench/val.parquet',
                       help="Input parquet file path")
    parser.add_argument("--output", type=str, default='data/toolbench/val_filtered.parquet',
                       help="Output parquet file path")
    parser.add_argument("--max_length", type=int, default=4096,
                       help="Maximum length threshold")
    parser.add_argument("--method", type=str, default="tokens",
                       choices=["chars", "tokens", "messages"],
                       help="Length calculation method (default: chars)")
    parser.add_argument("--tokenizer", type=str, default='ToolBench/ToolLLaMA-2-7b-v2',
                       help="Tokenizer path (required when method=tokens)")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress verbose output")
    
    args = parser.parse_args()
    
    # 验证参数
    if args.method == "tokens" and not args.tokenizer:
        parser.error("--tokenizer is required when --method=tokens")
    
    # 执行过滤
    filter_long_samples(
        input_file=args.input,
        output_file=args.output,
        max_length=args.max_length,
        method=args.method,
        tokenizer_path=args.tokenizer,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
