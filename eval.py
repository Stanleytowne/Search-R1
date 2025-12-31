#!/usr/bin/env python
import argparse
import os
import torch
import pandas as pd
import numpy as np
import json
from typing import List, Dict
from tqdm import tqdm

from torch.utils.data import DataLoader

from vllm import LLM, SamplingParams
from verl import DataProto
from verl.utils.model import compute_position_id_with_mask
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn

from search_r1.llm_agent.generation import LLMGenerationManager, GenerationConfig
from search_r1.llm_agent.toolbench_reward import ToolBenchRewardManager


class SimpleActorRolloutWrapper:
    """Simple Actor Rollout wrapper for simulating generate_sequences method"""
    
    def __init__(self, llm, tokenizer, max_new_tokens=512, temperature=0.7, top_p=0.95):
        self.llm = llm
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
    
    def generate_sequences(self, data: DataProto) -> DataProto:
        """
        Generate sequences
        
        Args:
            data: DataProto containing input_ids, attention_mask, etc.
            
        Returns:
            DataProto containing generated responses
        """
        input_ids = data.batch['input_ids']
        if input_ids.dtype != torch.int64:
            print("WARNING: input_ids dtype is not int64")
            input_ids = input_ids.to(torch.int64)
        
        batch_size = input_ids.shape[0]
        
        # Convert input_ids to prompt strings
        # Remove left padding first
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        prompt_token_ids_list = []
        
        for i in range(batch_size):
            # Remove left padding
            non_pad_mask = input_ids[i] != pad_token_id
            if non_pad_mask.any():
                first_non_pad = non_pad_mask.nonzero(as_tuple=False)[0][0].item()
                token_ids = input_ids[i][first_non_pad:].tolist()
            else:
                token_ids = input_ids[i].tolist()
            prompt_token_ids_list.append(token_ids)
        
        # Create sampling params
        sampling_params = SamplingParams(
            temperature=self.temperature if self.temperature > 0 else 0.0,
            top_p=self.top_p if self.top_p > 0 else 0.0,
            max_tokens=self.max_new_tokens,
            stop_token_ids=[self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id is not None else None,
            min_tokens=5,
            stop=["Observation: {"],
        )
        
        # Generate using vLLM
        outputs = self.llm.generate(
            prompt_token_ids=prompt_token_ids_list,
            sampling_params=sampling_params,
            use_tqdm=False
        )
        
        # Extract generated token ids
        generated_ids_list = []
        for idx, output in enumerate(outputs):
            generated_token_ids = output.outputs[0].token_ids
            # Convert to list
            generated_token_ids = list(generated_token_ids)
            generated_ids_list.append(generated_token_ids)
        
        # Pad to same length and convert to tensor
        max_len = max(len(ids) for ids in generated_ids_list)
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        
        generated_ids = []
        for ids in generated_ids_list:
            padded = ids + [pad_token_id] * (max_len - len(ids))
            generated_ids.append(padded[:self.max_new_tokens])  # Truncate if too long
        
        generated_ids_tensor = torch.tensor(generated_ids, dtype=torch.int64)
        
        # Create response DataProto
        response_data = DataProto.from_dict({
            'responses': generated_ids_tensor,
        })
        return response_data

def evaluate_model_performance(
    model_path: str,
    test_data_path: str,
    output_name: str,
    num_runs: int,
    toolbench_url: str,
    reward_server_url: str = "http://localhost:8000/evaluate_batch",
    batch_size: int = 4,
    max_turns: int = 5,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.95,
    num_gpus: int = 1,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
):
    """
    Evaluate model performance
    
    Args:
        model_path: Model path
        test_data_path: Test data path
        output_name: Output file name
        num_runs: Number of runs
        toolbench_url: ToolBench API server URL
        reward_server_url: Reward server URL
        batch_size: Batch size
        max_turns: Maximum number of turns
        max_new_tokens: Maximum tokens per generation
        temperature: Generation temperature
        top_p: Top-p for generation
        num_gpus: Number of GPUs
        tensor_parallel_size: Tensor parallel size for vLLM
        gpu_memory_utilization: GPU memory utilization for vLLM
    """
    
    # 2. Load model with vLLM
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    tokenizer = llm.get_tokenizer()
    
    # 3. Create ActorRolloutWrapper
    actor_rollout_wg = SimpleActorRolloutWrapper(
        llm=llm,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    
    # 4. Create GenerationConfig
    config = GenerationConfig(
        max_turns=max_turns,
        max_start_length=7168,
        max_prompt_length=16384,
        max_response_length=1024,
        max_obs_length=2000,
        num_gpus=num_gpus,
        toolbench_url=toolbench_url,
        toolbench_max_concurrent=20,
    )
    
    # 5. Create GenerationManager
    generation_manager = LLMGenerationManager(
        tokenizer=tokenizer,
        actor_rollout_wg=actor_rollout_wg,
        config=config,
        is_validation=True
    )
    
    # 6. Create RewardManager
    reward_manager = ToolBenchRewardManager(
        tokenizer=tokenizer,
        num_examine=0,
        reward_server_url=reward_server_url,
    )
    
    # 7. Load test data
    val_dataset = RLHFDataset(parquet_files=test_data_path,
                                tokenizer=tokenizer,
                                prompt_key='prompt',
                                max_prompt_length=16384,
                                filter_prompts=True,
                                return_raw_chat=False,
                                truncation='error')

    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                drop_last=False,
                                collate_fn=collate_fn)

    test_data_size = len(val_dataset)
    
    # 8. Evaluate
    all_results = []
    
    for i in range(num_runs):
        all_rewards = []
        all_queries = []
        all_responses = []
        for batch_dict in val_dataloader:
            test_batch: DataProto = DataProto.from_single_dict(batch_dict)

            test_gen_batch = test_batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
            test_gen_batch.meta_info = {
                'eos_token_id': tokenizer.eos_token_id,
                'pad_token_id': tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': False,
                'validate': True,
            }
            all_queries.extend(tokenizer.batch_decode(test_gen_batch.batch['prompts'], skip_special_tokens=True))

            # Copy extra_info from original batch to gen_batch if available
            if hasattr(test_batch, 'non_tensor_batch') and 'extra_info' in test_batch.non_tensor_batch:
                if not hasattr(test_gen_batch, 'non_tensor_batch'):
                    test_gen_batch.non_tensor_batch = {}
                test_gen_batch.non_tensor_batch['extra_info'] = test_batch.non_tensor_batch['extra_info']
                
            first_input_ids = test_gen_batch.batch['input_ids'].clone()
            final_gen_batch_output = generation_manager.run_llm_loop(
                gen_batch=test_gen_batch,
                initial_input_ids=first_input_ids,
            )

            all_responses.extend(tokenizer.batch_decode(final_gen_batch_output.batch['responses'], skip_special_tokens=True))
            
            test_batch = test_batch.union(final_gen_batch_output)
                
            for key in test_batch.batch.keys():
                test_batch.batch[key] = test_batch.batch[key].long()
                
            reward_tensor = reward_manager(test_batch)
            rewards = reward_tensor.sum(-1).cpu().tolist()
            all_rewards.extend(rewards)
        
        # Collect results
        for i in range(len(all_queries)):
            all_results.append({
                'query': all_queries[i],
                'response': all_responses[i],
                'acc': all_rewards[i],
                'run_idx': i,
            })
    
    # 9. Statistics
    print("\n" + "=" * 80)
    print("Evaluation Results Statistics")
    print("=" * 80)
    
    if not all_results:
        print("No valid evaluation results!")
        return
    
    all_accs = [r['acc'] for r in all_results]
    avg_acc = sum(all_accs) / len(all_accs)
    print(f"Average accuracy: {avg_acc:.4f}")

    run_accs = []
    for run_idx in range(num_runs):
        run_acc = sum(r['acc'] for r in all_results if r['run_idx'] == run_idx) / len(test_data_size)
        run_accs.append(run_acc)
    std_acc = np.std(run_accs).item()
    print(f"Standard deviation of accuracy: {std_acc:.4f}")
    
    print("=" * 80)
    
    # Save detailed results
    output_file = os.path.join(model_path, output_name)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="ToolBench Model Performance Test")
    
    parser.add_argument("--model_path", type=str, required=True, help="Model path")
    parser.add_argument("--output_name", type=str, default="test_results.json", help="Output file name")
    parser.add_argument("--category", type=str, default='Email', help="Category")
    parser.add_argument("--num_runs", type=int, default=1, help="Number of runs")
    parser.add_argument("--toolbench_url", type=str, default='http://127.0.0.1:12345', help="ToolBench API server URL")
    parser.add_argument("--reward_server_url", type=str, default="http://localhost:12346/evaluate_batch", 
                       help="Reward server URL")
    
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--max_turns", type=int, default=5, help="Maximum number of turns")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Maximum tokens per generation")
    parser.add_argument("--temperature", type=float, default=1., help="Generation temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p for generation")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size for vLLM")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="GPU memory utilization for vLLM")
    
    args = parser.parse_args()
    
    evaluate_model_performance(
        model_path=args.model_path,
        test_data_path=f'data/toolbench_test/{args.category}.parquet',
        output_name=args.output_name,
        num_runs=args.num_runs,
        toolbench_url=args.toolbench_url,
        reward_server_url=args.reward_server_url,
        batch_size=args.batch_size,
        max_turns=args.max_turns,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        num_gpus=args.num_gpus,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    

if __name__ == "__main__":
    main()

