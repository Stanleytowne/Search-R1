#!/usr/bin/env python
import argparse
import os
import torch
import pandas as pd
import json
from typing import List, Dict
from tqdm import tqdm

from vllm import LLM, SamplingParams
from verl import DataProto
from verl.utils.model import compute_position_id_with_mask

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
        
        # Preserve meta_info if exists
        if hasattr(data, 'meta_info'):
            response_data.meta_info = data.meta_info.copy()
        
        return response_data


def load_test_data(data_path: str) -> List[Dict]:
    """
    Load test data
    
    Args:
        data_path: Test data path (parquet format)
        
    Returns:
        List of test data
    """
    df = pd.read_parquet(data_path)
    
    # Convert to list format
    data_list = []
    for idx, row in df.iterrows():
        data_item = {
            'prompt': row['prompt'].tolist(),
            'data_source': row['data_source'],
            'extra_info': row.get('extra_info', {}),
        }
        data_list.append(data_item)
    
    return data_list


def create_prompt_from_data(data_item: Dict, tokenizer) -> str:
    prompt_messages = data_item['prompt']
    
    prompt_str = tokenizer.apply_chat_template(
        prompt_messages,
        add_generation_prompt=True,
        tokenize=False
    )
    
    return prompt_str


def evaluate_model_performance(
    model_path: str,
    test_data_path: str,
    output_name: str,
    num_runs: int,
    toolbench_url: str,
    reward_server_url: str = "http://localhost:8000/evaluate_batch",
    batch_size: int = 4,
    max_turns: int = 5,
    max_new_tokens: int = 512,
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
        max_prompt_length=8192,
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
        format_reward_weight=0,
        function_call_reward_weight=0,
        finish_reward_weight=0,
        pass_reward_weight=1,
        num_examine=0,
        reward_server_url=reward_server_url,
    )
    
    # 7. Load test data
    test_data = load_test_data(test_data_path)
    test_data = test_data * num_runs # repeat the data for num_runs times
    
    # 8. Evaluate
    all_results = []
    
    # Process in batches
    num_batches = (len(test_data) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Evaluation progress"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(test_data))
        batch_data = test_data[start_idx:end_idx]
        
        # Prepare batch data
        batch_prompts = []
        batch_extra_info = []
        batch_queries = []
        
        for data_item in batch_data:
            batch_queries.append(data_item['prompt'][1]['content'])
            prompt_str = create_prompt_from_data(data_item, tokenizer)
            batch_prompts.append(prompt_str)
            batch_extra_info.append(data_item.get('extra_info', {}))
        
        # Tokenize prompts
        tokenizer.padding_side = 'left'
        encoded = tokenizer(
            batch_prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=config.max_start_length,
        )
        
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        
        # Create position_ids from attention_mask
        position_ids = compute_position_id_with_mask(attention_mask)
        
        # Create initial DataProto
        initial_input_ids = input_ids
        gen_batch = DataProto.from_dict({
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
        })
        gen_batch.non_tensor_batch = {
            'extra_info': batch_extra_info,
            'data_source': batch_data[0].get('data_source', 'toolbench'),
        }
        
        # Run generation loop
        final_output = generation_manager.run_llm_loop(gen_batch, initial_input_ids)
        final_output.non_tensor_batch['data_source'] = batch_data[0].get('data_source', 'toolbench')

        if final_output.batch['responses'].dtype != torch.int64:
            print("WARNING: responses dtype is not int64")
            final_output.batch['responses'] = final_output.batch['responses'].to(torch.int64)
        
        # Calculate rewards
        rewards = reward_manager(final_output)
        
        # Collect results
        for i in range(len(batch_data)):
            sample_idx = start_idx + i
            reward_value = rewards[i].sum().item()
            
            # Get reward components (simplified, need to get from reward_manager internals)
            result = {
                'sample_idx': sample_idx,
                'acc': reward_value,
                'query': batch_queries[i],
                'response': tokenizer.decode(final_output.batch['responses'][i]),
            }
            
            # Get statistics from meta_info
            if hasattr(final_output, 'meta_info'):
                meta = final_output.meta_info
                result['turns'] = meta.get('turns_stats', [0])[i] if i < len(meta.get('turns_stats', [])) else 0
                result['valid_actions'] = meta.get('valid_action_stats', [0])[i] if i < len(meta.get('valid_action_stats', [])) else 0
                result['finish_called'] = meta.get('finish_called', {}).get(i, None)
            
            all_results.append(result)
    
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

