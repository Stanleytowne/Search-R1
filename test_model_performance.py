#!/usr/bin/env python
import argparse
import torch
import pandas as pd
import json
import logging
import sys
from typing import List, Dict, Any
from pathlib import Path
from tqdm import tqdm

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from verl import DataProto

from search_r1.llm_agent.generation import LLMGenerationManager, GenerationConfig, normalize_api_name
from search_r1.llm_agent.toolbench_reward import ToolBenchRewardManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class SimpleActorRolloutWrapper:
    """Simple Actor Rollout wrapper for simulating generate_sequences method"""
    
    def __init__(self, llm, tokenizer, max_new_tokens=512, temperature=0.7):
        self.llm = llm
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        logger.info(f"Initialized SimpleActorRolloutWrapper: max_new_tokens={max_new_tokens}, temperature={temperature}")
    
    def generate_sequences(self, data: DataProto) -> DataProto:
        """
        Generate sequences
        
        Args:
            data: DataProto containing input_ids, attention_mask, etc.
            
        Returns:
            DataProto containing generated responses
        """
        logger.debug("Starting generate_sequences")
        input_ids = data.batch['input_ids']
        attention_mask = data.batch.get('attention_mask', torch.ones_like(input_ids))
        
        batch_size = input_ids.shape[0]
        logger.debug(f"Processing batch size: {batch_size}, input_ids shape: {input_ids.shape}")
        
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
                logger.debug(f"Sample {i}: removed {first_non_pad} padding tokens, remaining length: {len(token_ids)}")
            else:
                token_ids = input_ids[i].tolist()
                logger.debug(f"Sample {i}: no padding found, length: {len(token_ids)}")
            prompt_token_ids_list.append(token_ids)
        
        # Create sampling params
        sampling_params = SamplingParams(
            temperature=self.temperature if self.temperature > 0 else 0.0,
            max_tokens=self.max_new_tokens,
            stop_token_ids=[self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id is not None else None,
        )
        logger.debug(f"Sampling params: temperature={sampling_params.temperature}, max_tokens={sampling_params.max_tokens}")
        
        # Generate using vLLM
        logger.info(f"Calling vLLM generate for {batch_size} samples")
        outputs = self.llm.generate(
            prompt_token_ids=prompt_token_ids_list,
            sampling_params=sampling_params,
            use_tqdm=False
        )
        logger.info(f"vLLM generate completed, received {len(outputs)} outputs")
        
        # Extract generated token ids
        generated_ids_list = []
        for idx, output in enumerate(outputs):
            generated_token_ids = output.outputs[0].token_ids
            generated_ids_list.append(generated_token_ids)
            logger.debug(f"Output {idx}: generated {len(generated_token_ids)} tokens")
        
        # Pad to same length and convert to tensor
        max_len = max(len(ids) for ids in generated_ids_list) if generated_ids_list else self.max_new_tokens
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        logger.debug(f"Max generated length: {max_len}, padding to max_new_tokens: {self.max_new_tokens}")
        
        generated_ids = []
        for ids in generated_ids_list:
            padded = ids + [pad_token_id] * (max_len - len(ids))
            generated_ids.append(padded[:self.max_new_tokens])  # Truncate if too long
        
        generated_ids_tensor = torch.tensor(generated_ids, dtype=torch.long)
        logger.debug(f"Generated tensor shape: {generated_ids_tensor.shape}")
        
        # Create response DataProto
        response_data = DataProto.from_dict({
            'responses': generated_ids_tensor,
        })
        
        # Preserve meta_info if exists
        if hasattr(data, 'meta_info'):
            response_data.meta_info = data.meta_info.copy()
            logger.debug("Preserved meta_info from input data")
        
        logger.info(f"Successfully generated sequences, response shape: {generated_ids_tensor.shape}")
        return response_data


def load_test_data(data_path: str, max_samples: int = None) -> List[Dict]:
    """
    Load test data
    
    Args:
        data_path: Test data path (parquet format)
        max_samples: Maximum number of samples (None means load all)
        
    Returns:
        List of test data
    """
    logger.info(f"Loading test data from: {data_path}")
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded parquet file, total rows: {len(df)}, columns: {df.columns.tolist()}")
    
    if max_samples:
        logger.info(f"Limiting to {max_samples} samples")
        df = df.head(max_samples)
    
    # Convert to list format
    data_list = []
    for idx, row in df.iterrows():
        try:
            data_item = {
                'prompt': row['prompt'] if isinstance(row['prompt'], list) else json.loads(row['prompt']),
                'data_source': row.get('data_source', 'toolbench'),
                'extra_info': row.get('extra_info', {}) if isinstance(row.get('extra_info', {}), dict) else json.loads(row.get('extra_info', '{}')),
            }
            data_list.append(data_item)
        except Exception as e:
            logger.warning(f"Error processing row {idx}: {e}, skipping")
            continue
    
    logger.info(f"Successfully loaded {len(data_list)} test samples")
    return data_list


def create_prompt_from_data(data_item: Dict, tokenizer) -> str:
    prompt_messages = data_item['prompt']
    
    prompt_str = tokenizer.apply_chat_template(
        prompt_messages,
        add_generation_prompt=True,
        tokenize=False
    )
    
    logger.debug(f"Created prompt, length: {len(prompt_str)} characters")
    return prompt_str


def evaluate_model_performance(
    model_path: str,
    test_data_path: str,
    toolbench_url: str,
    reward_server_url: str = "http://localhost:8000/evaluate_batch",
    max_samples: int = None,
    batch_size: int = 4,
    max_turns: int = 5,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    format_reward_weight: float = 0.1,
    function_call_reward_weight: float = 0.2,
    finish_reward_weight: float = 0.3,
    pass_reward_weight: float = 0.1,
    num_gpus: int = 1,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
):
    """
    Evaluate model performance
    
    Args:
        model_path: Model path
        test_data_path: Test data path
        toolbench_url: ToolBench API server URL
        reward_server_url: Reward server URL
        max_samples: Maximum number of test samples
        batch_size: Batch size
        max_turns: Maximum number of turns
        max_new_tokens: Maximum tokens per generation
        temperature: Generation temperature
        format_reward_weight: Format reward weight
        function_call_reward_weight: Function call reward weight
        finish_reward_weight: Finish reward weight
        pass_reward_weight: Pass reward weight
        num_gpus: Number of GPUs
        tensor_parallel_size: Tensor parallel size for vLLM
        gpu_memory_utilization: GPU memory utilization for vLLM
    """
    print("=" * 80)
    print("ToolBench Model Performance Test")
    print("=" * 80)
    print(f"Model path: {model_path}")
    print(f"Test data: {test_data_path}")
    print(f"ToolBench URL: {toolbench_url}")
    print(f"Reward Server URL: {reward_server_url}")
    print(f"Max samples: {max_samples if max_samples else 'All'}")
    print(f"Batch size: {batch_size}")
    print(f"Max turns: {max_turns}")
    print("=" * 80)
    
    # 1. Load tokenizer
    print("\n[1/6] Loading tokenizer...")
    logger.info(f"[Step 1/6] Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")
    logger.info(f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}, pad_token_id={tokenizer.pad_token_id}, eos_token_id={tokenizer.eos_token_id}")
    
    # 2. Load model with vLLM
    print("\n[2/6] Loading model with vLLM...")
    logger.info(f"[Step 2/6] Loading model with vLLM: tensor_parallel_size={tensor_parallel_size}, gpu_memory_utilization={gpu_memory_utilization}")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
    )
    logger.info("vLLM model loaded successfully")
    
    # 3. Create ActorRolloutWrapper
    print("\n[3/6] Creating generator...")
    logger.info(f"[Step 3/6] Creating ActorRolloutWrapper: max_new_tokens={max_new_tokens}, temperature={temperature}")
    actor_rollout_wg = SimpleActorRolloutWrapper(
        llm=llm,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature
    )
    logger.info("ActorRolloutWrapper created successfully")
    
    # 4. Create GenerationConfig
    print("\n[4/6] Creating generation config...")
    logger.info(f"[Step 4/6] Creating GenerationConfig: max_turns={max_turns}, toolbench_url={toolbench_url}")
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
    logger.info(f"GenerationConfig created: max_turns={config.max_turns}, max_start_length={config.max_start_length}")
    
    # 5. Create GenerationManager
    logger.info(f"[Step 5/6] Creating LLMGenerationManager")
    generation_manager = LLMGenerationManager(
        tokenizer=tokenizer,
        actor_rollout_wg=actor_rollout_wg,
        config=config,
        is_validation=True
    )
    logger.info("LLMGenerationManager created successfully")
    
    # 6. Create RewardManager
    logger.info(f"[Step 6/6] Creating ToolBenchRewardManager")
    logger.info(f"Reward weights: format={format_reward_weight}, function_call={function_call_reward_weight}, "
                f"finish={finish_reward_weight}, pass={pass_reward_weight}")
    reward_manager = ToolBenchRewardManager(
        tokenizer=tokenizer,
        format_reward_weight=format_reward_weight,
        function_call_reward_weight=function_call_reward_weight,
        finish_reward_weight=finish_reward_weight,
        pass_reward_weight=pass_reward_weight,
        num_examine=3,  # Print detailed info for first 3 samples
        reward_server_url=reward_server_url,
    )
    logger.info(f"ToolBenchRewardManager created, reward_server_url={reward_server_url}")
    
    # 7. Load test data
    print("\n[5/6] Loading test data...")
    logger.info(f"[Step 7] Loading test data from: {test_data_path}, max_samples={max_samples}")
    test_data = load_test_data(test_data_path, max_samples=max_samples)
    print(f"Loaded {len(test_data)} test samples")
    
    # 8. Evaluate
    print("\n[6/6] Starting evaluation...")
    logger.info(f"[Step 8] Starting evaluation: batch_size={batch_size}, total_samples={len(test_data)}")
    all_results = []
    
    # Process in batches
    num_batches = (len(test_data) + batch_size - 1) // batch_size
    logger.info(f"Will process {num_batches} batches")
    
    for batch_idx in tqdm(range(num_batches), desc="Evaluation progress"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(test_data))
        batch_data = test_data[start_idx:end_idx]
        logger.info(f"Processing batch {batch_idx + 1}/{num_batches}: samples {start_idx} to {end_idx - 1}")
        
        try:
            # Prepare batch data
            logger.debug(f"Preparing batch data for {len(batch_data)} samples")
            batch_prompts = []
            batch_extra_info = []
            
            for data_item in batch_data:
                prompt_str = create_prompt_from_data(data_item, tokenizer)
                batch_prompts.append(prompt_str)
                batch_extra_info.append(data_item.get('extra_info', {}))
            
            logger.debug(f"Created {len(batch_prompts)} prompts")
            
            # Tokenize prompts
            logger.debug("Tokenizing prompts")
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
            logger.debug(f"Tokenized: input_ids shape={input_ids.shape}, attention_mask shape={attention_mask.shape}")
            
            # Create initial DataProto
            logger.debug("Creating initial DataProto")
            initial_input_ids = input_ids
            gen_batch = DataProto.from_dict({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
            })
            gen_batch.non_tensor_batch = {
                'extra_info': batch_extra_info,
                'data_source': batch_data[0].get('data_source', 'toolbench'),
            }
            logger.debug(f"DataProto created: input_ids shape={input_ids.shape}")
            
            # Run generation loop
            logger.info(f"Running generation loop for batch {batch_idx + 1}")
            final_output = generation_manager.run_llm_loop(gen_batch, initial_input_ids)
            logger.info(f"Generation loop completed for batch {batch_idx + 1}")
            
            # Calculate rewards
            logger.debug("Calculating rewards")
            rewards = reward_manager(final_output)
            logger.debug(f"Rewards calculated: shape={[r.shape for r in rewards] if isinstance(rewards, list) else rewards.shape}")
            
            # Collect results
            logger.debug("Collecting results")
            for i in range(len(batch_data)):
                sample_idx = start_idx + i
                reward_value = rewards[i].max().item() if rewards[i].numel() > 0 else 0.0
                
                # Get reward components (simplified, need to get from reward_manager internals)
                result = {
                    'sample_idx': sample_idx,
                    'total_reward': reward_value,
                    'query': batch_prompts[i][:200] + '...' if len(batch_prompts[i]) > 200 else batch_prompts[i],
                }
                
                # Get statistics from meta_info
                if hasattr(final_output, 'meta_info'):
                    meta = final_output.meta_info
                    result['turns'] = meta.get('turns_stats', [0])[i] if i < len(meta.get('turns_stats', [])) else 0
                    result['valid_actions'] = meta.get('valid_action_stats', [0])[i] if i < len(meta.get('valid_action_stats', [])) else 0
                    result['finish_called'] = meta.get('finish_called', {}).get(i, None)
                
                all_results.append(result)
                logger.debug(f"Sample {sample_idx}: reward={reward_value:.4f}, turns={result.get('turns', 0)}, "
                            f"valid_actions={result.get('valid_actions', 0)}, finish_called={result.get('finish_called')}")
            
            logger.info(f"Batch {batch_idx + 1} completed successfully, collected {len(batch_data)} results")
                
        except Exception as e:
            logger.error(f"Batch {batch_idx} processing failed: {e}", exc_info=True)
            print(f"\n[Error] Batch {batch_idx} processing failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 9. Statistics
    print("\n" + "=" * 80)
    print("Evaluation Results Statistics")
    print("=" * 80)
    logger.info("=" * 80)
    logger.info("Computing evaluation statistics")
    
    if not all_results:
        logger.warning("No valid evaluation results!")
        print("No valid evaluation results!")
        return
    
    logger.info(f"Total results collected: {len(all_results)}")
    total_rewards = [r['total_reward'] for r in all_results]
    avg_reward = sum(total_rewards) / len(total_rewards)
    max_reward = max(total_rewards)
    min_reward = min(total_rewards)
    
    logger.info(f"Reward statistics: avg={avg_reward:.4f}, max={max_reward:.4f}, min={min_reward:.4f}")
    print(f"Total samples: {len(all_results)}")
    print(f"Average reward: {avg_reward:.4f}")
    print(f"Max reward: {max_reward:.4f}")
    print(f"Min reward: {min_reward:.4f}")
    
    # Statistics for Finish call rate
    finish_called = [r.get('finish_called') for r in all_results if r.get('finish_called') is not None]
    if finish_called:
        finish_rate = sum(finish_called) / len(finish_called) if finish_called else 0.0
        logger.info(f"Finish call rate: {finish_rate:.2%} ({sum(finish_called)}/{len(finish_called)})")
        print(f"Finish call rate: {finish_rate:.2%} ({sum(finish_called)}/{len(finish_called)})")
    
    # Statistics for average turns
    avg_turns = sum(r.get('turns', 0) for r in all_results) / len(all_results)
    logger.info(f"Average turns: {avg_turns:.2f}")
    print(f"Average turns: {avg_turns:.2f}")
    
    # Statistics for average valid actions
    avg_valid_actions = sum(r.get('valid_actions', 0) for r in all_results) / len(all_results)
    logger.info(f"Average valid actions: {avg_valid_actions:.2f}")
    print(f"Average valid actions: {avg_valid_actions:.2f}")
    
    print("=" * 80)
    
    # Save detailed results
    output_file = "test_results.json"
    logger.info(f"Saving detailed results to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    logger.info(f"Results saved successfully, {len(all_results)} samples")
    print(f"\nDetailed results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="ToolBench Model Performance Test")
    
    parser.add_argument("--model_path", type=str, required=True, help="Model path")
    parser.add_argument("--test_data_path", type=str, default='data/toolbench_test/Email.parquet', help="Test data path (parquet format)")
    parser.add_argument("--toolbench_url", type=str, default='http://127.0.0.1:12345', help="ToolBench API server URL")
    parser.add_argument("--output_file", type=str, default="test_results.json", help="Output file path")
    parser.add_argument("--reward_server_url", type=str, default="http://localhost:8000/evaluate_batch", 
                       help="Reward server URL")
    
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of test samples")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--max_turns", type=int, default=5, help="Maximum number of turns")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Maximum tokens per generation")
    parser.add_argument("--temperature", type=float, default=0.95, help="Generation temperature")
    parser.add_argument("--pass_reward_weight", type=float, default=1, help="Pass reward weight")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size for vLLM")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="GPU memory utilization for vLLM")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Set logging level
    logger.setLevel(getattr(logging, args.log_level))
    logger.info("=" * 80)
    logger.info("Starting ToolBench Model Performance Test")
    logger.info("=" * 80)
    logger.info(f"Arguments: {vars(args)}")
    
    evaluate_model_performance(
        model_path=args.model_path,
        test_data_path=args.test_data_path,
        toolbench_url=args.toolbench_url,
        reward_server_url=args.reward_server_url,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        max_turns=args.max_turns,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        format_reward_weight=0,
        function_call_reward_weight=0,
        finish_reward_weight=0,
        pass_reward_weight=args.pass_reward_weight,
        num_gpus=args.num_gpus,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    
    logger.info("=" * 80)
    logger.info("ToolBench Model Performance Test completed")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

