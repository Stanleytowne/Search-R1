import torch
import re
from collections import defaultdict
import os
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from .tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
from verl.utils.tracking import Tracking
import shutil
import requests
import json
import numpy as np
import asyncio
import httpx

FINISH_PROMPT = "Thought: Since the tool call limit has been reached, I now need to summarize my thoughts and call 'Finish' to end the task."

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

@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int 
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    no_think_rl: bool=False
    search_url: str = None
    topk: int = 3
    # ToolBench related configs
    toolbench_url: str = None
    toolbench_max_concurrent: int = 20  # 限制同时并发的 API 请求数量，防止内存 OOM

class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))
        
        # Track API calls and Finish calls for reward computation
        # Format: {sample_idx: [True, False, ...]}
        self.api_success_history = []
        # Format: {sample_idx: True | False}
        self.finish_call_history = []

        # =========================================================================
        # [网络优化] 初始化持久化 HTTP 客户端
        # max_keepalive_connections: 保持的长连接数
        # max_connections: 最大并发数，防止把服务器打挂
        # =========================================================================
        limits = httpx.Limits(max_keepalive_connections=20, max_connections=100)
        timeout = httpx.Timeout(30.0, connect=10.0) # connect 超时设短一点，fail fast
        self.async_client = httpx.AsyncClient(limits=limits, timeout=timeout)
        
        # =========================================================================
        # [并发控制] 使用 Semaphore 限制同时并发的 API 请求数量
        # 防止在 batch 很大时同时发起过多请求导致内存 OOM
        # =========================================================================
        max_concurrent = getattr(config, 'toolbench_max_concurrent', 20)
        self.api_call_semaphore = asyncio.Semaphore(max_concurrent)

    def __del__(self):
        """析构函数：确保关闭网络连接"""
        try:
            # 关闭同步 session
            self.sync_session.close()
            # 关闭异步 client (需要 event loop，这里尽力尝试)
            if hasattr(self, 'async_client') and not self.async_client.is_closed:
                asyncio.run(self.async_client.aclose())
        except Exception:
            pass

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']

    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        """remove the eos token from the responses"""
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )        
        responses = self._batch_tokenize(responses_str)
        return responses, responses_str

    def _process_next_obs(self, next_obs: List[str], last: bool = False, active_mask: torch.Tensor = None) -> torch.Tensor:
        """Process next observations from environment."""
        if last:
            # for the last turn, we add the finish prompt to force the model to call 'Finish'
            if active_mask is None:
                active_mask = torch.ones(len(next_obs), dtype=torch.bool)
            for i in range(len(next_obs)):
                if active_mask[i]:
                    next_obs[i] += FINISH_PROMPT

        next_obs_ids = self.tokenizer(
            next_obs, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']

        if next_obs_ids.shape[1] > self.config.max_obs_length:
            print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.config.max_obs_length}")            
            next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]

        return next_obs_ids

    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor, 
                            next_obs_ids: torch.Tensor) -> Dict:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding        
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ])
        
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        new_rollings = DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })
        new_rollings.meta_info.update(rollings.meta_info)
        
        return new_rollings

    def _info_masked_concatenate_with_padding(self, 
                prompt: torch.Tensor, 
                prompt_with_mask: torch.Tensor, 
                response: torch.Tensor, 
                info: torch.Tensor = None,
                pad_to_left: bool = True
            ) -> torch.Tensor:
        """Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to cover the information block if it exists."""
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        if info is not None:
            tensors.append(info)
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device) # information mask
            tensors_with_mask.append(info_mask)
        
        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)

        return padded_tensor, padded_tensor_with_info

    def _update_right_side(self, right_side: Dict, 
                          cur_responses: torch.Tensor,
                          next_obs_ids: torch.Tensor = None) -> Dict:
        """Update right side state."""
        if next_obs_ids != None:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    next_obs_ids, 
                    pad_to_left=False
                )
        else:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    pad_to_left=False
                )
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {'responses': responses[:, :max_len], 'responses_with_info_mask': responses_with_info_mask[:, :max_len]}

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        for key in active_batch.batch.keys():
            active_batch.batch[key] = active_batch.batch[key].long()
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
        
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)
        for key in padded_active_batch.batch.keys():
            padded_active_batch.batch[key] = padded_active_batch.batch[key].long()

        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)

        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output

    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""
        
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {'responses': initial_input_ids[:, []], 'responses_with_info_mask': initial_input_ids[:, []]}
        
        batch_size = gen_batch.batch['input_ids'].shape[0]

        active_mask = torch.ones(batch_size, dtype=torch.bool)
        turns_stats = torch.ones(batch_size, dtype=torch.int)
        # valid_action_stats = torch.zeros(batch_size, dtype=torch.int)
        # valid_action_stats is a list of lists, each list is a list of valid actions for a sample
        valid_action_stats = [[] for _ in range(batch_size)]
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch
        
        # 1. Initialize ToolBench related variables: api_name, category, etc.
        
        self.api_success_history = [[] for _ in range(batch_size)]
        self.finish_call_history = [False for _ in range(batch_size)]
        # Extract category and API list from extra_info for each sample
        self.sample_categories = {}
        self.sample_api_lists = {}  # Store API validation info for each sample
        
        # Get extra_info from non_tensor_batch or meta_info
        extra_info = None
        if hasattr(gen_batch, 'non_tensor_batch') and 'extra_info' in gen_batch.non_tensor_batch:
            extra_info = gen_batch.non_tensor_batch['extra_info']
        elif hasattr(gen_batch, 'meta_info') and 'extra_info' in gen_batch.meta_info:
            extra_info = gen_batch.meta_info['extra_info']
        else:
            raise ValueError(f"Missing extra_info in gen_batch. Cannot proceed without category information.")
        
        if not isinstance(extra_info, (list, tuple, np.ndarray)) or len(extra_info) != batch_size:
            raise ValueError(f"Invalid extra_info: expected list/tuple/array of length {batch_size}, got {type(extra_info)} with length {len(extra_info) if hasattr(extra_info, '__len__') else 'N/A'}")
        
        # Extract category and API info for each sample
        for i, info in enumerate(extra_info):
            
            self.sample_categories[i] = info['category']
            
            # Extract API validation info (simplified format: api, n_required_param, n_optional_param)
            # Parse comma-separated API names and counts
            api_names = [name.strip() for name in info['api'].split(',') if name.strip()]
            n_required_str = info.get('n_required_param', '')
            n_optional_str = info.get('n_optional_param', '')
            
            # Parse comma-separated counts
            n_required_list = [int(x.strip()) for x in n_required_str.split(',') if x.strip()] if n_required_str else []
            n_optional_list = [int(x.strip()) for x in n_optional_str.split(',') if x.strip()] if n_optional_str else []
            
            # Build API validation dict (normalize API names)
            api_validation_info = {}
            for idx_api, api_name in enumerate(api_names):
                normalized_name = normalize_api_name(api_name)
                required_count = n_required_list[idx_api] if idx_api < len(n_required_list) else 0
                optional_count = n_optional_list[idx_api] if idx_api < len(n_optional_list) else 0
                api_validation_info[normalized_name] = {
                    'required_count': required_count,
                    'optional_count': optional_count
                }
            self.sample_api_lists[i] = api_validation_info

        # 2. Main generation loop
        for step in range(self.config.max_turns):
            if not active_mask.sum():
                break
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            
            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })            
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # Execute in environment and process observations
            # Map active indices back to original batch indices for API call tracking
            active_indices = torch.where(active_mask)[0].tolist() if isinstance(active_mask, torch.Tensor) else [i for i, active in enumerate(active_mask) if active]
            next_obs, dones, valid_action = self.execute_predictions(
                responses_str, active_mask, original_indices=active_indices
            )
            
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            turns_stats[curr_active_mask] += 1
            # valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            for i, valid in enumerate(valid_action):
                valid_action_stats[i].append(valid)

            next_obs_ids = self._process_next_obs(next_obs, last=step==self.config.max_turns-1, active_mask=active_mask)
            
            # Update states
            rollings = self._update_rolling_state(
                rollings,
                responses_ids,
                next_obs_ids
            )
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                next_obs_ids
            )
            
        # final LLM rollout
        if active_mask.sum():
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )

            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })            
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])

            # add finish prompt to responses_str
            # will not effect the responses_ids and rollings
            responses_str = [FINISH_PROMPT + response for response in responses_str]
            
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # # Execute in environment and process observations
            # Map active indices back to original batch indices
            active_indices_final = torch.where(active_mask)[0].tolist() if isinstance(active_mask, torch.Tensor) else [i for i, active in enumerate(active_mask) if active]
            _, dones, valid_action = self.execute_predictions(
                responses_str, active_mask, original_indices=active_indices_final
            )

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            # valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            for i, valid in enumerate(valid_action):
                valid_action_stats[i].append(valid)
            

            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
            )
        
        meta_info['turns_stats'] = turns_stats.tolist()
        meta_info['active_mask'] = active_mask.tolist()
        meta_info['valid_action_stats'] = valid_action_stats
        
        # Add ToolBench reward computation info
        # meta_info['api_success_history'] = self.api_success_history.copy()
        meta_info['api_success_history'] = self.api_success_history
        meta_info['finish_called'] = self.finish_call_history
        
        print("ACTIVE_TRAJ_NUM:", active_num_list)
        
        return self._compose_final_output(original_left_side, original_right_side, meta_info)

    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        
        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        
        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        final_output['info_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses_with_info_mask'])
        ], dim=1)
        
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )
        
        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)
        
        return final_output

    def execute_predictions(self, predictions: List[str], active_mask=None, original_indices=None) -> List[str]:
        """
        Execute predictions across multiple environments.
        NOTE: the function is the actual `step` function in the environment
        NOTE penalty_for_invalid is not included in observation shown to the LLM
        
        Args:
            predictions: List of action predictions
            active_mask: Mask indicating which examples are still active
            original_indices: List of original batch indices (for mapping active batch indices back to original)
            
        Returns:
            Tuple of (next_obs, dones, valid_action)
        """
        cur_actions, contents = self.postprocess_predictions(predictions)
        next_obs, dones, valid_action = [], [], []
        
        # Handle None active_mask
        if active_mask is None:
            active_mask = [True] * len(predictions)
        elif isinstance(active_mask, torch.Tensor):
            active_mask = active_mask.tolist()
        
        assert len(predictions) == len(active_mask), f"predictions length {len(predictions)} != active_mask length {len(active_mask)}"
        
        # Map active batch indices to original batch indices
        if original_indices is None:
            original_indices = list(range(len(predictions)))
            
        api_calls = []
        api_indices = []
        for i, (action, content_dict) in enumerate(zip(cur_actions, contents)):
            if action and action != 'Finish' and active_mask[i]:
                original_idx = original_indices[i] if i < len(original_indices) else i
                api_calls.append({
                    'index': i,  # Active batch index (for api_results mapping)
                    'original_index': original_idx,  # Original batch index (for api_success_history)
                    'action': action,
                    'content': content_dict
                })
                api_indices.append(i)
        
        # Batch API calls
        api_results = {}
        if api_calls:
            api_results = self.batch_call_toolbench_apis(api_calls)
        
        # Process results
        for i, (action, content_dict, active) in enumerate(zip(cur_actions, contents, active_mask)):
            original_idx = original_indices[i] if i < len(original_indices) else i
            if not active:
                next_obs.append('')
                dones.append(1)
                valid_action.append(0)
            elif action == 'Finish':
                breakpoint()
                # Use original batch index
                original_idx = original_indices[i] if i < len(original_indices) else i
                if isinstance(content_dict, dict) and 'final_answer' in content_dict:
                    # Track Finish call for reward computation
                    self.finish_call_history[original_idx] = True
                else:
                    # Finish call is invalid
                    self.finish_call_history[original_idx] = False
                next_obs.append('')
                dones.append(1)
                valid_action.append(1)
            elif action and i in api_results:
                # API call succeeded
                result = api_results[i]
                error = result.get('error', '')
                response = result.get('response', '')
                
                # Track API call result for reward computation
                # Use original batch index for api_success_history
                has_error = bool(error and error.strip())
                if original_idx in self.api_success_history:
                    self.api_success_history[original_idx].append(not has_error)
                else:
                    # Initialize if not exists
                    self.api_success_history[original_idx] = [not has_error]
                
                # Format as function response (matching StableToolBench format)
                # In StableToolBench, function response format is: "Observation: {json_string}\n"
                # The JSON string format is: {"error": "", "response": "..."}
                function_response_json = json.dumps({"error": error, "response": response}, ensure_ascii=False)
                # Match StableToolBench format: "Observation: {content}\n" (as in tool_llama_model.py line 117)
                next_obs.append(f"\n\nObservation: {function_response_json}\n\n")
                dones.append(0)
                valid_action.append(1)
            elif action:
                # Invalid API call or parsing error
                next_obs.append('\n\nMy previous action is invalid. Let me check the Action and Action Input format and try again.\n\n')
                dones.append(0)
                valid_action.append(0)
                self.api_success_history[original_idx].append(False)
            else:
                # No action detected
                next_obs.append('\n\nMy previous action is invalid. I should organize my output into three parts: Thought, Action, and Action Input, and in the Action part, I should directly write the name of the API.\n\n')
                dones.append(0)
                valid_action.append(0)
                self.api_success_history[original_idx].append(False)
            
        return next_obs, dones, valid_action

    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[str], List[Dict]]:
        """
        Process (text-based) predictions from llm into actions and contents.
        
        Args:
            predictions: List of raw predictions
            
        Returns:
            Tuple of (actions list, contents list where each content is a dict with action_name and action_input)
        """
        actions = []
        contents = []
        
        for prediction in predictions:
            # Parse ToolBench format: Thought: ...\nAction: ...\nAction Input: ...
            # Thought is optional (?)
            thought_start = prediction.find("Thought: ")
            action_start = prediction.find("Action: ")
            action_input_start = prediction.find("\nAction Input: ")
            
            if thought_start != -1 and action_start != -1 and action_input_start != -1:
                action_name_raw = prediction[action_start + len("Action: "):action_input_start].strip()
                # Normalize API name: convert to lowercase and replace spaces with underscores
                action_name = normalize_api_name(action_name_raw)
                action_input_str = prediction[action_input_start + len("\nAction Input: "):].strip()
                
                # Try to parse JSON - handle both single-line and multi-line JSON
                action_input = {}
                try:
                    # First try direct JSON parsing
                    action_input = json.loads(action_input_str, strict=False)
                except json.JSONDecodeError:
                    # If that fails, try to find the JSON object boundaries
                    # Look for the first { and try to find matching }
                    brace_start = action_input_str.find('{')
                    if brace_start != -1:
                        brace_count = 0
                        brace_end = brace_start
                        for i in range(brace_start, len(action_input_str)):
                            if action_input_str[i] == '{':
                                brace_count += 1
                            elif action_input_str[i] == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    brace_end = i + 1
                                    break
                        if brace_count == 0:
                            try:
                                action_input = json.loads(action_input_str[brace_start:brace_end])
                            except json.JSONDecodeError:
                                # Last resort: try to extract key-value pairs
                                for match in re.finditer(r'"(\w+)":\s*"([^"]*)"', action_input_str):
                                    action_input[match.group(1)] = match.group(2)
                                for match in re.finditer(r'"(\w+)":\s*(\d+)', action_input_str):
                                    action_input[match.group(1)] = int(match.group(2))
                    else:
                        # No JSON object found, try regex extraction
                        for match in re.finditer(r'"(\w+)":\s*"([^"]*)"', action_input_str):
                            action_input[match.group(1)] = match.group(2)
                        for match in re.finditer(r'"(\w+)":\s*(\d+)', action_input_str):
                            action_input[match.group(1)] = int(match.group(2))
                
                actions.append(action_name)
                contents.append({
                    'action_name': action_name,
                    'action_input': action_input
                })
            else:
                actions.append(None)
                contents.append({})
            
        return actions, contents
    
    def _validate_api_call(self, action_name: str, action_input: dict, sample_idx: int) -> Optional[str]:
        """
        Validate API call before sending to server.
        
        Args:
            action_name: API name (format: api_name_for_tool_name)
            action_input: API input parameters
            sample_idx: Sample index (to get API list for this sample)
        
        Returns:
            Error message if validation fails, None if validation passes
        """
        # Get API list for this sample
        if not hasattr(self, 'sample_api_lists') or sample_idx not in self.sample_api_lists:
            # If no API list, skip validation (backward compatibility)
            raise ValueError(f"Missing API list for sample {sample_idx}. Cannot proceed without API validation information.")
        
        # Ensure action_name is normalized
        action_name = normalize_api_name(action_name)
        api_list = self.sample_api_lists[sample_idx]
        
        # Check if API name exists
        if action_name not in api_list:
            return f"Invalid API name: '{action_name}'. Please check the API name."
        
        # Get API validation info (simplified format)
        api_info = api_list[action_name]
        required_count = api_info.get('required_count', 0)
        optional_count = api_info.get('optional_count', 0)
        
        # Check required parameters count
        if not isinstance(action_input, dict):
            return f"Invalid action_input: expected dict, got {type(action_input)}"
        
        provided_param_count = len(action_input)
        
        # Check if required parameters are provided
        if provided_param_count < required_count:
            return f"Missing required parameters: expected at least {required_count}, got {provided_param_count}"
        
        # Check if too many parameters (strict mode: only allow required + optional)
        max_allowed = required_count + optional_count
        if max_allowed > 0 and provided_param_count > max_allowed:
            return f"Too many parameters: expected at most {max_allowed} (required: {required_count}, optional: {optional_count}), got {provided_param_count}"
        
        # Validation passed
        return None
    
    def batch_call_toolbench_apis(self, api_calls: List[Dict]) -> Dict[int, Dict]:
        """
        Batch call ToolBench API server for multiple API calls concurrently.
        
        Args:
            api_calls: List of dicts with 'index', 'action', and 'content' keys
            
        Returns:
            Dict mapping example index to API response
        """
        # Use asyncio to run concurrent API calls
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # If loop is closed or unusable, create a new one
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self._batch_call_toolbench_apis_async(api_calls))
    
    async def _batch_call_toolbench_apis_async(self, api_calls: List[Dict]) -> Dict[int, Dict]:
        """
        Async implementation of batch API calls for concurrent execution.
        
        Args:
            api_calls: List of dicts with 'index', 'action', and 'content' keys
            
        Returns:
            Dict mapping example index to API response
        """
        results = {}
        
        # Prepare all API calls first (validation and payload preparation)
        async_tasks = []
        task_indices = []  # Store index for each task in the same order
        
        for api_call in api_calls:
            idx = api_call['index']
            action_name_raw = api_call['action']
            # Normalize API name: convert to lowercase and replace spaces with underscores
            action_name = normalize_api_name(action_name_raw)
            content_dict = api_call['content']
            action_input = content_dict.get('action_input', {})
            
            # Extract tool_name and api_name from action_name
            # Format in StableToolBench: api_name_for_tool_name (normalized)
            # Example: get_company_data_by_linkedin_url_for_fresh_linkedin_profile_data
            #   -> api_name: get_company_data_by_linkedin_url
            #   -> tool_name: fresh_linkedin_profile_data
            if '_for_' in action_name:
                # Split on '_for_' - the last part is the tool_name
                parts = action_name.rsplit('_for_', 1)
                if len(parts) == 2:
                    tool_name = parts[1]
                else:
                    parts = action_name.split('_for_', 1)
                    tool_name = parts[1] if len(parts) > 1 else 'unknown'
            else:
                tool_name = 'unknown'
            
            # Extract category from sample's extra_info
            # Map original_index to category
            original_idx = api_call.get('original_index', idx)
            if not hasattr(self, 'sample_categories') or original_idx not in self.sample_categories:
                raise ValueError(f'Missing category for sample {original_idx}.')
            
            category = self.sample_categories[original_idx]
            
            # Validate API call before sending to server
            validation_error = self._validate_api_call(action_name, action_input, original_idx)
            if validation_error:
                results[idx] = {
                    'error': validation_error,
                    'response': ''
                }
                continue
            
            # Prepare payload
            payload = {
                'category': category,
                'tool_name': tool_name,
                'api_name': action_name,  # Use full action_name as api_name for ToolBench
                'tool_input': action_input,
                'strip': '',
            }
            
            # Create async task for this API call
            task = self._call_toolbench_api_async(
                url=f"{self.config.toolbench_url}/virtual",
                payload=payload,
                action_name=action_name,
                tool_name=tool_name,
                category=category,
                action_input=action_input
            )
            async_tasks.append(task)
            task_indices.append(idx)  # Store index in the same order as tasks
        
        # Execute all API calls concurrently
        if async_tasks:
            # return_exceptions=True 确保一个请求崩了不会导致整个 batch 失败
            responses = await asyncio.gather(*async_tasks, return_exceptions=True)
            
            for idx, response in zip(task_indices, responses):
                if isinstance(response, Exception):
                    results[idx] = {
                        'error': f'API call internal error: {str(response)}',
                        'response': ''
                    }
                    print(f"[ERROR] API call exception: {str(response)}")
                else:
                    results[idx] = response
        
        return results
    
    async def _call_toolbench_api_async(
        self, 
        url: str, 
        payload: Dict, 
        action_name: str,
        tool_name: str,
        category: str,
        action_input: Dict
    ) -> Dict:
        """
        使用 self.async_client 发送请求，不再创建新连接。
        使用 Semaphore 控制并发数量，防止内存 OOM。
        """
        # 使用 semaphore 限制并发数量
        async with self.api_call_semaphore:
            try:
                # 直接使用初始化好的 client
                response = await self.async_client.post(url, json=payload)
                
                if response.status_code == 200:
                    return response.json()
                else:
                    # 只有出错时才 print，减少 I/O
                    # print(f"[DEBUG] Error {response.status_code} for {action_name}")
                    return {
                        'error': f'API call failed with status {response.status_code}',
                        'response': ''
                    }
            except Exception as e:
                # print(f"[DEBUG] Exception for {action_name}: {str(e)}")
                return {
                    'error': f'API call error: {str(e)}',
                    'response': ''
                }
