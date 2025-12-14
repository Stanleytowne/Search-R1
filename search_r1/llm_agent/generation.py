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
    use_toolbench: bool = False
    toolbench_url: str = None
    toolbench_key: str = ""

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
        
        # Cache for API information extracted from prompts
        self.api_info_cache = {}
        
        # Track API calls and Finish calls for reward computation
        # Format: {sample_idx: [list of (error, success)]}
        self.api_call_history = {}
        # Format: {sample_idx: 'give_answer' | 'give_up' | None}
        self.finish_call_history = {}

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']

    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        """Process responses to stop at search operation or answer operation or function call."""
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )

        if self.config.use_toolbench:
            # For ToolBench format, stop at Action Input (complete JSON) or end of response
            # Format: Thought: ...\nAction: ...\nAction Input: {...}
            processed_responses = []
            for resp in responses_str:
                # Remove eos_token if present at the end of the sequence
                eos_token = self.tokenizer.eos_token if hasattr(self.tokenizer, 'eos_token') and self.tokenizer.eos_token is not None else '</s>'
                resp = resp.rstrip()
                if resp.endswith(eos_token):
                    resp = resp[:-len(eos_token)].rstrip()
                
                # Check if there's a complete Action Input (function call)
                # Look for "Action Input:" followed by JSON object
                action_input_start = resp.find('Action Input:')
                if action_input_start != -1:
                    # Find the JSON object after "Action Input:"
                    json_start = resp.find('{', action_input_start)
                    if json_start != -1:
                        # Try to find the matching closing brace
                        brace_count = 0
                        json_end = json_start
                        for i in range(json_start, len(resp)):
                            if resp[i] == '{':
                                brace_count += 1
                            elif resp[i] == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    json_end = i + 1
                                    break
                        if brace_count == 0:
                            # Found complete JSON, extract up to here (including the closing brace)
                            processed_responses.append(resp[:json_end])
                        else:
                            # Incomplete JSON, keep original (but remove </s>)
                            processed_responses.append(resp)
                    else:
                        processed_responses.append(resp)
                else:
                    # No Action Input found, keep original response (but remove </s>)
                    processed_responses.append(resp)
            responses_str = processed_responses
        else:
            # Original Search-R1 format
            responses_str = [resp.split('</search>')[0] + '</search>'
                     if '</search>' in resp 
                     else resp.split('</answer>')[0] + '</answer>'
                     if '</answer>' in resp 
                     else resp
                     for resp in responses_str]

        if self.config.no_think_rl:
            raise ValueError('stop')
            # if no_think_rl is enabled, only keep action in the str
            actions, _ = self.env.postprocess_predictions(responses_str)
            responses_str=[f"<answer>{envs[idx].ACTION_LOOKUP[action]}</answer>" for idx, action in enumerate(actions)]
            print("RESPONSES:", responses_str)
        responses = self._batch_tokenize(responses_str)
        return responses, responses_str

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """Process next observations from environment."""
        
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

        # INSERT_YOUR_CODE
        # 打印第一条数据的input
        if gen_batch.batch['input_ids'].shape[0] > 0:
            first_input_ids = gen_batch.batch['input_ids'][0]
            input_text = self.tokenizer.decode(first_input_ids, skip_special_tokens=True)
            print("[DEBUG] First sample input:")
            print(input_text)
        
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {'responses': initial_input_ids[:, []], 'responses_with_info_mask': initial_input_ids[:, []]}
        
        active_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)
        turns_stats = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_action_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_search_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch
        
        if self.config.use_toolbench:
            batch_size = gen_batch.batch['input_ids'].shape[0]
            self.api_call_history = {i: [] for i in range(batch_size)}
            self.finish_call_history = {i: None for i in range(batch_size)}
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

        # Main generation loop
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
            next_obs, dones, valid_action, is_search = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask, original_indices=active_indices
            )
            
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            turns_stats[curr_active_mask] += 1
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)

            next_obs_ids = self._process_next_obs(next_obs)
            
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
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # # Execute in environment and process observations
            # Map active indices back to original batch indices
            active_indices_final = torch.where(active_mask)[0].tolist() if isinstance(active_mask, torch.Tensor) else [i for i, active in enumerate(active_mask) if active]
            _, dones, valid_action, is_search = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask, do_search=False, original_indices=active_indices_final
            )

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)
            

            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
            )
        
        meta_info['turns_stats'] = turns_stats.tolist()
        meta_info['active_mask'] = active_mask.tolist()
        meta_info['valid_action_stats'] = valid_action_stats.tolist()
        meta_info['valid_search_stats'] = valid_search_stats.tolist()
        
        # Add ToolBench reward computation info
        if self.config.use_toolbench:
            print(f"[DEBUG run_llm_loop] Finalizing: api_call_history has {len(self.api_call_history)} entries")
            print(f"[DEBUG run_llm_loop] api_call_history keys: {list(self.api_call_history.keys())[:10]}...")  # Show first 10
            meta_info['api_errors'] = {
                i: [error for error in errors] 
                for i, errors in self.api_call_history.items()
            }
            meta_info['finish_called'] = self.finish_call_history.copy()
            print(f"[DEBUG run_llm_loop] Added to meta_info: api_errors has {len(meta_info['api_errors'])} entries")
        
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
        
        # Debug: Print full conversation for first sample
        if final_output.batch['input_ids'].shape[0] > 0:
            self._debug_print_full_conversation(final_output, 0)
        
        return final_output

    def _debug_print_full_conversation(self, final_output: DataProto, sample_idx: int = 0):
        """Print full conversation for debugging purposes."""
        try:
            # Get prompt (initial input)
            prompt_ids = final_output.batch['prompts'][sample_idx]
            prompt_mask = final_output.batch['attention_mask'][sample_idx, :prompt_ids.shape[0]]
            valid_prompt_ids = prompt_ids[prompt_mask.bool()]
            
            # Get full input_ids (prompt + responses)
            input_ids = final_output.batch['input_ids'][sample_idx]
            attention_mask = final_output.batch['attention_mask'][sample_idx]
            valid_input_ids = input_ids[attention_mask.bool()]
            
            # Decode prompt
            prompt_text = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)
            
            # Decode full sequence
            full_text = self.tokenizer.decode(valid_input_ids, skip_special_tokens=False)
            
            # Extract response part (everything after prompt)
            # Find where prompt ends in the full text
            prompt_len = len(prompt_text)
            if len(full_text) > prompt_len:
                response_text = full_text[prompt_len:].lstrip()
            else:
                response_text = ""
            
            # Get metadata if available
            turns = final_output.meta_info.get('turns_stats', [0])[sample_idx] if 'turns_stats' in final_output.meta_info else 0
            valid_actions = final_output.meta_info.get('valid_action_stats', [0])[sample_idx] if 'valid_action_stats' in final_output.meta_info else 0
            
            # Print formatted conversation
            print("\n" + "=" * 100)
            print(f"[DEBUG] FULL CONVERSATION - Sample {sample_idx} (Turns: {turns}, Valid Actions: {valid_actions})")
            print("=" * 100)
            print("\n[PROMPT (Initial Context)]")
            print("-" * 100)
            print(prompt_text)
            print("\n[RESPONSES (Generated Content)]")
            print("-" * 100)
            if response_text:
                print(response_text)
            else:
                print("(No response generated)")
            print("\n" + "=" * 100 + "\n")
        except Exception as e:
            print(f"[DEBUG] Error printing full conversation: {e}")
            import traceback
            traceback.print_exc()

    def execute_predictions(self, predictions: List[str], pad_token: str, active_mask=None, do_search=True, original_indices=None) -> List[str]:
        """
        Execute predictions across multiple environments.
        NOTE: the function is the actual `step` function in the environment
        NOTE penalty_for_invalid is not included in observation shown to the LLM
        
        Args:
            predictions: List of action predictions
            pad_token: Token to use for padding
            active_mask: Mask indicating which examples are still active
            do_search: Whether to actually execute search/API calls
            original_indices: List of original batch indices (for mapping active batch indices back to original)
            
        Returns:
            Tuple of (next_obs, dones, valid_action, is_search)
        """
        cur_actions, contents = self.postprocess_predictions(predictions)
        next_obs, dones, valid_action, is_search = [], [], [], []
        
        # Handle None active_mask
        if active_mask is None:
            active_mask = [True] * len(predictions)
        elif isinstance(active_mask, torch.Tensor):
            active_mask = active_mask.tolist()
        
        # Map active batch indices to original batch indices
        if original_indices is None:
            original_indices = list(range(len(predictions)))
        
        if self.config.use_toolbench:
            
            api_calls = []
            api_indices = []
            for i, (action, content_dict) in enumerate(zip(cur_actions, contents)):
                if action and action != 'Finish' and active_mask[i]:
                    original_idx = original_indices[i] if i < len(original_indices) else i
                    api_calls.append({
                        'index': i,  # Active batch index (for api_results mapping)
                        'original_index': original_idx,  # Original batch index (for api_call_history)
                        'action': action,
                        'content': content_dict
                    })
                    api_indices.append(i)
            
            # Batch API calls
            api_results = {}
            if api_calls:
                api_results = self.batch_call_toolbench_apis(api_calls)
            elif not api_calls:
                print(f"[DEBUG] No API calls to make (api_calls is empty)")
            
            # Process results
            api_result_idx = 0
            for i, (action, content_dict, active) in enumerate(zip(cur_actions, contents, active_mask)):
                if not active:
                    next_obs.append('')
                    dones.append(1)
                    valid_action.append(0)
                    is_search.append(0)
                elif action == 'Finish':
                    # Check if it's give_answer or give_up
                    if isinstance(content_dict, dict) and 'action_input' in content_dict:
                        return_type = content_dict['action_input'].get('return_type', 'give_answer')
                        # Track Finish call for reward computation
                        # Use original batch index
                        original_idx = original_indices[i] if i < len(original_indices) else i
                        if original_idx in self.finish_call_history:
                            self.finish_call_history[original_idx] = return_type
                        else:
                            self.finish_call_history[original_idx] = return_type
                        next_obs.append('')
                        dones.append(1)
                        valid_action.append(1)
                        is_search.append(0)
                    else:
                        next_obs.append('')
                        dones.append(1)
                        valid_action.append(1)
                        is_search.append(0)
                elif action and i in api_results:
                    # API call succeeded
                    result = api_results[i]
                    error = result.get('error', '')
                    response = result.get('response', '')
                    
                    # Track API call result for reward computation
                    # Use original batch index for api_call_history
                    original_idx = original_indices[i] if i < len(original_indices) else i
                    has_error = bool(error and error.strip())
                    if original_idx in self.api_call_history:
                        self.api_call_history[original_idx].append(has_error)
                    else:
                        # Initialize if not exists
                        self.api_call_history[original_idx] = [has_error]
                    
                    # Format as function response (matching StableToolBench format)
                    # In StableToolBench, function response format is: "Observation: {json_string}\n"
                    # The JSON string format is: {"error": "", "response": "..."}
                    function_response_json = json.dumps({"error": error, "response": response}, ensure_ascii=False)
                    # Match StableToolBench format: "Observation: {content}\n" (as in tool_llama_model.py line 117)
                    next_obs.append(f"\n\nObservation: {function_response_json}\n\n")
                    dones.append(0)
                    valid_action.append(1)
                    is_search.append(1)  # Treat API calls as search-like operations
                elif action:
                    # Invalid API call or parsing error
                    next_obs.append('\nMy previous action is invalid. Let me check the Action and Action Input format and try again.\n')
                    dones.append(0)
                    valid_action.append(0)
                    is_search.append(0)
                else:
                    # No action detected
                    next_obs.append('\nMy previous action is invalid. I should provide Thought, Action, and Action Input.\n')
                    dones.append(0)
                    valid_action.append(0)
                    is_search.append(0)
        else:
            # Original Search-R1 mode
            search_queries = [content.get('content', '') if isinstance(content, dict) else content 
                            for action, content in zip(cur_actions, contents) if action == 'search']
            if do_search:
                search_results = self.batch_search(search_queries)
                assert len(search_results) == sum([1 for action in cur_actions if action == 'search'])
            else:
                search_results = [''] * sum([1 for action in cur_actions if action == 'search'])

            for i, (action, content_dict, active) in enumerate(zip(cur_actions, contents, active_mask)):
                if not active:
                    next_obs.append('')
                    dones.append(1)
                    valid_action.append(0)
                    is_search.append(0)
                else:
                    if action == 'answer':
                        next_obs.append('')
                        dones.append(1)
                        valid_action.append(1)
                        is_search.append(0)
                    elif action == 'search':
                        content = content_dict.get('content', '') if isinstance(content_dict, dict) else content_dict
                        next_obs.append(f'\n\n<information>{search_results.pop(0).strip()}</information>\n\n')
                        dones.append(0)
                        valid_action.append(1)
                        is_search.append(1)
                    else:
                        next_obs.append(f'\nMy previous action is invalid. \
If I want to search, I should put the query between <search> and </search>. \
If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n')
                        dones.append(0)
                        valid_action.append(0)
                        is_search.append(0)
            
            assert len(search_results) == 0
            
        return next_obs, dones, valid_action, is_search

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
                
        for idx, prediction in enumerate(predictions):
            if isinstance(prediction, str): # for llm output
                if self.config.use_toolbench:
                    # Parse ToolBench format: Thought: ...\nAction: ...\nAction Input: ...
                    thought_start = prediction.find("Thought: ")
                    action_start = prediction.find("\nAction: ")
                    action_input_start = prediction.find("\nAction Input: ")
                    
                    if thought_start != -1 and action_start != -1 and action_input_start != -1:
                        action_name_raw = prediction[action_start + len("\nAction: "):action_input_start].strip()
                        # Normalize API name: convert to lowercase and replace spaces with underscores
                        action_name = normalize_api_name(action_name_raw)
                        action_input_str = prediction[action_input_start + len("\nAction Input: "):].strip()
                        
                        # Try to parse JSON - handle both single-line and multi-line JSON
                        action_input = {}
                        try:
                            # First try direct JSON parsing
                            action_input = json.loads(action_input_str)
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
                else:
                    # Original Search-R1 format
                    pattern = r'<(search|answer)>(.*?)</\1>'
                    match = re.search(pattern, prediction, re.DOTALL)
                    if match:
                        content = match.group(2).strip()  # Return only the content inside the tags
                        action = match.group(1)
                        actions.append(action)
                        contents.append({'action': action, 'content': content})
                    else:
                        content = ''
                        action = None
                        actions.append(action)
                        contents.append({})
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")
            
        return actions, contents

    def batch_search(self, queries: List[str] = None) -> str:
        """
        Batchified search for queries.
        Args:
            queries: queries to call the search engine
        Returns:
            search results which is concatenated into a string
        """
        results = self._batch_search(queries)['result']
        
        return [self._passages2string(result) for result in results]

    def _batch_search(self, queries):
        
        payload = {
            "queries": queries,
            "topk": self.config.topk,
            "return_scores": True
        }
        
        return requests.post(self.config.search_url, json=payload).json()

    def _passages2string(self, retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

        return format_reference
    
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
            return None
        
        # Ensure action_name is normalized
        action_name = normalize_api_name(action_name)
        api_list = self.sample_api_lists[sample_idx]
        
        # Check if API name exists
        if action_name not in api_list:
            available_apis = list(api_list.keys())
            return f"Invalid API name: '{action_name}'. Available APIs: {available_apis[:5]}..."
        
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
        Batch call ToolBench API server for multiple API calls.
        
        Args:
            api_calls: List of dicts with 'index', 'action', and 'content' keys
            
        Returns:
            Dict mapping example index to API response
        """
        results = {}
        
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
                    api_name = parts[0]
                    tool_name = parts[1]
                else:
                    # Fallback: split on first occurrence
                    parts = action_name.split('_for_', 1)
                    api_name = parts[0]
                    tool_name = parts[1] if len(parts) > 1 else 'unknown'
            else:
                # No '_for_' in name, use the whole name as api_name
                api_name = action_name
                tool_name = 'unknown'
            
            # Extract category from sample's extra_info
            # Map original_index to category
            original_idx = api_call.get('original_index', idx)
            if not hasattr(self, 'sample_categories') or original_idx not in self.sample_categories:
                raise ValueError(f"Missing category for sample {original_idx}. Category must be provided in extra_info.")
            category = self.sample_categories[original_idx]
            
            # Validate API call before sending to server
            validation_error = self._validate_api_call(action_name, action_input, original_idx)
            if validation_error:
                results[idx] = {
                    'error': validation_error,
                    'response': ''
                }
                print(f"[DEBUG] API call validation failed for action: {action_name} (tool: {tool_name}, category: {category}): {validation_error}\n")
                continue
            
            # Call ToolBench server
            try:
                payload = {
                    'category': category,
                    'tool_name': tool_name,
                    'api_name': action_name,  # Use full action_name as api_name for ToolBench
                    'tool_input': action_input,
                    'strip': '',
                    'toolbench_key': self.config.toolbench_key
                }

                response = requests.post(
                    f"{self.config.toolbench_url}/virtual",
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    results[idx] = response.json()
                else:
                    results[idx] = {
                        'error': f'API call failed with status {response.status_code}',
                        'response': ''
                    }
                    print(f"[DEBUG] Calling ToolBench API {action_name} (tool: {tool_name}, category: {category}) with action input: {action_input} error with response status: {response.status_code}\n")
            except Exception as e:
                results[idx] = {
                    'error': f'API call error: {str(e)}',
                    'response': ''
                }
                print(f"[DEBUG] API call exception: {str(e)}")
        
        return results
