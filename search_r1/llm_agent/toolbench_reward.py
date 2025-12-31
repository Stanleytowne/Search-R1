"""
ToolBench reward manager
1. format and function call reward for each turn (excluding the final turn)
2. finish reward for the final turn
3. pass reward for the final turn
"""

import torch
import re
import json
from typing import List, Dict
import requests
from verl import DataProto


class ToolBenchRewardManager:

    def __init__(
        self,
        tokenizer,
        num_examine: int = 0,
        reward_server_url: str = "http://localhost:8000/evaluate_batch"
    ):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_server_url = reward_server_url

    def _validate_turn_consistency(
        self,
        sample_idx: int,
        n_turns_from_info_mask: int,
        response_str: str,
        turns_stats_tensor: torch.Tensor,
    ):
        """
        验证 turn 数是否一致：
        - info_mask 解析得到的 n_turns（模型段落数）
        - generation 端记录的 turns_stats（每次 env step 统计的 turn 数；最后 forced finish 时可能少 1）
        - valid_*_stats_len（每次 execute_predictions 都会 append，包含最终 forced finish）
        """
        turns_stats_i = int(turns_stats_tensor[sample_idx].item())

        # turns_stats is expected to be either equal to executed turns, or 1 smaller when a final forced-finish
        # rollout happened (because turns_stats isn't incremented in the final rollout block).
        ok = (n_turns_from_info_mask == turns_stats_i)
        if not ok:
            msg = (
                f"[TURN CHECK FAILED] sample={sample_idx} "
                f"info_mask_turns={n_turns_from_info_mask}, "
                f"turns_stats={turns_stats_i}, "
                f"Problems: info_mask_turns({n_turns_from_info_mask}) not in {{turns_stats({turns_stats_i}), turns_stats+1}}\n"
                f"response: {response_str}"
            )
            print(msg)

    def __call__(self, data: DataProto) -> torch.Tensor:
        # if 'rm_scores' in data.batch.keys(), return the rm_scores
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']
        
        batch_size = data.batch['responses'].shape[0]
        response_length = data.batch['responses'].shape[1]
        # init reward tensor
        reward_tensor = torch.zeros((batch_size, response_length), dtype=torch.float32)

        # Per-sample bookkeeping must come from batch (reliably aligned).
        active_mask_tensor = data.batch.get('active_mask', None)
        turns_stats_tensor = data.batch.get('turns_stats', None)
        valid_action_stats_tensor = data.batch.get('valid_action_stats', None)
        valid_api_call_stats_tensor = data.batch.get('valid_api_call_stats', None)

        # get all queries, trajectories and each turn end locations
        all_queries = []
        all_trajectories = []
        each_turn_end_loc = [[] for _ in range(batch_size)]
        for i in range(batch_size):
            data_item = data[i]

            response_ids = data_item.batch['responses']
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            attention_mask = data_item.batch['attention_mask']

            valid_prompt_length = attention_mask[:prompt_length].sum().item()
            valid_response_length = attention_mask[prompt_length:].sum().item()
            
            valid_prompt_ids = prompt_ids[-valid_prompt_length:] if valid_prompt_length > 0 else prompt_ids
            valid_response_ids = response_ids[:valid_response_length] if valid_response_length > 0 else response_ids

            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)
            query_str = self._extract_query(prompt_str)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)
            
            all_queries.append(query_str)
            all_trajectories.append(response_str)

            # get the end location of each turn
            full_info_mask = data_item.batch['info_mask']
            response_mask = full_info_mask[prompt_length : prompt_length + valid_response_length]
            
            mask_list = response_mask.tolist()
            turn_indices = []
            
            for t, is_model_token in enumerate(mask_list):
                if is_model_token == 1 and (t == len(mask_list) - 1 or mask_list[t + 1] == 0):
                    turn_indices.append(t)
            
            each_turn_end_loc[i] = turn_indices

            # turn consistency check: info_mask vs stats recorded during generation/env execution
            self._validate_turn_consistency(
                sample_idx=i,
                n_turns_from_info_mask=len(turn_indices),
                response_str=response_str,
                turns_stats_tensor=turns_stats_tensor,
            )

            if i < self.num_examine:
                print(f"\n{'='*20} [DEBUG REWARD LOC] Sample {i} {'='*20}")
                print(f"Calculated Indices: {each_turn_end_loc[i]}")
                print(f"Turn stats: {turns_stats_tensor[i]}")
                print(f"Valid action stats: {valid_action_stats_tensor[i]}")
                print(f"Valid api call stats: {valid_api_call_stats_tensor[i]}")
                
                # 获取用于显示的 token ID 和 mask
                # 注意：这里我们使用 valid_response_ids，确保只打印有效部分
                debug_tokens = valid_response_ids.tolist()
                debug_mask = mask_list  # 沿用上面计算出的 list
                
                print("\n[Visualized Response Flow]")
                print("Legend: [M] = Model Token (Mask=1), [E] = Env Token (Mask=0), 📍 = Reward Location")
                print("-" * 60)
                
                # 逐个 Token 还原并打印，遇到关键位置换行或标记
                buffer_str = ""
                current_type = debug_mask[0] if len(debug_mask) > 0 else 1
                
                for idx, (tid, is_model) in enumerate(zip(debug_tokens, debug_mask)):
                    token_str = self.tokenizer.decode([tid], skip_special_tokens=False)
                    
                    # 标记是否是 Reward 位置
                    is_reward_loc = idx in each_turn_end_loc[i]
                    
                    # 如果 mask 类型发生变化（从模型->环境 或 环境->模型），先打印之前的 buffer
                    if is_model != current_type:
                        prefix = "[Model]: " if current_type == 1 else "[Env]:   "
                        print(f"{prefix}{buffer_str}")
                        buffer_str = ""
                        current_type = is_model
                    
                    # 拼接到 buffer
                    buffer_str += token_str
                    
                    # 如果这里是 Reward 位置，插入显眼标记
                    if is_reward_loc:
                        buffer_str += " [📍REWARD] "
                
                # 打印剩余的 buffer
                if buffer_str:
                    prefix = "[Model]: " if current_type == 1 else "[Env]:   "
                    print(f"{prefix}{buffer_str}")
                
                print("="*60 + "\n")
        
        pass_rewards = self._get_remote_pass_rewards(all_queries, all_trajectories)
        pass_reward_tensor = torch.tensor(pass_rewards, dtype=torch.float32)
        data.batch['pass_reward'] = pass_reward_tensor

        if data[0].non_tensor_batch['data_source'] == 'toolbench-eval':
            print("Using toolbench-eval reward")
            for i in range(batch_size):
                last_turn_end_loc = each_turn_end_loc[i][-1]
                reward_tensor[i, last_turn_end_loc] = pass_rewards[i]
                
                if i < self.num_examine:
                    response_str = all_trajectories[i]
                    print(f"\n[Eval Reward Sample {i}]")
                    print(f"  Response: {response_str[:200]}...")
                    print(f"  Pass reward: {pass_rewards[i]:.3f}")
            return reward_tensor

        # 1. format and function call reward for each turn (excluding the final turn)
        format_and_function_call_reward = self._compute_format_and_function_call_reward(
            each_turn_end_loc=each_turn_end_loc,
            valid_action_stats=valid_action_stats_tensor,
            valid_api_call_stats=valid_api_call_stats_tensor,
        )
        # 2. finish reward for the final turn
        finish_reward = self._compute_finish_reward(active_mask_tensor, batch_size=batch_size)
        data.batch['finish_reward'] = torch.tensor(finish_reward, dtype=torch.float32)
        # optional: sum format reward per sample for metrics
        data.batch['format_reward_sum'] = torch.tensor([float(sum(x)) for x in format_and_function_call_reward], dtype=torch.float32)
        # 3. penalty for direct call of Finish
        finish_penalty = torch.where(turns_stats_tensor == 1, -10, 0.0)
        data.batch['finish_penalty'] = torch.tensor(finish_penalty, dtype=torch.float32)

        for i in range(batch_size):
            for j in range(len(each_turn_end_loc[i]) - 1):
                reward_tensor[i, each_turn_end_loc[i][j]] = format_and_function_call_reward[i][j]
            reward_tensor[i, each_turn_end_loc[i][-1]] = finish_reward[i] + pass_rewards[i] * 5 + finish_penalty[i]
            
            if i < self.num_examine:
                print(f"\n[Reward Sample {i}]")
                print(f"  Response: {all_trajectories[i][:200]}...")
                print(f"  Pass reward: {pass_rewards[i]}")
                print(f"  Format reward: {format_and_function_call_reward[i]}")
                print(f"  Finish reward: {finish_reward[i]}")
        
        return reward_tensor
    
    def _extract_query(self, full_prompt: str) -> str:
        import re
        # 修改：严格提取<|im_start|>user\n和<|im_end|>之间的内容
        pattern = r'<\|im_start\|>user\n(.*?)<\|im_end\|>'
        matches = re.findall(pattern, full_prompt, re.DOTALL)
        if matches:
            # 如果有多个，取最后一个
            query = matches[-1].strip()
            return query
        return full_prompt.strip()

    def _compute_format_and_function_call_reward(
        self,
        each_turn_end_loc: List[List[int]],
        valid_action_stats: torch.Tensor,
        valid_api_call_stats: torch.Tensor,
    ) -> List[List[float]]:
        """
        Compute per-turn format reward (excluding the final turn), aligned to the
        *turns visible in `info_mask`* (i.e., `each_turn_end_loc`).
        """
        batch_size = len(each_turn_end_loc)
        format_rewards: List[List[float]] = [[] for _ in range(batch_size)]

        va = valid_action_stats.detach().cpu().to(torch.int64)
        vc = valid_api_call_stats.detach().cpu().to(torch.int64)

        for i in range(batch_size):
            n_turns = len(each_turn_end_loc[i])
            n_reward_turns = max(0, n_turns - 1)
            for j in range(n_reward_turns):
                v_a = int(va[i, j].item())
                v_c = int(vc[i, j].item())
                if v_a == 1 and v_c == 1:
                    format_rewards[i].append(0.4)
                elif v_a == 1 and v_c == 0:
                    format_rewards[i].append(-0.1)
                else:
                    format_rewards[i].append(-0.2)

        return format_rewards

    def _compute_finish_reward(self, active_mask: torch.Tensor, batch_size: int, device=None) -> List[float]:
        """
        Compute finish reward
        Args:
            sample_idx: sample index
            active_mask: per-sample active mask, True means NOT finished.
        Returns:
            finish reward for each sample
        """
        if active_mask is None:
            # If unknown, be conservative: treat as not finished.
            active_mask = torch.ones((batch_size,), dtype=torch.bool, device=device)
        finish_rewards = []
        active_mask_list = active_mask.detach().cpu().tolist()
        for i in range(len(active_mask_list)):
            if not active_mask_list[i]:
                finish_rewards.append(0.)
            else:
                finish_rewards.append(-0.2)
        return finish_rewards

    def _get_remote_pass_rewards(self, queries: List[str], trajectories: List[str]) -> List[float]:
        """通过 HTTP 调用远程 Reward Server"""
        payload = {
            "queries": queries,
            "trajectories": trajectories
        }
        try:
            response = requests.post(self.reward_server_url, json=payload, timeout=60)
            if response.status_code == 200:
                return response.json().get("scores", [0.5] * len(queries))
            else:
                print(f"Remote server error: {response.status_code}")
                return [0.0] * len(queries)
        except Exception as e:
            print(f"Failed to connect to reward server: {e}")
            return [0.0] * len(queries)



def create_toolbench_reward_manager(
    tokenizer,
    **kwargs
) -> ToolBenchRewardManager:
    return ToolBenchRewardManager(
        tokenizer=tokenizer,
        **kwargs
    )
