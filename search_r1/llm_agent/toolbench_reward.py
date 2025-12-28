"""
ToolBench模式的Reward计算
包括：
1. 格式奖励：奖励模型生成正确的格式（Thought/Action/Action Input）
2. Function call正确奖励：如果API调用结果有error，则惩罚
3. Finish调用奖励：最后一次是否调用Finish
"""

import torch
import re
import json
from typing import List, Dict
import requests
from verl import DataProto


class ToolBenchRewardManager:
    """ToolBench模式的Reward管理器"""
    
    def __init__(
        self,
        tokenizer,
        num_examine: int = 0,
        reward_server_url: str = "http://localhost:8000/evaluate_batch"
    ):
        """
        Args:
            tokenizer: Tokenizer用于解码
            num_examine: 打印的样本数量
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_server_url = reward_server_url
    
    def __call__(self, data: DataProto) -> torch.Tensor:
        """
        计算ToolBench模式的reward
        
        Args:
            data: DataProto包含生成的数据和meta_info
            
        Returns:
            token_level_rewards: (batch_size, response_length)的reward tensor
        """
        # if 'rm_scores' in data.batch.keys(), return the rm_scores
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']
        
        batch_size = data.batch['responses'].shape[0]
        response_length = data.batch['responses'].shape[1]
        
        # init reward tensor
        reward_tensor = torch.zeros((batch_size, response_length), dtype=torch.float32)
        
        # get ToolBench related information from meta_info
        meta_info = data.meta_info

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
            assert len(each_turn_end_loc[i]) == meta_info['turns_stats'][i], f"Sample {i} has turns stats as {meta_info['turns_stats'][i]} and each turn end loc as {each_turn_end_loc[i]}"

            if i < self.num_examine:
                print(f"\n{'='*20} [DEBUG REWARD LOC] Sample {i} {'='*20}")
                print(f"Calculated Indices: {each_turn_end_loc[i]}")
                
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
                    
                    # 简单处理换行符，防止打印混乱
                    token_str_repr = token_str.replace('\n', '\\n')
                    
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
        data.meta_info['pass_rewards'] = pass_rewards

        if data[0].non_tensor_batch['data_source'] == 'toolbench-eval':
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
        format_and_function_call_reward = self._compute_format_and_function_call_reward(meta_info)
        # 2. finish reward for the final turn
        finish_reward = self._compute_finish_reward(meta_info)
        data.meta_info['format_and_function_call_reward'] = format_and_function_call_reward
        data.meta_info['finish_reward'] = finish_reward

        for i in range(batch_size):
            for j in range(len(each_turn_end_loc[i]) - 1):
                reward_tensor[i, each_turn_end_loc[i][j]] = format_and_function_call_reward[i][j]
            reward_tensor[i, each_turn_end_loc[i][-1]] = finish_reward[i] + pass_rewards[i]
            
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

    def _compute_format_and_function_call_reward(self, meta_info: Dict) -> List[List[float]]:
        turns_stats = meta_info['turns_stats']
        valid_action_stats = meta_info['valid_action_stats']
        valid_api_call_stats = meta_info['valid_api_call_stats']
        batch_size = len(turns_stats)
        format_rewards = [[] for _ in range(batch_size)]
        
        for i in range(batch_size):
            for j in range(turns_stats[i] - 1):
                if valid_action_stats[i][j] and valid_api_call_stats[i][j]:
                    format_rewards[i].append(0.1)
                elif valid_action_stats[i][j] and not valid_api_call_stats[i][j]:
                    format_rewards[i].append(-0.1)
                else:
                    format_rewards[i].append(-0.2)
        
        return format_rewards

    def _compute_finish_reward(self, meta_info: Dict) -> List[float]:
        """
        Compute finish reward
        Args:
            sample_idx: sample index
            meta_info: meta information
        Returns:
            finish reward for each sample
        """
        finish_called = meta_info['finish_called']
        finish_rewards = []
        for i in range(len(finish_called)):
            if finish_called[i]:
                finish_rewards.append(0.2)
            else:
                finish_rewards.append(-0.5)
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
