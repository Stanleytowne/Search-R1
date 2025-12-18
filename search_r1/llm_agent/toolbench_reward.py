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
from verl import DataProto


class ToolBenchRewardManager:
    """ToolBench模式的Reward管理器"""
    
    def __init__(
        self,
        tokenizer,
        format_reward_weight: float = 0.1,
        function_call_reward_weight: float = 0.2,
        finish_reward_weight: float = 0.3,
        error_penalty: float = -0.5,
        finish_bonus: float = 0.5,
        num_examine: int = 0
    ):
        """
        Args:
            tokenizer: Tokenizer用于解码
            format_reward_weight: 格式奖励权重
            function_call_reward_weight: Function call奖励权重
            finish_reward_weight: Finish调用奖励权重
            error_penalty: API调用错误的惩罚
            finish_bonus: 正确调用Finish的奖励
            num_examine: 打印的样本数量
        """
        self.tokenizer = tokenizer
        self.format_reward_weight = format_reward_weight
        self.function_call_reward_weight = function_call_reward_weight
        self.finish_reward_weight = finish_reward_weight
        self.error_penalty = error_penalty
        self.finish_bonus = finish_bonus
        self.num_examine = num_examine
    
    def __call__(self, data: DataProto) -> torch.Tensor:
        """
        计算ToolBench模式的reward
        
        Args:
            data: DataProto包含生成的数据和meta_info
            
        Returns:
            token_level_rewards: (batch_size, response_length)的reward tensor
        """
        # 如果已经有rm_scores，直接返回
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']
        
        batch_size = data.batch['responses'].shape[0]
        response_length = data.batch['responses'].shape[1]
        
        # 初始化reward tensor
        reward_tensor = torch.zeros((batch_size, response_length), dtype=torch.float32)
        
        # 从meta_info中获取ToolBench相关信息
        # meta_info存储在data.meta_info中（DataProto的属性）
        meta_info = {}
        if hasattr(data, 'meta_info') and data.meta_info:
            meta_info = data.meta_info
        
        # 获取每个样本的信息
        for i in range(batch_size):
            data_item = data[i]
            
            # 获取response
            response_ids = data_item.batch['responses']
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            attention_mask = data_item.batch['attention_mask']
            
            # 使用attention_mask确定有效的prompt和response长度
            valid_prompt_length = attention_mask[:prompt_length].sum().item()
            valid_response_length = attention_mask[prompt_length:].sum().item()
            
            # 只取有效的tokens（避免解码padding tokens）
            valid_prompt_ids = prompt_ids[-valid_prompt_length:] if valid_prompt_length > 0 else prompt_ids
            valid_response_ids = response_ids[:valid_response_length] if valid_response_length > 0 else response_ids

            # 解码response（只解码有效部分）
            # 注意：response_str包含模型生成的response（Thought/Action/Action Input）和observation（function调用结果）
            # observation格式为 "Observation: {"error": "...", "response": "..."}"
            # 虽然observation在训练时会被info_mask排除，但我们可以直接从response_str中解析来获取error信息
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)
            
            # 调试输出（只解码有效部分）
            if i == 0:
                print("#" * 30)
                print("[DEBUG REWARD] PROMPT (valid tokens only):")
                if valid_prompt_length > 0:
                    print(self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=False))
                else:
                    print("[DEBUG REWARD] (empty)")
                print("#" * 30)
                print("[DEBUG REWARD] RESPONSE:")
                print("[DEBUG REWARD] " + response_str)
                print("#" * 30)
                print("[DEBUG REWARD] Trained response:")
                info_mask = data_item.batch['info_mask']
                info_mask = info_mask[prompt_length:]
                trained = torch.where(info_mask.bool(), valid_response_ids, self.tokenizer.pad_token_id)
                print(self.tokenizer.decode(trained, skip_special_tokens=False))
                print("#" * 30)

            
            # 获取原始样本索引（处理batch repeat的情况）
            # 如果batch被repeat了，需要通过index找到原始样本
            original_idx = i
            if hasattr(data_item, 'non_tensor_batch') and data_item.non_tensor_batch and 'index' in data_item.non_tensor_batch:
                original_idx = int(data_item.non_tensor_batch['index'])
            elif hasattr(data, 'non_tensor_batch') and data.non_tensor_batch and 'index' in data.non_tensor_batch:
                if i < len(data.non_tensor_batch['index']):
                    original_idx = int(data.non_tensor_batch['index'][i])
            
            # 1. 格式奖励：检查是否包含正确的格式
            format_reward = self._compute_format_reward(original_idx, meta_info)
            
            # 2. Function call奖励：从response_str中解析Observation获取API调用结果
            function_call_reward = self._compute_function_call_reward(
                original_idx, meta_info
            )
            
            # 3. Finish调用奖励：检查最后一次是否调用了Finish
            finish_reward = self._compute_finish_reward(
                original_idx, meta_info
            )
            
            # 组合reward
            total_reward = (
                self.format_reward_weight * format_reward +
                self.function_call_reward_weight * function_call_reward +
                self.finish_reward_weight * finish_reward
            )
            
            # 将reward分配到最后一个有效response token
            # 注意：reward_tensor的形状是(batch_size, response_length)，
            # observation tokens会被info_mask标记，在训练时被排除
            if valid_response_length > 0:
                reward_tensor[i, valid_response_length - 1] = total_reward
            
            # 打印示例（用于调试）
            if i < self.num_examine:
                print(f"\n[Reward Sample {i}]")
                print(f"  Response: {response_str[:200]}...")
                print(f"  Format reward: {format_reward:.3f}")
                print(f"  Function call reward: {function_call_reward:.3f}")
                print(f"  Finish reward: {finish_reward:.3f}")
                print(f"  Total reward: {total_reward:.3f}")
        
        return reward_tensor
    
    def _compute_format_reward(self, original_idx: int, meta_info: Dict) -> float:
        turn_stats = meta_info.get('turn_stats', [])
        valid_action_stats = meta_info.get('valid_action_stats', [])
        if valid_action_stats and len(valid_action_stats) > original_idx and turn_stats and len(turn_stats) > original_idx:
            return valid_action_stats[original_idx] / turn_stats[original_idx]
        else:
            raise ValueError(f"turn_stats or valid_action_stats is not found for original_idx {original_idx}")
    
    def _parse_api_calls(self, response_str: str) -> List[Dict[str, str]]:
        """
        解析response_str中的所有API调用（Thought/Action/Action Input组合）
        
        Returns:
            List of dicts, each dict contains 'thought', 'action', 'action_input' fields
        """
        api_calls = []
        
        # 使用正则表达式查找所有 Thought: ... Action: ... Action Input: ... 的组合
        # 匹配完整的API调用模式：Thought: ... \nAction: ... \nAction Input: ...
        # 使用非贪婪匹配，直到下一个Thought:（如果有多个调用）或字符串结束
        # 注意：需要匹配换行符的不同形式，可能是\n或\n\n
        pattern = r'Thought:\s*(.*?)\nAction:\s*(.*?)\nAction Input:\s*(.*?)(?=\n+Thought:|$)'
        matches = re.finditer(pattern, response_str, re.DOTALL)
        
        for match in matches:
            thought_content = match.group(1).strip()
            action_content = match.group(2).strip()
            action_input_str = match.group(3).strip()
            
            # 检查是否有基本内容
            if thought_content and action_content and action_input_str:
                api_calls.append({
                    'thought': thought_content,
                    'action': action_content,
                    'action_input': action_input_str
                })
        
        return api_calls
    
    def _evaluate_single_api_call_format(self, api_call: Dict[str, str]) -> float:
        """
        评估单个API调用的格式正确性
        
        Args:
            api_call: 包含 'thought', 'action', 'action_input' 的字典
            
        Returns:
            格式奖励分数 (0.0 - 1.0)
        """
        thought = api_call.get('thought', '').strip()
        action = api_call.get('action', '').strip()
        action_input = api_call.get('action_input', '').strip()
        
        # 检查是否有基本的三个部分
        if not thought or not action or not action_input:
            return 0.0
        
        # 尝试解析Action Input是否为有效JSON
        try:
            # 尝试找到JSON对象
            brace_start = action_input.find('{')
            if brace_start != -1:
                # 尝试解析JSON
                brace_count = 0
                brace_end = brace_start
                for j in range(brace_start, min(brace_start + 2000, len(action_input))):
                    if action_input[j] == '{':
                        brace_count += 1
                    elif action_input[j] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            brace_end = j + 1
                            break
                
                if brace_count == 0:
                    json_str = action_input[brace_start:brace_end]
                    try:
                        json.loads(json_str)
                        return 1.0  # 完整且有效的格式
                    except json.JSONDecodeError:
                        return 0.5  # 格式存在但JSON无效
        except Exception:
            pass
        
        # 部分格式奖励（有基本格式但JSON可能不完整）
        return 0.5
    
    def _compute_function_call_reward(self, original_idx: int, meta_info: Dict) -> float:
        turn_stats = meta_info.get('turn_stats', [])
        api_error_stats = meta_info.get('api_error_stats', [])
        if api_error_stats and len(api_error_stats) > original_idx and turn_stats and len(turn_stats) > original_idx:
            return 1 - api_error_stats[original_idx] / (turn_stats[original_idx] - 1)
        else:
            raise ValueError(f"turn_stats or api_error_stats is not found for original_idx {original_idx}")

    def _parse_observations(self, response_str: str) -> List[Dict]:
        """
        从response_str中解析所有Observation
        
        Observation格式: "Observation: {"error": "...", "response": "..."}"
        
        Returns:
            List of observation dicts, each containing 'error' and 'response' keys
        """
        observations = []
        
        # 查找所有 "Observation:" 标签
        pattern = r'Observation:\s*(\{.*?\})(?=\n|$)'
        matches = re.finditer(pattern, response_str, re.DOTALL)
        
        for match in matches:
            json_str = match.group(1).strip()
            try:
                # 尝试解析JSON（需要正确匹配大括号）
                # 找到第一个 {，然后匹配到对应的 }
                brace_start = json_str.find('{')
                if brace_start != -1:
                    brace_count = 0
                    brace_end = brace_start
                    for j in range(brace_start, min(brace_start + 5000, len(json_str))):
                        if json_str[j] == '{':
                            brace_count += 1
                        elif json_str[j] == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                brace_end = j + 1
                                break
                    
                    if brace_count == 0:
                        complete_json_str = json_str[brace_start:brace_end]
                        obs_dict = json.loads(complete_json_str)
                        # 确保有error字段
                        if 'error' in obs_dict:
                            observations.append({
                                'error': obs_dict.get('error', ''),
                                'response': obs_dict.get('response', '')
                            })
            except (json.JSONDecodeError, ValueError):
                # JSON解析失败，跳过这个observation
                continue
        
        return observations
    
    def _compute_finish_reward(self, original_idx: int, meta_info: Dict) -> float:
        """
        计算Finish调用奖励
        检查最后一次是否调用了Finish函数
        
        注意：这个函数会在每轮生成时被调用，但Finish奖励应该只在最后一步给予。
        我们通过检查meta_info中的finish_called来判断是否真的调用了Finish。
        如果当前response包含Finish但meta_info中没有记录，说明这是中间步骤，不应该给奖励。
        """
        # 优先从meta_info中获取（更可靠）
        # meta_info中的finish_called只在execute_predictions中检测到Finish时才会设置
        finish_called = meta_info.get('finish_called', {})
        if original_idx in finish_called and finish_called[original_idx] is not None:
            # 只有在meta_info中记录了Finish调用，才给予奖励
            # 这确保了只有在execute_predictions中真正检测到Finish时才给奖励
            return_type = finish_called[original_idx]
            if return_type == 'give_answer':
                return 1
            elif return_type == 'give_up':
                return 0.5  # 部分奖励（至少调用了Finish）
            else:
                return 0.3  # Finish存在但格式可能不对
        
        return 0.0

    def _compute_pass_reward(self, response_str: str, sample_idx: int, meta_info: Dict, response_length: int) -> float:
        # TODO: implement pass rate reward
        finish_called = meta_info.get('finish_called', {})

        if sample_idx not in finish_called or finish_called[sample_idx] is None or finish_called[sample_idx] == 'give_up':
            return 0.0


def create_toolbench_reward_manager(
    tokenizer,
    format_reward_weight: float = 0.1,
    function_call_reward_weight: float = 0.2,
    finish_reward_weight: float = 0.3,
    **kwargs
) -> ToolBenchRewardManager:
    """
    创建ToolBench Reward Manager的工厂函数
    """
    return ToolBenchRewardManager(
        tokenizer=tokenizer,
        format_reward_weight=format_reward_weight,
        function_call_reward_weight=function_call_reward_weight,
        finish_reward_weight=finish_reward_weight,
        **kwargs
    )
