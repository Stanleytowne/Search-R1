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
from typing import List, Dict, Any
from verl import DataProto


class ToolBenchRewardManager:
    """ToolBench模式的Reward管理器"""
    
    def __init__(
        self,
        tokenizer,
        format_reward_weight: float = 0.1,
        function_call_reward_weight: float = 0.2,
        finish_reward_weight: float = 0.3,
        format_reward_per_token: float = 0.01,
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
            format_reward_per_token: 每个正确格式token的奖励
            error_penalty: API调用错误的惩罚
            finish_bonus: 正确调用Finish的奖励
            num_examine: 打印的样本数量
        """
        self.tokenizer = tokenizer
        self.format_reward_weight = format_reward_weight
        self.function_call_reward_weight = function_call_reward_weight
        self.finish_reward_weight = finish_reward_weight
        self.format_reward_per_token = format_reward_per_token
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
            prompt_length = data_item.batch['prompts'].shape[-1]
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum().item()
            valid_response_ids = response_ids[:valid_response_length]
            
            # 解码response
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)
            
            breakpoint()
            # 移除</s> token（如果存在）
            response_str = response_str.replace('</s>', '').strip()
            
            # 获取原始样本索引（处理batch repeat的情况）
            # 如果batch被repeat了，需要通过index找到原始样本
            original_idx = i
            if hasattr(data_item, 'non_tensor_batch') and data_item.non_tensor_batch and 'index' in data_item.non_tensor_batch:
                original_idx = int(data_item.non_tensor_batch['index'])
            elif hasattr(data, 'non_tensor_batch') and data.non_tensor_batch and 'index' in data.non_tensor_batch:
                if i < len(data.non_tensor_batch['index']):
                    original_idx = int(data.non_tensor_batch['index'][i])
            
            # 1. 格式奖励：检查是否包含正确的格式
            format_reward = self._compute_format_reward(response_str, valid_response_length)
            
            # 2. Function call奖励：从meta_info中获取API调用结果
            function_call_reward = self._compute_function_call_reward(
                original_idx, meta_info, valid_response_length
            )
            
            # 3. Finish调用奖励：检查最后一次是否调用了Finish
            finish_reward = self._compute_finish_reward(
                response_str, original_idx, meta_info, valid_response_length
            )
            
            # 组合reward
            total_reward = (
                self.format_reward_weight * format_reward +
                self.function_call_reward_weight * function_call_reward +
                self.finish_reward_weight * finish_reward
            )
            
            # 将reward分配到最后一个有效token
            if valid_response_length > 0:
                reward_tensor[i, valid_response_length - 1] = total_reward
            
            # 格式奖励可以分配到每个token（可选）
            if self.format_reward_per_token > 0:
                format_tokens_reward = self._compute_per_token_format_reward(
                    response_str, valid_response_length
                )
                reward_tensor[i, :valid_response_length] += format_tokens_reward
            
            # 打印示例（用于调试）
            if i < self.num_examine:
                print(f"\n[Reward Sample {i}]")
                print(f"  Response: {response_str[:200]}...")
                print(f"  Format reward: {format_reward:.3f}")
                print(f"  Function call reward: {function_call_reward:.3f}")
                print(f"  Finish reward: {finish_reward:.3f}")
                print(f"  Total reward: {total_reward:.3f}")
        
        return reward_tensor
    
    def _compute_format_reward(self, response_str: str, response_length: int) -> float:
        """
        计算格式奖励
        检查是否包含Thought/Action/Action Input格式
        """
        # 检查是否包含完整的格式
        has_thought = "Thought:" in response_str or "Thought:" in response_str
        has_action = "\nAction:" in response_str or "Action:" in response_str
        has_action_input = "\nAction Input:" in response_str or "Action Input:" in response_str
        
        # 如果包含完整格式，给予奖励
        if has_thought and has_action and has_action_input:
            # 尝试解析格式是否正确
            try:
                thought_start = response_str.find("Thought:")
                action_start = response_str.find("\nAction:")
                action_input_start = response_str.find("\nAction Input:")
                
                if thought_start != -1 and action_start != -1 and action_input_start != -1:
                    # 检查顺序是否正确
                    if thought_start < action_start < action_input_start:
                        # 尝试解析Action Input是否为有效JSON
                        action_input_str = response_str[action_input_start + len("\nAction Input:"):].strip()
                        # 尝试找到JSON对象
                        brace_start = action_input_str.find('{')
                        if brace_start != -1:
                            # 尝试解析JSON
                            brace_count = 0
                            brace_end = brace_start
                            for j in range(brace_start, min(brace_start + 1000, len(action_input_str))):
                                if action_input_str[j] == '{':
                                    brace_count += 1
                                elif action_input_str[j] == '}':
                                    brace_count -= 1
                                    if brace_count == 0:
                                        brace_end = j + 1
                                        break
                            if brace_count == 0:
                                try:
                                    json.loads(action_input_str[brace_start:brace_end])
                                    return 1.0  # 完整且有效的格式
                                except json.JSONDecodeError:
                                    return 0.5  # 格式存在但JSON无效
                        return 0.5  # 格式存在但可能不完整
            except Exception:
                pass
        
        # 部分格式奖励
        if has_thought or has_action:
            return 0.2
        
        return 0.0
    
    def _compute_per_token_format_reward(self, response_str: str, response_length: int) -> torch.Tensor:
        """
        计算每个token的格式奖励
        对于符合格式的token给予小奖励
        """
        # 简单实现：对包含格式关键词的token给予奖励
        # 这里需要更精细的实现，但为了简化，我们给一个小的基础奖励
        reward = torch.zeros(response_length, dtype=torch.float32)
        
        # 如果包含格式关键词，给所有token小奖励
        if "Thought:" in response_str or "Action:" in response_str:
            reward.fill_(self.format_reward_per_token)
        
        return reward
    
    def _compute_function_call_reward(self, sample_idx: int, meta_info: Dict, response_length: int) -> float:
        """
        计算Function call奖励
        如果API调用结果有error，则惩罚
        """
        # 从meta_info中获取API调用结果
        api_errors = meta_info.get('api_errors', {})
        
        # 获取这个样本的API调用结果
        # api_errors格式: {sample_idx: [bool, bool, ...]} 其中True表示有error
        sample_errors = api_errors.get(sample_idx, [])
        
        # 计算奖励：每个成功的API调用给奖励，每个错误给惩罚
        reward = 0.0
        
        if sample_errors:
            for has_error in sample_errors:
                if has_error:  # 如果有error
                    reward += self.error_penalty
                else:  # 如果没有error，给小的正奖励
                    reward += 0.1
        else:
            # 如果没有API调用，可能是格式错误或没有调用API
            # 不给奖励也不给惩罚（中性）
            pass
        
        return reward
    
    def _compute_finish_reward(self, response_str: str, sample_idx: int, meta_info: Dict, response_length: int) -> float:
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
        if sample_idx in finish_called and finish_called[sample_idx] is not None:
            # 只有在meta_info中记录了Finish调用，才给予奖励
            # 这确保了只有在execute_predictions中真正检测到Finish时才给奖励
            return_type = finish_called[sample_idx]
            if return_type == 'give_answer':
                return self.finish_bonus
            elif return_type == 'give_up_and_restart':
                return self.finish_bonus * 0.5  # 部分奖励（至少调用了Finish）
            else:
                return self.finish_bonus * 0.3  # Finish存在但格式可能不对
        
        # 如果meta_info中没有Finish记录，即使response_str中有Finish，也不给奖励
        # 因为可能是中间步骤的response，还没有真正执行Finish
        # 这样可以避免在中间步骤错误地给予Finish奖励
        
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
