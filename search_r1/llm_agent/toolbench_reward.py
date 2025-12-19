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
        pass_reward_weight: float = 0.1,
        error_penalty: float = -0.5,
        finish_bonus: float = 0.5,
        num_examine: int = 0,
        reward_server_url: str = "http://localhost:8000/evaluate_batch"
    ):
        """
        Args:
            tokenizer: Tokenizer用于解码
            format_reward_weight: 格式奖励权重
            function_call_reward_weight: Function call奖励权重
            finish_reward_weight: Finish调用奖励权重
            pass_reward_weight: Remote pass奖励权重
            error_penalty: API调用错误的惩罚
            finish_bonus: 正确调用Finish的奖励
            num_examine: 打印的样本数量
        """
        self.tokenizer = tokenizer
        self.format_reward_weight = format_reward_weight
        self.function_call_reward_weight = function_call_reward_weight
        self.finish_reward_weight = finish_reward_weight
        self.pass_reward_weight = pass_reward_weight
        self.error_penalty = error_penalty
        self.finish_bonus = finish_bonus
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
        meta_info = {}
        if hasattr(data, 'meta_info') and data.meta_info:
            meta_info = data.meta_info

        all_queries = []
        all_trajectories = []
        valid_info_list = []

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

            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            query_str = self._extract_query(prompt_str)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)
            
            all_queries.append(query_str)
            all_trajectories.append(response_str)
            valid_info_list.append(valid_response_length)

            # debug output
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
                info_mask = info_mask[prompt_length:][:valid_response_length]
                trained = torch.where(info_mask.bool(), valid_response_ids, self.tokenizer.pad_token_id)
                print(self.tokenizer.decode(trained, skip_special_tokens=False))
                print("#" * 30)
        
        pass_rewards = self._get_remote_pass_rewards(all_queries, all_trajectories)

        # 获取每个样本的信息
        for i in range(batch_size):
            data_item = data[i]
            response_str = all_trajectories[i]
            valid_response_length = valid_info_list[i]

            original_idx = i
            if hasattr(data_item, 'non_tensor_batch') and data_item.non_tensor_batch and 'index' in data_item.non_tensor_batch:
                original_idx = int(data_item.non_tensor_batch['index'])
            elif hasattr(data, 'non_tensor_batch') and data.non_tensor_batch and 'index' in data.non_tensor_batch:
                if i < len(data.non_tensor_batch['index']):
                    original_idx = int(data.non_tensor_batch['index'][i])
            
            # 1. 格式奖励：检查是否包含正确的格式
            format_reward = self._compute_format_reward(response_str, valid_response_length)
            
            # 2. Function call奖励：从response_str中解析Observation获取API调用结果
            function_call_reward = self._compute_function_call_reward(
                response_str, valid_response_length
            )
            
            # 3. Finish调用奖励：检查最后一次是否调用了Finish
            finish_reward = self._compute_finish_reward(
                response_str, original_idx, meta_info, valid_response_length
            )
            
            # 组合reward
            total_reward = (
                self.pass_reward_weight * pass_rewards[i] +
                self.format_reward_weight * format_reward +
                self.function_call_reward_weight * function_call_reward +
                self.finish_reward_weight * finish_reward
            )
            
            # 将reward分配到最后一个有效response token
            # 注意：reward_tensor的形状是(batch_size, response_length)，
            # 这里的response_length只包含模型生成的response tokens，不包含prompt和observation
            # observation tokens会被info_mask标记，在训练时被排除
            if valid_response_length > 0:
                reward_tensor[i, valid_response_length - 1] = total_reward
            
            # 打印示例（用于调试）
            if i < self.num_examine:
                print(f"\n[Reward Sample {i}]")
                print(f"  Response: {response_str[:200]}...")
                print(f"  Pass reward: {pass_rewards[i]:.3f}")
                print(f"  Format reward: {format_reward:.3f}")
                print(f"  Function call reward: {function_call_reward:.3f}")
                print(f"  Finish reward: {finish_reward:.3f}")
                print(f"  Total reward: {total_reward:.3f}")
        
        return reward_tensor
    
    def _extract_query(self, full_prompt: str) -> str:
        import re
        pattern = r'<|im_start|>user\n?(.*?)(?=<|im_end|>|$)'
        matches = re.findall(pattern, full_prompt, re.DOTALL)
        
        if matches:
            query = matches[-1].strip()
            return query
        
        return full_prompt.strip()

    def _compute_format_reward(self, response_str: str, response_length: int) -> float:
        """
        计算格式奖励
        检查是否包含Thought/Action/Action Input格式
        关键：需要对每次API调用（每次Thought/Action/Action Input组合）取平均，
        防止模型通过重复调用API来获得高奖励
        """
        # 解析出所有API调用（每次Thought/Action/Action Input组合）
        api_calls = self._parse_api_calls(response_str)
        
        if not api_calls:
            # 没有找到任何API调用格式
            return 0.0
        
        # 对每次API调用计算格式奖励，然后取平均
        format_scores = []
        for api_call in api_calls:
            score = self._evaluate_single_api_call_format(api_call)
            format_scores.append(score)
        
        # 取平均值，防止重复调用API获得高奖励
        avg_score = sum(format_scores) / len(format_scores) if format_scores else 0.0
        return avg_score
    
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
    
    def _compute_function_call_reward(self, response_str: str, response_length: int) -> float:
        """
        计算Function call奖励
        如果API调用结果有error，则惩罚
        
        直接从response_str中解析Observation来获取error信息
        Observation格式: "Observation: {"error": "...", "response": "..."}"
        """
        # 从response_str中解析所有Observation
        observations = self._parse_observations(response_str)
        
        # 计算奖励：每个成功的API调用给奖励，每个错误给惩罚
        reward = 0.0
        
        if observations:
            for obs_json in observations:
                has_error = bool(obs_json.get('error', '').strip())
                if not has_error:  # 如果有error
                    reward += 1
            
            return reward / len(observations)
        else:
            # 如果没有API调用或Observation，可能是格式错误或没有调用API
            # 不给奖励也不给惩罚（中性）
            return 0.0

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
                return 1
            elif return_type == 'give_up':
                return 0.5  # 部分奖励（至少调用了Finish）
            else:
                return 0.3  # Finish存在但格式可能不对
        
        # 如果meta_info中没有Finish记录，即使response_str中有Finish，也不给奖励
        # 因为可能是中间步骤的response，还没有真正执行Finish
        # 这样可以避免在中间步骤错误地给予Finish奖励
        
        return 0.0

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
                return [0.5] * len(queries)
        except Exception as e:
            print(f"Failed to connect to reward server: {e}")
            return [0.5] * len(queries)



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
