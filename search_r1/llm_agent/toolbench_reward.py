"""
ToolBench模式的Reward计算
包括：
1. 格式奖励：奖励模型生成正确的格式（Thought/Action/Action Input）
2. Function call正确奖励：如果API调用结果有error，则惩罚
3. Finish调用奖励：最后一次是否调用Finish
4. Pass rate奖励：基于OpenAI模型判断问题是否被正确解决
"""

import torch
import re
import json
import os
from typing import List, Dict, Optional
from verl import DataProto

from openai import OpenAI

class ToolBenchRewardManager:
    """ToolBench模式的Reward管理器"""
    
    def __init__(
        self,
        tokenizer,
        format_reward_weight: float = 0.1,
        function_call_reward_weight: float = 0.2,
        finish_reward_weight: float = 0.3,
        pass_rate_reward_weight: float = 0.4,
        error_penalty: float = -0.5,
        finish_bonus: float = 1,
        pass_rate_solved_reward: float = 1.0,
        pass_rate_unsolved_reward: float = 0.0,
        pass_rate_unsure_reward: float = 0.5,
        eval_model: str = "gpt-3.5-turbo",
        openai_api_key: Optional[str] = None,
        num_examine: int = 0
    ):
        """
        Args:
            tokenizer: Tokenizer用于解码
            format_reward_weight: 格式奖励权重
            function_call_reward_weight: Function call奖励权重
            finish_reward_weight: Finish调用奖励权重
            pass_rate_reward_weight: Pass rate奖励权重
            error_penalty: API调用错误的惩罚
            finish_bonus: 正确调用Finish的奖励
            pass_rate_solved_reward: 问题被解决时的reward
            pass_rate_unsolved_reward: 问题未解决时的reward
            pass_rate_unsure_reward: 不确定时的reward
            eval_model: 用于评估的OpenAI模型名称
            openai_api_key: OpenAI API key，如果不提供则从环境变量获取
            num_examine: 打印的样本数量
        """
        self.tokenizer = tokenizer
        self.format_reward_weight = format_reward_weight
        self.function_call_reward_weight = function_call_reward_weight
        self.finish_reward_weight = finish_reward_weight
        self.pass_rate_reward_weight = pass_rate_reward_weight
        self.error_penalty = error_penalty
        self.finish_bonus = finish_bonus
        self.pass_rate_solved_reward = pass_rate_solved_reward
        self.pass_rate_unsolved_reward = pass_rate_unsolved_reward
        self.pass_rate_unsure_reward = pass_rate_unsure_reward
        self.eval_model = eval_model
        self.num_examine = num_examine
        
        # 初始化OpenAI客户端
        self.use_pass_rate_reward = pass_rate_reward_weight > 0
        if self.use_pass_rate_reward:
            api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
            if not api_key:
                print("[WARNING] OpenAI API key not found. Pass rate reward will be disabled.")
                self.use_pass_rate_reward = False
            else:
                self.openai_client = OpenAI(api_key=api_key)
                # 定义评估函数schema
                self.eval_function_schema = {
                    "name": "check_answer_status",
                    "description": "Check if the answer solves the query",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "answer_status": {
                                "type": "string",
                                "enum": ["Solved", "Unsolved", "Unsure"],
                                "description": "Status indicating if the query is solved"
                            },
                            "reason": {
                                "type": "string",
                                "description": "Explanation for the answer status"
                            }
                        },
                        "required": ["answer_status", "reason"]
                    }
                }
        else:
            self.openai_client = None
    
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
                    print("[DEBUG REWARD] " + self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=False))
                else:
                    print("[DEBUG REWARD] (empty)")
                print("#" * 30)
                print("[DEBUG REWARD] RESPONSE (including observation):")
                print("[DEBUG REWARD] " + response_str)
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
            format_reward = self._compute_format_reward(response_str, valid_response_length)
            
            # 2. Function call奖励：从response_str中解析Observation获取API调用结果
            function_call_reward = self._compute_function_call_reward(
                response_str, valid_response_length
            )
            
            # 调试输出
            if i == 0:
                observations = self._parse_observations(response_str)
                print(f"[DEBUG Reward] Sample {i} (original_idx={original_idx})")
                print(f"  Parsed {len(observations)} observations from response_str")
                print(f"  function_call_reward: {function_call_reward}")
            
            # 3. Finish调用奖励：检查最后一次是否调用了Finish
            finish_reward = self._compute_finish_reward(
                response_str, original_idx, meta_info, valid_response_length
            )
            
            # 4. Pass rate奖励：基于OpenAI模型判断问题是否被正确解决
            pass_rate_reward = 0.0
            if self.use_pass_rate_reward:
                # 从prompt或meta_info中提取query
                query = self._extract_query(data_item, valid_prompt_ids)
                if query:
                    pass_rate_reward = self._compute_pass_rate_reward(
                        query, response_str, original_idx
                    )
            
            # 组合reward
            total_reward = (
                self.format_reward_weight * format_reward +
                self.function_call_reward_weight * function_call_reward +
                self.finish_reward_weight * finish_reward +
                self.pass_rate_reward_weight * pass_rate_reward
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
                print(f"  Format reward: {format_reward:.3f}")
                print(f"  Function call reward: {function_call_reward:.3f}")
                print(f"  Finish reward: {finish_reward:.3f}")
                if self.use_pass_rate_reward:
                    print(f"  Pass rate reward: {pass_rate_reward:.3f}")
                print(f"  Total reward: {total_reward:.3f}")
        
        return reward_tensor
    
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
                return 0.5
            else:
                return 0.3
        
        return 0.0
    
    def _extract_query(self, data_item, valid_prompt_ids) -> Optional[str]:
        """
        从data_item中提取query
        
        Args:
            data_item: DataProtoItem
            valid_prompt_ids: 有效的prompt token ids
            
        Returns:
            query字符串，如果无法提取则返回None
        """
        if hasattr(data_item, 'non_tensor_batch') and data_item.non_tensor_batch:
            return data_item.non_tensor_batch['prompt'][1]['content']
        
        print(data_item.non_tensor_batch)
        raise ValueError("No query found in data_item")
    
    def _compute_pass_rate_reward(
        self, 
        query: str, 
        response_str: str, 
        sample_idx: int
    ) -> float:
        """
        计算pass rate reward，通过调用OpenAI模型判断问题是否被正确解决
        
        Args:
            query: 用户的问题
            response_str: 模型的完整输出轨迹（包含Thought/Action/Action Input/Observation）
            sample_idx: 样本索引
            
        Returns:
            reward分数 (Solved: pass_rate_solved_reward, Unsolved: pass_rate_unsolved_reward, Unsure: pass_rate_unsure_reward)
        """
        if not self.use_pass_rate_reward or not self.openai_client:
            return 0.0
        
        try:
            # 提取最终答案（从Finish调用中提取，或者使用最后一个有效的回答）
            final_answer = self._extract_final_answer(response_str)
            
            # 如果final_answer为空，直接返回Unsolved的reward
            if not final_answer or final_answer.strip() == '' or 'give_up' in str(final_answer).lower():
                return self.pass_rate_unsolved_reward
            
            # 第一阶段：简单检查（只使用query和final_answer）
            answer_status = self._check_answer_status_simple(query, final_answer)
            
            # 如果Unsure，进行详细检查（使用完整的执行轨迹）
            if answer_status == "Unsure":
                answer_status = self._check_answer_status_detailed(query, response_str)
            
            # 根据状态返回相应的reward
            if answer_status == "Solved":
                return self.pass_rate_solved_reward
            elif answer_status == "Unsolved":
                return self.pass_rate_unsolved_reward
            else:  # Unsure
                return self.pass_rate_unsure_reward
                
        except Exception as e:
            # 如果评估失败，返回Unsure的reward（中性）
            print(f"[WARNING] Pass rate evaluation failed for sample {sample_idx}: {e}")
            return self.pass_rate_unsure_reward
    
    def _extract_final_answer(self, response_str: str) -> str:
        """
        从response_str中提取最终答案
        
        Args:
            response_str: 完整的响应字符串
            
        Returns:
            最终答案字符串
        """
        # 查找Finish调用中的answer
        finish_pattern = r'Action:\s*Finish\s*\nAction Input:\s*({.*?})'
        match = re.search(finish_pattern, response_str, re.DOTALL | re.IGNORECASE)
        if match:
            try:
                finish_input = json.loads(match.group(1))
                if 'answer' in finish_input:
                    return str(finish_input['answer'])
            except json.JSONDecodeError:
                pass
        
        # 如果没有找到Finish，尝试提取最后一个Action Input中的answer
        # 或者返回整个response_str的最后部分作为答案
        return response_str.split('Action Input:')[-1].strip() if 'Action Input:' in response_str else ''
    
    def _check_answer_status_simple(self, query: str, final_answer: str) -> str:
        """
        第一阶段检查：使用query和final_answer进行简单判断
        
        Args:
            query: 用户问题
            final_answer: 最终答案
            
        Returns:
            "Solved", "Unsolved", 或 "Unsure"
        """
        prompt_template = """Giving the query and answer, you need give `answer_status` of the answer by following rules:
1. If the answer is a sorry message or not a positive/straight response for the given query, return "Unsolved".
2. If the answer is a positive/straight response for the given query, you have to further check.
2.1 If the answer is not sufficient to determine whether the solve the query or not, return "Unsure".
2.2 If you are confident that the answer is sufficient to determine whether the solve the query or not, return "Solved" or "Unsolved".

Query:
{query}

Answer:
{answer}

Now give your reason in "content" and `answer_status` of JSON to `check_answer_status`."""
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.eval_model,
                messages=[{
                    "role": "user",
                    "content": prompt_template.format(query=query, answer=final_answer)
                }],
                tools=[{"type": "function", "function": self.eval_function_schema}],
                tool_choice={"type": "function", "function": {"name": "check_answer_status"}},
                temperature=0.0
            )
            
            tool_call = response.choices[0].message.tool_calls[0]
            result = json.loads(tool_call.function.arguments)
            return result.get('answer_status', 'Unsure')
        except Exception as e:
            print(f"[WARNING] Simple answer status check failed: {e}")
            return "Unsure"
    
    def _check_answer_status_detailed(self, query: str, response_str: str) -> str:
        """
        第二阶段检查：使用完整的执行轨迹进行详细判断
        
        Args:
            query: 用户问题
            response_str: 完整的响应字符串（包含所有执行细节）
            
        Returns:
            "Solved", "Unsolved", 或 "Unsure"
        """
        prompt_template = """Giving the query and the correspond execution detail of an answer, you need give `answer_status` of the answer by following rules:
1. If all 'tool' nodes' message indicate that there are errors happened, return "Unsolved"
2. If you find the information in the "final_answer" is not true/valid according to the messages in 'tool' nodes, return "Unsolved"
3. If you are unable to verify the authenticity and validity of the information, return "Unsure"
4. If there are 'tool' node in the chain contains successful func calling and those calling indeed solve the query, return "Solved"

Query:
{query}

Answer:
{answer}

Now you are requested to give reason in "content" and `answer_status` of JSON to `parse_answer_status`."""
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.eval_model,
                messages=[{
                    "role": "user",
                    "content": prompt_template.format(query=query, answer=response_str)
                }],
                tools=[{"type": "function", "function": {
                    **self.eval_function_schema,
                    "name": "parse_answer_status"
                }}],
                tool_choice={"type": "function", "function": {"name": "parse_answer_status"}},
                temperature=0.0
            )
            
            tool_call = response.choices[0].message.tool_calls[0]
            result = json.loads(tool_call.function.arguments)
            return result.get('answer_status', 'Unsure')
        except Exception as e:
            print(f"[WARNING] Detailed answer status check failed: {e}")
            return "Unsure"


def create_toolbench_reward_manager(
    tokenizer,
    format_reward_weight: float = 0.1,
    function_call_reward_weight: float = 0.2,
    finish_reward_weight: float = 0.3,
    pass_rate_reward_weight: float = 0.4,
    **kwargs
) -> ToolBenchRewardManager:
    """
    创建ToolBench Reward Manager的工厂函数
    
    Args:
        tokenizer: Tokenizer用于解码
        format_reward_weight: 格式奖励权重
        function_call_reward_weight: Function call奖励权重
        finish_reward_weight: Finish调用奖励权重
        pass_rate_reward_weight: Pass rate奖励权重（基于OpenAI模型判断）
        **kwargs: 其他参数传递给ToolBenchRewardManager
    """
    return ToolBenchRewardManager(
        tokenizer=tokenizer,
        format_reward_weight=format_reward_weight,
        function_call_reward_weight=function_call_reward_weight,
        finish_reward_weight=finish_reward_weight,
        pass_rate_reward_weight=pass_rate_reward_weight,
        **kwargs
    )
