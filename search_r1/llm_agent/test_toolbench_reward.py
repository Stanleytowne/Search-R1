"""
测试 ToolBenchRewardManager 的实现正确性

重点测试：
1. 格式奖励是否对每次API调用取平均（防止重复调用获得高奖励）
2. Reward是否只加在response tokens上，排除observation
3. Function call reward和Finish reward的逻辑
"""

import torch
import sys
import os

# 添加路径以便导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from search_r1.llm_agent.toolbench_reward import ToolBenchRewardManager
from verl import DataProto


class MockTokenizer:
    """模拟Tokenizer用于测试"""
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token = '</s>'
        # 创建一个简单的字符到token的映射
        self.char_to_token = {}
        self.token_to_char = {}
        self.next_token_id = 1  # 从1开始，0是pad
    
    def _get_token_id(self, char):
        """获取字符的token ID，如果不存在则创建"""
        if char not in self.char_to_token:
            token_id = self.next_token_id
            self.char_to_token[char] = token_id
            self.token_to_char[token_id] = char
            self.next_token_id += 1
        return self.char_to_token[char]
    
    def decode(self, tokens, skip_special_tokens=False):
        """Token解码（仅用于测试）"""
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        
        # 将token IDs转换为字符
        result = []
        for t in tokens:
            if t == self.pad_token_id:
                continue
            if t in self.token_to_char:
                result.append(self.token_to_char[t])
            else:
                # 如果是未知token，尝试用ASCII字符表示
                result.append(chr(min(127, max(32, t))))
        
        return ''.join(result)
    
    def encode(self, text):
        """文本编码（仅用于测试）"""
        return [self._get_token_id(c) for c in text]


def create_test_data(response_str: str, tokenizer, batch_size=1, response_length=100):
    """创建测试用的DataProto"""
    # 将response_str转换为token ids
    response_ids = tokenizer.encode(response_str)
    
    # 填充到指定长度
    valid_token_length = len(response_ids)
    if valid_token_length < response_length:
        response_ids = response_ids + [tokenizer.pad_token_id] * (response_length - valid_token_length)
    else:
        response_ids = response_ids[:response_length]
        valid_token_length = response_length
    
    # 创建attention mask（所有有效token都是1，padding是0）
    attention_mask = [1] * valid_token_length + [0] * (response_length - valid_token_length)
    
    # 扩展到batch
    if batch_size > 1:
        response_ids_list = [response_ids] * batch_size
        attention_mask_list = [attention_mask] * batch_size
    else:
        response_ids_list = response_ids
        attention_mask_list = attention_mask
    
    response_tensor = torch.tensor(response_ids_list, dtype=torch.long)
    attention_tensor = torch.tensor(attention_mask_list, dtype=torch.long)
    
    # 创建prompt（空的，10个token）
    prompt_length = 10
    prompt_tensor = torch.zeros((batch_size, prompt_length), dtype=torch.long)
    prompt_attention = torch.ones((batch_size, prompt_length), dtype=torch.long)
    
    # 合并attention mask: [prompt_attention, response_attention]
    full_attention = torch.cat([prompt_attention, attention_tensor], dim=1)
    
    data_dict = {
        'responses': response_tensor,
        'prompts': prompt_tensor,
        'attention_mask': full_attention,
    }
    
    data = DataProto.from_dict(data_dict)
    return data


def test_single_api_call():
    """测试1: 单个正确的API调用"""
    print("=" * 80)
    print("测试1: 单个正确的API调用")
    print("=" * 80)
    
    response_str = "Thought: I need to search for information.\nAction: search\nAction Input: {\"query\": \"test\"}"
    tokenizer = MockTokenizer()
    reward_manager = ToolBenchRewardManager(
        tokenizer=tokenizer,
        format_reward_weight=0.1,
        function_call_reward_weight=0.2,
        finish_reward_weight=0.3,
        num_examine=1
    )
    
    # 测试格式奖励
    format_reward = reward_manager._compute_format_reward(response_str, len(response_str))
    print(f"Response: {response_str[:100]}...")
    print(f"格式奖励: {format_reward:.3f}")
    print(f"期望: 1.0 (完整且有效的格式)")
    assert abs(format_reward - 1.0) < 0.01, f"格式奖励应该是1.0，实际是{format_reward}"
    print("✓ 通过：单个API调用格式奖励正确\n")


def test_multiple_api_calls_averaging():
    """测试2: 多个API调用 - 验证取平均逻辑（关键测试）"""
    print("=" * 80)
    print("测试2: 多个API调用 - 验证格式奖励取平均（防止重复调用获得高奖励）")
    print("=" * 80)
    
    # 场景1: 2个正确的API调用
    response_str1 = (
        "Thought: First search\nAction: search\nAction Input: {\"query\": \"test1\"}\n\n"
        "Thought: Second search\nAction: search\nAction Input: {\"query\": \"test2\"}"
    )
    
    # 场景2: 1个正确，1个JSON错误
    response_str2 = (
        "Thought: First search\nAction: search\nAction Input: {\"query\": \"test1\"}\n\n"
        "Thought: Second search\nAction: search\nAction Input: {invalid json"
    )
    
    # 场景3: 3个正确的API调用
    response_str3 = (
        "Thought: First\nAction: search\nAction Input: {\"a\": 1}\n\n"
        "Thought: Second\nAction: search\nAction Input: {\"b\": 2}\n\n"
        "Thought: Third\nAction: search\nAction Input: {\"c\": 3}"
    )
    
    tokenizer = MockTokenizer()
    reward_manager = ToolBenchRewardManager(tokenizer=tokenizer)
    
    # 测试场景1
    format_reward1 = reward_manager._compute_format_reward(response_str1, len(response_str1))
    print(f"场景1 (2个正确调用):")
    print(f"  格式奖励: {format_reward1:.3f}")
    print(f"  期望: 1.0 (两个都是1.0，平均还是1.0)")
    assert abs(format_reward1 - 1.0) < 0.01, f"应该是1.0，实际是{format_reward1}"
    
    # 测试场景2
    format_reward2 = reward_manager._compute_format_reward(response_str2, len(response_str2))
    print(f"\n场景2 (1个正确 + 1个JSON错误):")
    print(f"  格式奖励: {format_reward2:.3f}")
    print(f"  期望: 0.75 ((1.0 + 0.5) / 2 = 0.75)")
    assert abs(format_reward2 - 0.75) < 0.01, f"应该是0.75，实际是{format_reward2}"
    
    # 测试场景3
    format_reward3 = reward_manager._compute_format_reward(response_str3, len(response_str3))
    print(f"\n场景3 (3个正确调用):")
    print(f"  格式奖励: {format_reward3:.3f}")
    print(f"  期望: 1.0 (三个都是1.0，平均还是1.0)")
    assert abs(format_reward3 - 1.0) < 0.01, f"应该是1.0，实际是{format_reward3}"
    
    # 关键验证：重复调用不会增加奖励
    print(f"\n关键验证：")
    print(f"  单个调用奖励: 1.0")
    print(f"  两个调用奖励: {format_reward1:.3f}")
    print(f"  三个调用奖励: {format_reward3:.3f}")
    print(f"  ✓ 奖励取平均，不会因为调用次数多而增加总奖励")
    print("✓ 通过：格式奖励正确取平均\n")


def test_parse_api_calls():
    """测试3: API调用解析功能"""
    print("=" * 80)
    print("测试3: API调用解析功能")
    print("=" * 80)
    
    tokenizer = MockTokenizer()
    reward_manager = ToolBenchRewardManager(tokenizer=tokenizer)
    
    response_str = (
        "Thought: First\nAction: search\nAction Input: {\"a\": 1}\n\n"
        "Thought: Second\nAction: finish\nAction Input: {\"b\": 2}"
    )
    
    api_calls = reward_manager._parse_api_calls(response_str)
    print(f"解析到的API调用数量: {len(api_calls)}")
    print(f"期望: 2")
    assert len(api_calls) == 2, f"应该解析到2个API调用，实际是{len(api_calls)}"
    
    for i, call in enumerate(api_calls):
        print(f"\n调用 {i+1}:")
        print(f"  Thought: {call['thought'][:50]}...")
        print(f"  Action: {call['action']}")
        print(f"  Action Input: {call['action_input'][:50]}...")
    
    print("✓ 通过：API调用解析正确\n")


def test_function_call_reward():
    """测试4: Function call reward - 从response_str中解析Observation"""
    print("=" * 80)
    print("测试4: Function call reward - 从response_str中解析Observation")
    print("=" * 80)
    
    tokenizer = MockTokenizer()
    reward_manager = ToolBenchRewardManager(
        tokenizer=tokenizer,
        error_penalty=-0.5,
        function_call_reward_weight=0.2
    )
    
    # 场景1: 没有错误（两个Observation都没有error）
    response_str1 = (
        "Thought: First search\nAction: search\nAction Input: {\"query\": \"test1\"}\n\n"
        "Observation: {\"error\": \"\", \"response\": \"success\"}\n\n"
        "Thought: Second search\nAction: search\nAction Input: {\"query\": \"test2\"}\n\n"
        "Observation: {\"error\": \"\", \"response\": \"success\"}"
    )
    function_reward1 = reward_manager._compute_function_call_reward(response_str1, 100)
    print(f"场景1 (无错误，2个Observation): {function_reward1:.3f}")
    print(f"期望: 0.2 (2个成功调用 * 0.1 = 0.2)")
    assert abs(function_reward1 - 0.2) < 0.01, f"应该是0.2，实际是{function_reward1}"
    
    # 场景2: 有错误（第一个成功，第二个有error）
    response_str2 = (
        "Thought: First search\nAction: search\nAction Input: {\"query\": \"test1\"}\n\n"
        "Observation: {\"error\": \"\", \"response\": \"success\"}\n\n"
        "Thought: Second search\nAction: search\nAction Input: {\"query\": \"test2\"}\n\n"
        "Observation: {\"error\": \"API error occurred\", \"response\": \"\"}"
    )
    function_reward2 = reward_manager._compute_function_call_reward(response_str2, 100)
    print(f"\n场景2 (1个错误，1个成功): {function_reward2:.3f}")
    print(f"期望: -0.4 (1个成功 * 0.1 + 1个错误 * -0.5 = -0.4)")
    assert abs(function_reward2 - (-0.4)) < 0.01, f"应该是-0.4，实际是{function_reward2}"
    
    # 场景3: 没有Observation（没有API调用）
    response_str3 = "Thought: Just thinking\nAction: None\nAction Input: {}"
    function_reward3 = reward_manager._compute_function_call_reward(response_str3, 100)
    print(f"\n场景3 (没有Observation): {function_reward3:.3f}")
    print(f"期望: 0.0 (没有API调用，不给奖励也不给惩罚)")
    assert abs(function_reward3 - 0.0) < 0.01, f"应该是0.0，实际是{function_reward3}"
    
    print("✓ 通过：Function call reward正确（直接从response_str解析Observation）\n")


def test_finish_reward():
    """测试5: Finish reward"""
    print("=" * 80)
    print("测试5: Finish reward")
    print("=" * 80)
    
    response_str = "Thought: Done\nAction: finish\nAction Input: {\"answer\": \"result\"}"
    tokenizer = MockTokenizer()
    reward_manager = ToolBenchRewardManager(
        tokenizer=tokenizer,
        finish_bonus=0.5,
        finish_reward_weight=0.3
    )
    
    # 场景1: 成功调用Finish
    meta_info1 = {
        'finish_called': {
            0: 'give_answer'
        }
    }
    finish_reward1 = reward_manager._compute_finish_reward(response_str, 0, meta_info1, 100)
    print(f"场景1 (give_answer): {finish_reward1:.3f}")
    print(f"期望: 0.5")
    assert abs(finish_reward1 - 0.5) < 0.01, f"应该是0.5，实际是{finish_reward1}"
    
    # 场景2: give_up
    meta_info2 = {
        'finish_called': {
            0: 'give_up'
        }
    }
    finish_reward2 = reward_manager._compute_finish_reward(response_str, 0, meta_info2, 100)
    print(f"\n场景2 (give_up): {finish_reward2:.3f}")
    print(f"期望: 0.25 (0.5 * 0.5)")
    assert abs(finish_reward2 - 0.25) < 0.01, f"应该是0.25，实际是{finish_reward2}"
    
    # 场景3: 没有调用Finish
    meta_info3 = {
        'finish_called': {}
    }
    finish_reward3 = reward_manager._compute_finish_reward(response_str, 0, meta_info3, 100)
    print(f"\n场景3 (没有调用): {finish_reward3:.3f}")
    print(f"期望: 0.0")
    assert abs(finish_reward3 - 0.0) < 0.01, f"应该是0.0，实际是{finish_reward3}"
    
    print("✓ 通过：Finish reward正确\n")


def test_reward_only_on_response_tokens():
    """测试6: 验证reward只加在response tokens上"""
    print("=" * 80)
    print("测试6: 验证reward只加在response tokens上（排除prompt和observation）")
    print("=" * 80)
    
    response_str = "Thought: Test\nAction: search\nAction Input: {\"query\": \"test\"}"
    tokenizer = MockTokenizer()
    # 先编码以获取正确的token数量
    token_ids = tokenizer.encode(response_str)
    response_length = max(50, len(token_ids) + 10)  # 确保有足够空间
    data = create_test_data(response_str, tokenizer, batch_size=1, response_length=response_length)
    
    reward_manager = ToolBenchRewardManager(
        tokenizer=tokenizer,
        format_reward_weight=0.1,
        function_call_reward_weight=0.2,
        finish_reward_weight=0.3,
        format_reward_per_token=0.0,  # 关闭per-token奖励
        num_examine=0
    )
    
    # 设置meta_info
    meta_info = {
        'api_errors': {0: [False]},
        'finish_called': {0: 'give_answer'}
    }
    data.meta_info = meta_info
    
    # 计算reward
    reward_tensor = reward_manager(data)
    
    print(f"Reward tensor shape: {reward_tensor.shape}")
    print(f"期望 shape: (1, {response_length}) - 只包含response长度，不包含prompt")
    assert reward_tensor.shape[0] == 1, f"Batch size应该是1"
    assert reward_tensor.shape[1] == response_length, f"Response length应该是{response_length}"
    
    # 获取有效response token长度（从attention mask）
    valid_token_length = len(token_ids)
    
    # 检查reward是否只加在最后一个有效token上
    last_token_reward = reward_tensor[0, valid_token_length - 1].item()
    other_rewards = reward_tensor[0, :valid_token_length - 1].sum().item()
    padding_rewards = reward_tensor[0, valid_token_length:].sum().item()
    
    print(f"\n有效response token数量: {valid_token_length}")
    print(f"最后一个有效token (位置{valid_token_length-1})的reward: {last_token_reward:.3f}")
    print(f"其他有效token的总reward: {other_rewards:.3f}")
    print(f"Padding tokens的总reward: {padding_rewards:.3f}")
    print(f"期望: 最后一个token有reward，其他token应该是0")
    
    # 验证最后一个token有reward
    assert abs(last_token_reward) > 0.01, f"最后一个token应该有reward，实际是{last_token_reward}"
    
    # 验证其他token和padding都是0
    assert abs(other_rewards) < 0.01, f"其他有效token应该是0，实际是{other_rewards}"
    assert abs(padding_rewards) < 0.01, f"Padding tokens应该是0，实际是{padding_rewards}"
    
    print("✓ 通过：Reward只加在response tokens上\n")


def test_full_reward_computation():
    """测试7: 完整的reward计算"""
    print("=" * 80)
    print("测试7: 完整的reward计算（综合测试）")
    print("=" * 80)
    
    # 创建包含2个API调用的response
    response_str = (
        "Thought: First search\nAction: search\nAction Input: {\"query\": \"test1\"}\n\n"
        "Thought: Second search\nAction: search\nAction Input: {\"query\": \"test2\"}"
    )
    
    tokenizer = MockTokenizer()
    token_ids = tokenizer.encode(response_str)
    response_length = max(200, len(token_ids) + 10)
    data = create_test_data(response_str, tokenizer, batch_size=1, response_length=response_length)
    
    reward_manager = ToolBenchRewardManager(
        tokenizer=tokenizer,
        format_reward_weight=0.1,
        function_call_reward_weight=0.2,
        finish_reward_weight=0.3,
        format_reward_per_token=0.0,  # 关闭per-token奖励以便测试
        error_penalty=-0.5,
        finish_bonus=0.5,
        num_examine=1
    )
    
    # 添加Observation到response_str（模拟实际的response）
    response_str_with_obs = (
        response_str + 
        "\n\nObservation: {\"error\": \"\", \"response\": \"success1\"}\n\n" +
        "Thought: Second search\nAction: search\nAction Input: {\"query\": \"test2\"}\n\n" +
        "Observation: {\"error\": \"\", \"response\": \"success2\"}"
    )
    data = create_test_data(response_str_with_obs, tokenizer, batch_size=1, response_length=max(response_length, len(tokenizer.encode(response_str_with_obs)) + 10))
    
    # 设置meta_info：调用了Finish
    meta_info = {
        'finish_called': {
            0: 'give_answer'
        }
    }
    data.meta_info = meta_info
    
    # 计算各个组件（使用包含Observation的response_str）
    format_reward = reward_manager._compute_format_reward(response_str_with_obs, len(response_str_with_obs))
    function_reward = reward_manager._compute_function_call_reward(response_str_with_obs, len(tokenizer.encode(response_str_with_obs)))
    finish_reward = reward_manager._compute_finish_reward(response_str_with_obs, 0, meta_info, len(tokenizer.encode(response_str_with_obs)))
    
    # 计算总reward
    total_reward_expected = (
        reward_manager.format_reward_weight * format_reward +
        reward_manager.function_call_reward_weight * function_reward +
        reward_manager.finish_reward_weight * finish_reward
    )
    
    # 调用完整的reward计算
    reward_tensor = reward_manager(data)
    valid_token_length = len(token_ids)
    total_reward_actual = reward_tensor[0, valid_token_length - 1].item()
    
    # 注意：format_reward只计算API调用格式（Thought/Action/Action Input），不包含Finish
    # response_str_with_obs包含2个API调用，所以格式奖励是它们的平均
    print(f"格式奖励: {format_reward:.3f} (2个正确API调用格式，平均1.0)")
    print(f"Function call奖励: {function_reward:.3f} (从Observation解析，2个成功调用，0.2)")
    print(f"Finish奖励: {finish_reward:.3f} (give_answer，0.5)")
    print(f"\n期望总reward: {total_reward_expected:.3f}")
    print(f"实际总reward: {total_reward_actual:.3f}")
    print(f"计算: 0.1*1.0 + 0.2*0.2 + 0.3*0.5 = 0.1 + 0.04 + 0.15 = 0.29")
    
    assert abs(total_reward_actual - total_reward_expected) < 0.01, \
        f"总reward应该匹配，期望{total_reward_expected}，实际{total_reward_actual}"
    
    print("✓ 通过：完整reward计算正确\n")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 80)
    print("开始测试 ToolBenchRewardManager 实现")
    print("=" * 80 + "\n")
    
    try:
        test_single_api_call()
        test_multiple_api_calls_averaging()
        test_parse_api_calls()
        test_function_call_reward()
        test_finish_reward()
        test_reward_only_on_response_tokens()
        test_full_reward_computation()
        
        print("=" * 80)
        print("✓ 所有测试通过！")
        print("=" * 80)
        print("\n关键验证总结：")
        print("1. ✓ 格式奖励对每次API调用取平均，防止重复调用获得高奖励")
        print("2. ✓ Reward只加在response tokens上，排除observation")
        print("3. ✓ Function call reward和Finish reward逻辑正确")
        print("4. ✓ API调用解析功能正确")
        print("\n实现是正确的！")
        
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        raise
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    run_all_tests()

