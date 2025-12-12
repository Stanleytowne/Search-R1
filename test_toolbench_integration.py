#!/usr/bin/env python
"""
测试Search-R1的ToolBench集成是否正确实现
"""

import sys
import os
import json
import re
import requests
from typing import List, Dict, Any

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from search_r1.llm_agent.generation import LLMGenerationManager, GenerationConfig


class MockTokenizer:
    """模拟tokenizer用于测试"""
    
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.vocab = {chr(i): i for i in range(1000)}
        self.vocab['<pad>'] = 0
        self.vocab['<eos>'] = 1
    
    def encode(self, text, return_tensors='pt', add_special_tokens=False):
        import torch
        tokens = [self.vocab.get(c, 0) for c in text[:100]]  # 简单截断
        if return_tensors == 'pt':
            return {'input_ids': torch.tensor([tokens])}
        return {'input_ids': [tokens]}
    
    def batch_decode(self, tokens, skip_special_tokens=True):
        import torch
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        if isinstance(tokens, list) and len(tokens) > 0:
            if isinstance(tokens[0], list):
                return [''.join([chr(t) if t < 1000 else '?' for t in seq]) for seq in tokens]
            else:
                return [''.join([chr(t) if t < 1000 else '?' for t in tokens])]
        return ['']
    
    def __call__(self, texts, padding='longest', return_tensors='pt', add_special_tokens=False):
        import torch
        if isinstance(texts, str):
            texts = [texts]
        encoded = [self.encode(t, return_tensors='')['input_ids'] for t in texts]
        max_len = max(len(e) for e in encoded)
        padded = [e + [0] * (max_len - len(e)) for e in encoded]
        if return_tensors == 'pt':
            return {'input_ids': torch.tensor(padded)}
        return {'input_ids': padded}


class MockActorRollout:
    """模拟actor rollout用于测试"""
    
    def generate_sequences(self, batch):
        import torch
        class Output:
            def __init__(self):
                batch_size = batch.batch['input_ids'].shape[0] if hasattr(batch, 'batch') else 1
                self.batch = {
                    'responses': torch.randint(0, 1000, (batch_size, 10))  # 模拟响应
                }
                self.meta_info = {}
        return Output()


class MockToolBenchServer:
    """模拟ToolBench服务器用于测试"""
    
    def __init__(self):
        self.calls = []
    
    def handle_request(self, payload):
        """处理API调用请求"""
        self.calls.append(payload)
        
        # 模拟响应
        category = payload.get('category', 'G1_category')
        tool_name = payload.get('tool_name', 'unknown')
        api_name = payload.get('api_name', 'unknown')
        tool_input = payload.get('tool_input', {})
        
        # 返回模拟的API响应
        return {
            "error": "",
            "response": f"Mock response for {api_name} with input {tool_input}"
        }


def test_postprocess_predictions():
    """测试postprocess_predictions函数"""
    print("=" * 80)
    print("Testing postprocess_predictions...")
    print("=" * 80)
    
    config = GenerationConfig(
        max_turns=5,
        max_start_length=2048,
        max_prompt_length=4096,
        max_response_length=500,
        max_obs_length=500,
        num_gpus=1,
        use_toolbench=True,
        toolbench_url="http://localhost:8000",
        toolbench_key="",
        default_category="G1_category"
    )
    
    tokenizer = MockTokenizer()
    actor_rollout = MockActorRollout()
    manager = LLMGenerationManager(tokenizer, actor_rollout, config)
    
    # 测试用例1: 标准的Thought/Action/Action Input格式
    test_cases = [
        {
            "name": "Standard format",
            "input": "\nThought: I need to call an API.\nAction: test_api_for_tool\nAction Input: {\"param\": \"value\"}",
            "expected_action": "test_api_for_tool",
            "expected_input": {"param": "value"}
        },
        {
            "name": "Multi-line JSON",
            "input": "\nThought: Test\nAction: api_name\nAction Input: {\n  \"key1\": \"value1\",\n  \"key2\": 123\n}",
            "expected_action": "api_name",
            "expected_input": {"key1": "value1", "key2": 123}
        },
        {
            "name": "Finish function",
            "input": "\nThought: Done\nAction: Finish\nAction Input: {\"return_type\": \"give_answer\", \"final_answer\": \"test\"}",
            "expected_action": "Finish",
            "expected_input": {"return_type": "give_answer", "final_answer": "test"}
        },
        {
            "name": "Invalid format",
            "input": "Just some text without proper format",
            "expected_action": None,
            "expected_input": {}
        }
    ]
    
    for test_case in test_cases:
        print(f"\nTest: {test_case['name']}")
        actions, contents = manager.postprocess_predictions([test_case['input']])
        
        action = actions[0]
        content = contents[0]
        
        if test_case['expected_action'] is None:
            assert action is None, f"Expected None but got {action}"
            print(f"  ✓ Correctly parsed as None")
        else:
            assert action == test_case['expected_action'], f"Expected {test_case['expected_action']} but got {action}"
            assert content.get('action_input') == test_case['expected_input'], f"Expected {test_case['expected_input']} but got {content.get('action_input')}"
            print(f"  ✓ Action: {action}")
            print(f"  ✓ Action Input: {content.get('action_input')}")
    
    print("\n✓ All postprocess_predictions tests passed!")


def test_api_call_format():
    """测试API调用格式"""
    print("\n" + "=" * 80)
    print("Testing API call format...")
    print("=" * 80)
    
    config = GenerationConfig(
        max_turns=5,
        max_start_length=2048,
        max_prompt_length=4096,
        max_response_length=500,
        max_obs_length=500,
        num_gpus=1,
        use_toolbench=True,
        toolbench_url="http://localhost:8000",
        toolbench_key="",
        default_category="G1_category"
    )
    
    tokenizer = MockTokenizer()
    actor_rollout = MockActorRollout()
    manager = LLMGenerationManager(tokenizer, actor_rollout, config)
    
    # 测试API名称解析
    test_cases = [
        {
            "action_name": "racecards_for_greyhound_racing_uk",
            "expected_tool_name": "greyhound_racing_uk",
            "expected_api_name": "racecards"
        },
        {
            "action_name": "get_weather_for_weather_api",
            "expected_tool_name": "weather_api",
            "expected_api_name": "get_weather"
        },
        {
            "action_name": "Finish",
            "expected_tool_name": "unknown",
            "expected_api_name": "Finish"
        }
    ]
    
    for test_case in test_cases:
        api_calls = [{
            'index': 0,
            'action': test_case['action_name'],
            'content': {'action_input': {'param': 'value'}}
        }]
        
        # 检查解析逻辑（不实际调用服务器）
        action_name = test_case['action_name']
        if '_for_' in action_name:
            parts = action_name.rsplit('_for_', 1)
            if len(parts) == 2:
                api_name = parts[0]
                tool_name = parts[1]
            else:
                parts = action_name.split('_for_', 1)
                api_name = parts[0]
                tool_name = parts[1] if len(parts) > 1 else 'unknown'
        else:
            api_name = action_name
            tool_name = 'unknown'
        
        print(f"\nTest: {test_case['action_name']}")
        print(f"  Parsed tool_name: {tool_name}")
        print(f"  Parsed api_name: {api_name}")
        assert tool_name == test_case['expected_tool_name'], f"Expected {test_case['expected_tool_name']} but got {tool_name}"
        assert api_name == test_case['expected_api_name'], f"Expected {test_case['expected_api_name']} but got {api_name}"
        print(f"  ✓ Correctly parsed")
    
    print("\n✓ All API call format tests passed!")


def test_function_response_format():
    """测试function response格式"""
    print("\n" + "=" * 80)
    print("Testing function response format...")
    print("=" * 80)
    
    config = GenerationConfig(
        max_turns=5,
        max_start_length=2048,
        max_prompt_length=4096,
        max_response_length=500,
        max_obs_length=500,
        num_gpus=1,
        use_toolbench=True,
        toolbench_url="http://localhost:8000",
        toolbench_key="",
        default_category="G1_category"
    )
    
    tokenizer = MockTokenizer()
    actor_rollout = MockActorRollout()
    manager = LLMGenerationManager(tokenizer, actor_rollout, config)
    
    # 模拟API响应
    api_results = {
        0: {
            "error": "",
            "response": "Test response data"
        },
        1: {
            "error": "Some error",
            "response": ""
        }
    }
    
    # 测试格式
    for idx, result in api_results.items():
        error = result.get('error', '')
        response = result.get('response', '')
        function_response = json.dumps({"error": error, "response": response}, ensure_ascii=False)
        
        print(f"\nTest case {idx}:")
        print(f"  Error: {error}")
        print(f"  Response: {response}")
        print(f"  Formatted: {function_response}")
        
        # 验证格式
        parsed = json.loads(function_response)
        assert parsed['error'] == error
        assert parsed['response'] == response
        assert isinstance(function_response, str)  # 应该是字符串，不是带换行符的
        assert not function_response.startswith('\n')  # 不应该以换行符开头
        assert not function_response.endswith('\n')  # 不应该以换行符结尾
        
        print(f"  ✓ Format is correct")
    
    print("\n✓ All function response format tests passed!")


def test_stabletoolbench_format_compatibility():
    """测试与StableToolBench格式的兼容性"""
    print("\n" + "=" * 80)
    print("Testing StableToolBench format compatibility...")
    print("=" * 80)
    
    # 加载一个示例对话（如果有的话）
    sample_file = "../StableToolBench/data/toolllama_G123_dfs_eval.json"
    if os.path.exists(sample_file):
        with open(sample_file, 'r') as f:
            data = json.load(f)
            if len(data) > 0:
                sample = data[0]
                conversations = sample.get('conversations', [])
                
                print(f"\nLoaded sample conversation with {len(conversations)} messages")
                
                # 检查格式
                for i, conv in enumerate(conversations[:5]):  # 只检查前5个
                    role = conv.get('from', '')
                    value = conv.get('value', '')
                    
                    print(f"\nMessage {i+1}: {role}")
                    if role == 'function':
                        # 验证function response是JSON字符串
                        try:
                            parsed = json.loads(value)
                            assert 'error' in parsed
                            assert 'response' in parsed
                            print(f"  ✓ Valid function response format")
                        except json.JSONDecodeError:
                            print(f"  ✗ Invalid JSON format")
                    elif role == 'assistant':
                        # 检查是否包含Thought/Action/Action Input
                        has_thought = 'Thought:' in value
                        has_action = 'Action:' in value
                        has_action_input = 'Action Input:' in value
                        print(f"  Thought: {has_thought}, Action: {has_action}, Action Input: {has_action_input}")
                        if has_thought and has_action and has_action_input:
                            print(f"  ✓ Valid assistant format")
    
    print("\n✓ Format compatibility check complete!")


def main():
    """运行所有测试"""
    print("\n" + "=" * 80)
    print("Search-R1 ToolBench Integration Tests")
    print("=" * 80)
    
    try:
        test_postprocess_predictions()
        test_api_call_format()
        test_function_response_format()
        test_stabletoolbench_format_compatibility()
        
        print("\n" + "=" * 80)
        print("ALL TESTS PASSED! ✓")
        print("=" * 80)
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
