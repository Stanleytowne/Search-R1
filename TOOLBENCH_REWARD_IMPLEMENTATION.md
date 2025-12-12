# ToolBench Reward实现说明

## 概述

实现了ToolBench模式的reward函数，包含三个主要部分：
1. **格式奖励**：奖励模型生成正确的格式
2. **Function call奖励**：根据API调用结果给予奖励或惩罚
3. **Finish调用奖励**：奖励正确调用Finish函数

## 实现细节

### 1. 格式奖励 (Format Reward)

**位置**: `search_r1/llm_agent/toolbench_reward.py::_compute_format_reward`

**检查内容**:
- 是否包含 `Thought:`, `Action:`, `Action Input:`
- 格式顺序是否正确（Thought -> Action -> Action Input）
- Action Input是否为有效的JSON

**奖励值**:
- 完整且有效的格式: 1.0
- 格式存在但JSON无效: 0.5
- 部分格式（有Thought或Action）: 0.2
- 无格式: 0.0

**可选**: 每个token的小奖励（`format_reward_per_token`）

### 2. Function Call奖励 (Function Call Reward)

**位置**: `search_r1/llm_agent/toolbench_reward.py::_compute_function_call_reward`

**数据来源**: `meta_info['api_errors']`
- 格式: `{sample_idx: [bool, bool, ...]}`
- `True` 表示API调用有error
- `False` 表示API调用成功

**计算方式**:
- 每个成功的API调用: +0.1
- 每个失败的API调用: `error_penalty` (默认: -0.5)

**实现位置**: `search_r1/llm_agent/generation.py::execute_predictions`
- 在API调用后，将结果保存到 `self.api_call_history`
- 在 `run_llm_loop` 结束时，将 `api_call_history` 转换为 `api_errors` 并添加到 `meta_info`

### 3. Finish调用奖励 (Finish Reward)

**位置**: `search_r1/llm_agent/toolbench_reward.py::_compute_finish_reward`

**数据来源**: `meta_info['finish_called']`
- 格式: `{sample_idx: 'give_answer' | 'give_up_and_restart' | None}`

**计算方式**:
- `give_answer`: `finish_bonus` (默认: 0.5)
- `give_up_and_restart`: `finish_bonus * 0.5`
- 格式不对: `finish_bonus * 0.3`
- 未调用: 0.0

**实现位置**: `search_r1/llm_agent/generation.py::execute_predictions`
- 检测到Finish调用时，保存到 `self.finish_call_history`
- 在 `run_llm_loop` 结束时，添加到 `meta_info`

## 数据流

```
Generation Phase:
  LLMGenerationManager.run_llm_loop()
    -> execute_predictions()
      -> batch_call_toolbench_apis()  # 调用API
      -> 保存到 self.api_call_history
      -> 保存到 self.finish_call_history
    -> 添加到 meta_info['api_errors']
    -> 添加到 meta_info['finish_called']

Reward Phase:
  ToolBenchRewardManager.__call__(batch)
    -> 从 batch.meta_info 读取信息
    -> 计算格式奖励（从response字符串）
    -> 计算Function call奖励（从meta_info）
    -> 计算Finish奖励（从meta_info和response字符串）
    -> 组合reward并返回
```

## 使用示例

### 在训练脚本中使用

```python
from search_r1.llm_agent.toolbench_reward import ToolBenchRewardManager

# 创建reward函数
reward_fn = ToolBenchRewardManager(
    tokenizer=tokenizer,
    format_reward_weight=0.1,
    function_call_reward_weight=0.2,
    finish_reward_weight=0.3,
    error_penalty=-0.5,
    finish_bonus=0.5,
    num_examine=0
)
```

### 在训练命令中配置

```bash
python verl/trainer/main_ppo_toolbench.py \
    use_toolbench=true \
    reward_model.format_reward_weight=0.1 \
    reward_model.function_call_reward_weight=0.2 \
    reward_model.finish_reward_weight=0.3 \
    reward_model.error_penalty=-0.5 \
    reward_model.finish_bonus=0.5
```

## 测试

运行测试脚本验证实现：

```bash
python test_toolbench_reward.py
```

## 注意事项

1. **索引映射**: 当batch被repeat时，通过`non_tensor_batch['index']`找到原始样本索引
2. **meta_info传递**: 确保`generation.py`中的meta_info正确传递到reward函数
3. **权重调整**: 根据实际训练效果调整各部分的权重
4. **格式检查**: 格式奖励基于response字符串解析，需要tokenizer正确解码

## 未来扩展

- LLM判断reward（第4部分）：可以使用LLM评估任务完成质量
- 更细粒度的格式奖励：对每个格式组件分别奖励
- 动态权重调整：根据训练进度调整reward权重
- Token级别的reward：为每个token分配更精细的reward
