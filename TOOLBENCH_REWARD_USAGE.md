# ToolBench Reward使用说明

## 概述

ToolBench模式的reward函数包含三个主要部分：
1. **格式奖励**：奖励模型生成正确的格式（Thought/Action/Action Input）
2. **Function call奖励**：如果API调用结果有error，则惩罚
3. **Finish调用奖励**：最后一次是否调用Finish函数

## Reward组成

### 1. 格式奖励 (Format Reward)

**权重**: `format_reward_weight` (默认: 0.1)

**计算方式**:
- 完整格式（Thought + Action + Action Input，且JSON有效）: 1.0
- 部分格式（有Thought或Action但JSON无效）: 0.5
- 部分关键词: 0.2
- 无格式: 0.0

**可选**: 每个token的小奖励 (`format_reward_per_token`)

### 2. Function Call奖励 (Function Call Reward)

**权重**: `function_call_reward_weight` (默认: 0.2)

**计算方式**:
- 每个成功的API调用（无error）: +0.1
- 每个失败的API调用（有error）: `error_penalty` (默认: -0.5)

### 3. Finish调用奖励 (Finish Reward)

**权重**: `finish_reward_weight` (默认: 0.3)

**计算方式**:
- 正确调用Finish且return_type为give_answer: `finish_bonus` (默认: 0.5)
- 调用Finish但return_type为give_up_and_restart: `finish_bonus * 0.5`
- 调用Finish但格式不对: `finish_bonus * 0.3`
- 未调用Finish: 0.0

## 使用方法

### 方法1: 使用main_ppo_toolbench.py

创建新的训练脚本，使用ToolBench reward：

```bash
python verl/trainer/main_ppo_toolbench.py \
    # ... 其他配置 ...
    use_toolbench=true \
    toolbench_url="http://127.0.0.1:8000" \
    reward_model.format_reward_weight=0.1 \
    reward_model.function_call_reward_weight=0.2 \
    reward_model.finish_reward_weight=0.3 \
    reward_model.error_penalty=-0.5 \
    reward_model.finish_bonus=0.5
```

### 方法2: 在现有训练脚本中集成

修改 `verl/trainer/main_ppo.py` 或创建新版本：

```python
from search_r1.llm_agent.toolbench_reward import ToolBenchRewardManager

# 检查是否使用ToolBench模式
if config.use_toolbench:
    reward_fn = ToolBenchRewardManager(
        tokenizer=tokenizer,
        format_reward_weight=0.1,
        function_call_reward_weight=0.2,
        finish_reward_weight=0.3,
        error_penalty=-0.5,
        finish_bonus=0.5,
        num_examine=0
    )
else:
    # 使用原始的RewardManager
    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0)
```

## 配置参数

在训练配置中可以设置以下参数：

```yaml
reward_model:
  format_reward_weight: 0.1      # 格式奖励权重
  function_call_reward_weight: 0.2  # Function call奖励权重
  finish_reward_weight: 0.3      # Finish调用奖励权重
  error_penalty: -0.5            # API调用错误的惩罚
  finish_bonus: 0.5              # Finish调用的奖励
  format_reward_per_token: 0.01  # 每个token的格式奖励（可选）
```

## Reward计算流程

1. **生成阶段**: `LLMGenerationManager.run_llm_loop()` 执行生成，调用API，并将结果保存到`meta_info`中
   - `api_errors`: {sample_idx: [bool, bool, ...]} - 每个API调用是否有error
   - `finish_called`: {sample_idx: 'give_answer' | 'give_up_and_restart' | None}

2. **Reward计算阶段**: `ToolBenchRewardManager.__call__()` 从`meta_info`中读取信息并计算reward
   - 解析response字符串检查格式
   - 从meta_info读取API调用结果
   - 从meta_info读取Finish调用信息
   - 组合三个部分的reward

3. **训练阶段**: Reward用于PPO训练

## 示例

### 完整训练命令

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export DATA_DIR='data/toolbench_train'

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo_toolbench \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.max_prompt_length=4096 \
    data.max_response_length=500 \
    data.max_start_length=2048 \
    data.max_obs_length=500 \
    max_turns=5 \
    use_toolbench=true \
    toolbench_url="http://127.0.0.1:8000" \
    toolbench_key="" \
    default_category="G1_category" \
    reward_model.format_reward_weight=0.1 \
    reward_model.function_call_reward_weight=0.2 \
    reward_model.finish_reward_weight=0.3 \
    reward_model.error_penalty=-0.5 \
    reward_model.finish_bonus=0.5 \
    # ... 其他训练配置 ...
```

## 调试

设置 `num_examine > 0` 可以打印reward计算的详细信息：

```python
reward_fn = ToolBenchRewardManager(
    tokenizer=tokenizer,
    num_examine=5,  # 打印前5个样本的reward信息
    ...
)
```

输出示例：
```
[Reward Sample 0]
  Response: Thought: I need to call an API...
  Format reward: 1.000
  Function call reward: 0.100
  Finish reward: 0.500
  Total reward: 0.280
```

## 注意事项

1. **meta_info传递**: 确保`generation.py`中的meta_info正确传递到reward函数
2. **权重调整**: 根据实际训练效果调整各部分的权重
3. **Finish奖励**: 确保模型理解必须调用Finish函数
4. **Error惩罚**: 根据实际情况调整error_penalty的大小

## 未来扩展

- LLM判断reward（第4部分）：可以使用LLM评估任务完成质量
- 更细粒度的格式奖励：对每个格式组件分别奖励
- 动态权重调整：根据训练进度调整reward权重
