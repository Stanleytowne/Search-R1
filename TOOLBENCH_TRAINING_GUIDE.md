# ToolBench RL训练完整指南

## 概述

本指南介绍如何准备数据并开始ToolBench模式的RL训练。

## 步骤1: 数据准备

### 1.1 转换数据格式

将StableToolBench的JSON数据转换为verl所需的parquet格式：

```bash
# 基本转换（单文件）
python scripts/convert_toolbench_to_verl.py \
    --input StableToolBench/data/toolllama_G123_dfs_eval.json \
    --output ~/data/toolbench/toolbench.parquet

# 分割为训练集和验证集（推荐）
python scripts/convert_toolbench_to_verl.py \
    --input StableToolBench/data/toolllama_G123_dfs_eval.json \
    --split \
    --train_output ~/data/toolbench/train.parquet \
    --val_output ~/data/toolbench/val.parquet \
    --train_ratio 0.9

# 测试模式（只处理前100个样本）
python scripts/convert_toolbench_to_verl.py \
    --input StableToolBench/data/toolllama_G123_dfs_eval.json \
    --output ~/data/toolbench/test.parquet \
    --max_samples 100
```

### 1.2 数据格式说明

转换后的parquet文件包含以下列：

- **prompt**: Chat格式的对话列表
  ```python
  [
      {"role": "system", "from": "system", "content": "..."},
      {"role": "user", "from": "user", "content": "..."},
      {"role": "assistant", "from": "assistant", "content": "..."},
      {"role": "function", "from": "function", "content": "..."}
  ]
  ```

- **data_source**: 数据来源标识（"toolbench"）

- **reward_model**: Reward相关信息
  ```python
  {
      "style": "function",
      "ground_truth": None
  }
  ```

- **extra_info**: 额外信息
  ```python
  {
      "index": 0,
      "sample_id": "...",
      "category": "G1_category"
  }
  ```

## 步骤2: 启动ToolBench服务器

在开始训练前，需要启动ToolBench API模拟服务器：

```bash
cd StableToolBench/server
bash start_server.sh
```

服务器默认运行在 `http://127.0.0.1:8000`

## 步骤3: 配置训练参数

### 3.1 修改配置文件

**对于GRPO训练**，编辑 `verl/trainer/config/grpo_toolbench_trainer.yaml`：

```yaml
data:
  train_files: ~/data/toolbench/train.parquet
  val_files: ~/data/toolbench/val.parquet
  max_prompt_length: 4096
  max_response_length: 500

actor_rollout_ref:
  model:
    path: ~/models/your-model-path
  actor:
    use_kl_loss: true  # GRPO需要KL loss
    kl_loss_coef: 0.001
  rollout:
    n_agent: 5  # GRPO需要多个响应

# ToolBench配置
use_toolbench: true
toolbench_url: "http://127.0.0.1:8000"
toolbench_key: ""
default_category: "G1_category"

# GRPO配置
algorithm:
  adv_estimator: grpo  # 使用GRPO而不是GAE

# Reward权重
reward_model:
  format_reward_weight: 0.1
  function_call_reward_weight: 0.2
  finish_reward_weight: 0.3
  error_penalty: -0.5
  finish_bonus: 0.5
```

**注意**: GRPO与PPO的主要区别：
- `algorithm.adv_estimator=grpo` (而不是gae)
- `actor.use_kl_loss=true` (GRPO需要KL loss)
- `rollout.n_agent=5` (每个prompt生成多个响应)
- 不需要critic（GRPO是policy-only方法）

### 3.2 环境变量配置

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
```

## 步骤4: 开始训练

### 4.1 GRPO训练（推荐）

```bash
# 基本使用
bash scripts/train_toolbench_grpo.sh

# 自定义参数
DATA_DIR=~/data/toolbench \
MODEL_PATH=~/models/your-model \
TOOLBENCH_URL=http://127.0.0.1:8000 \
N_AGENT=5 \
bash scripts/train_toolbench_grpo.sh
```

### 4.2 直接使用Python（GRPO）

```bash
python -m verl.trainer.main_ppo_toolbench \
    --config-path verl/trainer/config \
    --config-name grpo_toolbench_trainer \
    data.train_files=~/data/toolbench/train.parquet \
    data.val_files=~/data/toolbench/val.parquet \
    use_toolbench=true \
    toolbench_url="http://127.0.0.1:8000" \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.rollout.n_agent=5 \
    reward_model.format_reward_weight=0.1 \
    reward_model.function_call_reward_weight=0.2 \
    reward_model.finish_reward_weight=0.3
```

### 4.3 PPO训练（如果需要）

```bash
# 使用PPO训练脚本
bash scripts/train_toolbench_ppo.sh
```

## 步骤5: 监控训练

训练过程中会输出以下信息：

- **格式奖励**: 模型生成正确格式的奖励
- **Function call奖励**: API调用成功/失败的奖励
- **Finish奖励**: 是否调用Finish函数的奖励
- **总Reward**: 组合后的总奖励

可以通过wandb或其他logger查看训练曲线。

## 常见问题

### Q1: 数据转换失败

**问题**: `convert_toolbench_to_verl.py` 报错

**解决**:
1. 检查输入JSON文件格式是否正确
2. 确保安装了必要的依赖：`pandas`, `pyarrow`
3. 检查文件路径是否正确

### Q2: ToolBench服务器连接失败

**问题**: 训练时无法连接到ToolBench服务器

**解决**:
1. 确认服务器已启动：`curl http://127.0.0.1:8000/health`
2. 检查`toolbench_url`配置是否正确
3. 确认防火墙设置允许连接

### Q3: 内存不足

**问题**: OOM错误

**解决**:
1. 减小`train_batch_size`
2. 减小`max_prompt_length`和`max_response_length`
3. 使用更少的GPU或启用gradient checkpointing

### Q4: Reward始终为0

**问题**: 训练时reward没有变化

**解决**:
1. 检查`meta_info`是否正确传递（查看日志）
2. 确认API调用是否成功（检查ToolBench服务器日志）
3. 检查reward权重配置是否正确

## 训练检查清单

- [ ] 数据已转换为parquet格式
- [ ] ToolBench服务器已启动
- [ ] 配置文件已正确设置
- [ ] 模型路径正确
- [ ] GPU环境已配置
- [ ] Reward权重已调整
- [ ] 训练脚本可执行

## 下一步

训练完成后，可以：

1. **评估模型**: 使用验证集评估模型性能
2. **调整Reward**: 根据训练结果调整reward权重
3. **扩展数据**: 添加更多训练数据
4. **微调参数**: 调整学习率、batch size等超参数

## GRPO vs PPO

### GRPO特点
- **Policy-only方法**: 不需要critic网络
- **多个响应**: 每个prompt生成多个响应（`n_agent`参数）
- **KL loss**: 需要启用KL loss来约束策略更新
- **更高效**: 通常比PPO更高效，适合大规模训练

### 关键配置差异

| 配置项 | GRPO | PPO |
|--------|------|-----|
| `algorithm.adv_estimator` | `grpo` | `gae` |
| `actor.use_kl_loss` | `true` | `false` |
| `rollout.n_agent` | `5` (多个响应) | `1` |
| Critic | 不需要 | 需要 |

## 参考文档

- `TOOLBENCH_REWARD_USAGE.md`: Reward函数使用说明
- `TOOLBENCH_REWARD_IMPLEMENTATION.md`: Reward实现细节
- `verl/trainer/config/grpo_toolbench_trainer.yaml`: GRPO训练配置示例
- `verl/trainer/config/ppo_toolbench_trainer.yaml`: PPO训练配置示例（如果需要）
