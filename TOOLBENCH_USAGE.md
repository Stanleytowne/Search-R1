# ToolBench Integration for Search-R1

本文档说明如何使用修改后的Search-R1框架来训练LLM的tool use能力，使用StableToolBench的API模拟服务器。

## 概述

Search-R1原本只支持调用search API，现在已扩展为支持调用StableToolBench中定义的任何API。训练时，模型的prompt格式与StableToolBench的格式相同（参考`toolllama_G123_dfs_eval.json`）。

## 主要修改

1. **GenerationConfig**: 添加了toolbench相关配置项
2. **postprocess_predictions**: 支持解析StableToolBench格式（Thought/Action/Action Input）
3. **execute_predictions**: 支持调用StableToolBench的API server
4. **_postprocess_responses**: 支持StableToolBench格式的停止条件

## 配置说明

在训练脚本中，需要添加以下配置参数：

```bash
use_toolbench=true \
toolbench_url="http://127.0.0.1:8000" \
toolbench_key="your_toolbench_key" \
default_category="G1_category" \
```

### 配置参数说明

- `use_toolbench`: 是否启用ToolBench模式（默认: false）
- `toolbench_url`: StableToolBench服务器的URL（例如: "http://127.0.0.1:8000"）
- `toolbench_key`: ToolBench API密钥（如果需要）
- `default_category`: 默认的API类别（例如: "G1_category"）

## 使用步骤

### 1. 启动StableToolBench服务器

首先，需要启动StableToolBench的API模拟服务器：

```bash
cd StableToolBench/server
python main_mirrorapi_cache.py
# 或者
python main.py
```

服务器默认运行在 `http://127.0.0.1:8000`

### 2. 准备训练数据

训练数据应该使用StableToolBench的格式，即包含`conversations`数组，其中：
- `system`: 包含所有可用的API信息
- `user`: 用户查询
- `assistant`: 模型的回复（格式：Thought/Action/Action Input）
- `function`: API返回的结果

### 3. 运行训练

在训练脚本中添加toolbench相关配置：

```bash
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    # ... 其他配置 ...
    use_toolbench=true \
    toolbench_url="http://127.0.0.1:8000" \
    toolbench_key="" \
    default_category="G1_category" \
    # ... 其他配置 ...
```

## Prompt格式

模型在训练时会接收到以下格式的prompt（与StableToolBench相同）：

```
System: You are AutoGPT, you can use many tools(functions) to do the following task.
...
You have access of the following tools:
1.tool_name: Tool description

Specifically, you have access to the following APIs: [{'name': 'api_name_for_tool_name', ...}]

User: [user query]
Begin!

Assistant: 
Thought: [model's thought]
Action: [api_name]
Action Input: [json parameters]

Function: {"error": "", "response": "..."}
```

## API调用流程

1. 模型生成包含Thought、Action和Action Input的回复
2. `postprocess_predictions`解析Action和Action Input
3. `execute_predictions`调用StableToolBench服务器
4. 服务器返回模拟的API响应
5. 响应被格式化为function消息，添加到对话历史中
6. 模型继续生成下一步的回复

## 注意事项

1. **API名称格式**: StableToolBench使用`api_name_for_tool_name`格式，代码会自动解析
2. **Category**: 默认使用`G1_category`，可以通过`default_category`配置修改
3. **Finish函数**: 模型必须调用`Finish`函数来结束任务，参数为`{"return_type": "give_answer"}`或`{"return_type": "give_up_and_restart"}`
4. **错误处理**: 如果API调用失败或解析错误，会返回相应的错误消息

## 示例

完整的训练命令示例：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export DATA_DIR='data/toolbench_train'

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
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
    # ... 其他训练配置 ...
```

## 兼容性

- 当`use_toolbench=false`时，代码行为与原始Search-R1相同
- 可以无缝切换search模式和toolbench模式