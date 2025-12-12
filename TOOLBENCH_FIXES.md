# ToolBench Integration Fixes

根据对StableToolBench推理代码的仔细分析，发现并修复了以下问题：

## 发现的问题

### 1. Function Response格式问题
**问题**：之前返回的function response格式为 `\n{json}\n`，但StableToolBench中function response应该是纯JSON字符串。

**修复**：改为直接返回JSON字符串，不添加额外的换行符：
```python
function_response = json.dumps({"error": error, "response": response}, ensure_ascii=False)
next_obs.append(function_response)  # 不再添加 \n
```

### 2. Action Input解析问题
**问题**：使用正则表达式 `r'Action Input:\s*(\{.*?\})'` 无法正确处理多行JSON，且与StableToolBench的react_parser实现不一致。

**修复**：改用与StableToolBench相同的解析逻辑（使用`find`方法）：
```python
thought_start = prediction.find("Thought: ")
action_start = prediction.find("\nAction: ")
action_input_start = prediction.find("\nAction Input: ")

if thought_start != -1 and action_start != -1 and action_input_start != -1:
    action_name = prediction[action_start + len("\nAction: "):action_input_start].strip()
    action_input_str = prediction[action_input_start + len("\nAction Input: "):].strip()
```

### 3. JSON解析改进
**问题**：对于多行JSON或格式不规范的JSON，解析可能失败。

**修复**：添加了更健壮的JSON解析逻辑：
- 首先尝试直接解析
- 如果失败，尝试找到完整的JSON对象（通过匹配大括号）
- 最后才使用正则表达式提取key-value对

### 4. Response停止条件改进
**问题**：`_postprocess_responses`中的正则表达式无法正确处理嵌套的JSON对象。

**修复**：改用大括号计数来找到完整的JSON对象：
```python
# 找到"Action Input:"后的JSON对象
brace_start = action_input_str.find('{')
if brace_start != -1:
    brace_count = 0
    for i in range(brace_start, len(action_input_str)):
        if action_input_str[i] == '{':
            brace_count += 1
        elif action_input_str[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                json_end = i + 1
                break
```

## StableToolBench格式说明

根据代码分析，StableToolBench的格式如下：

### 1. 模型输入格式
在ToolLLaMA的`parse`方法中，prompt的构建方式是：
```
System: {system_message}
User: {user_message}
Function: {function_response_json}
Assistant:
```

其中：
- `System`、`User`、`Function`、`Assistant`是conversation template中定义的角色
- `Function`角色的content是JSON字符串：`{"error": "", "response": "..."}`

### 2. 模型输出格式
模型应该生成：
```
Thought: {thought}
Action: {action_name}
Action Input: {json_object}
```

### 3. Function Response格式
Function response是一个JSON字符串，格式为：
```json
{"error": "", "response": "..."}
```

这个字符串会被添加到conversation history中作为`Function`角色的content。

## 验证要点

1. ✅ Function response格式：纯JSON字符串，无额外换行
2. ✅ Action Input解析：使用find方法，与StableToolBench一致
3. ✅ JSON解析：支持多行和嵌套JSON
4. ✅ Response停止：正确识别完整的Action Input JSON

## 注意事项

1. **Category提取**：目前使用默认category（`G1_category`），实际使用时可能需要从数据中提取
2. **Tool name提取**：从`api_name_for_tool_name`格式中提取tool_name，使用`rsplit('_for_', 1)`
3. **API name**：传递给ToolBench server的`api_name`应该是完整的action_name（包含`_for_`部分）
