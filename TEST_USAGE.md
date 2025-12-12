# Search-R1 ToolBench集成测试使用说明

## 功能说明

`test_toolbench_integration.py` 用于测试Search-R1的ToolBench集成是否正确实现，包括：
- `postprocess_predictions` 函数测试
- API调用格式测试
- Function response格式测试
- StableToolBench格式兼容性测试

## 使用方法

### 基本用法

```bash
cd Search-R1
python test_toolbench_integration.py
```

### 测试内容

#### 1. postprocess_predictions测试
测试模型输出的解析是否正确：
- 标准格式（Thought/Action/Action Input）
- 多行JSON解析
- Finish函数处理
- 无效格式处理

#### 2. API调用格式测试
测试API名称解析：
- `api_name_for_tool_name` 格式解析
- Tool name和API name提取
- Finish函数处理

#### 3. Function response格式测试
测试Function response格式：
- JSON字符串格式
- Error和response字段
- 无额外换行符

#### 4. StableToolBench格式兼容性测试
测试与StableToolBench数据格式的兼容性：
- Conversation格式
- Function response格式
- Assistant输出格式

## 测试输出

测试会输出详细的测试结果：

```
================================================================================
Testing postprocess_predictions...
================================================================================

Test: Standard format
  ✓ Action: test_api_for_tool
  ✓ Action Input: {'param': 'value'}

Test: Multi-line JSON
  ✓ Action: api_name
  ✓ Action Input: {'key1': 'value1', 'key2': 123}

...

✓ All postprocess_predictions tests passed!
```

## 注意事项

1. 测试使用模拟的tokenizer和actor rollout，不需要实际模型
2. 测试会检查格式是否正确，但不会实际调用API服务器
3. 如果测试失败，会显示详细的错误信息
4. 所有测试通过表示集成实现正确

## 故障排除

如果测试失败：

1. **Import错误**: 确保在Search-R1目录下运行，且路径正确
2. **格式错误**: 检查代码实现是否与StableToolBench格式一致
3. **解析错误**: 检查正则表达式和JSON解析逻辑

## 扩展测试

可以添加更多测试用例：

```python
# 在test_postprocess_predictions函数中添加
test_cases.append({
    "name": "Your test case",
    "input": "Your input",
    "expected_action": "expected",
    "expected_input": {}
})
```
