import os
import json
import glob
from vllm import LLM, SamplingParams

# ================= 配置区域 =================
# 输入文件夹路径 (存放原始 category.json 文件)
INPUT_DIR = "data/api_infos"

# 输出文件夹路径 (存放生成结果)
OUTPUT_DIR = "data/api_infos_processed"

# 模型路径 (可以是 HuggingFace ID 或 本地路径)
MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"  # 示例，请替换为你实际使用的模型

# vLLM 参数设置
TENSOR_PARALLEL_SIZE = 1  # 如果有多张显卡，可以设置为显卡数量
MAX_TOKENS = 2048         # 生成的最大 token 数
TEMPERATURE = 0.7         # 温度采样

# ================= PROMPT TEMPLATES =================
PROMPT_TEMPLATES = [
    """Transform the following API JSON into a coherent, natural language paragraph describing how to use it.

The description must act as a complete guide and include:
- The API's specific functionality.
- A detailed breakdown of the required and optional arguments, including their expected types (e.g., number, string).
- A clear explanation of what the API returns, describing the structure of the response object and the data types of its fields.

Weave the information into smooth, logical sentences. Ensure strictly all information from the JSON is preserved in the text.

Input JSON:
{JSON_DATA}

Your description:
""",
    """You are a technical documentation expert. Your task is to convert the provided API JSON definition into a comprehensive, natural language technical reference. 

Please follow these rules:
1. **Context**: Start by stating the API name clearly.
2. **Purpose**: Explain what the API does based on its description.
3. **Inputs**: Detail every parameter. Specify the data type, whether it is required, and its description.
4. **Outputs**: Describe the response structure based on the `template_response`, explaining the fields and their data types.

Ensure no technical detail (like types or constraints) is omitted. The output should be concise yet complete, suitable for a developer or an LLM to understand how to invoke this tool.

Input JSON:
{JSON_DATA}

Your technical reference:
""",
    """Convert the provided API definition into a structured natural language summary optimized for model training.

Format the output into three distinct sections:
1. **Identity**: explicitly state the 'API Name' found in the JSON key or name field.
2. **Intent**: What user goal does this API serve? (Derive this from the description).
3. **Action**: How is the API called? Describe the parameters, emphasizing which are mandatory and their specific constraints (e.g., "a number between 1 and 5").
4. **Result**: What is the exact schema of the returned data? Describe the nested fields and their types found in the `template_response`.

Keep the language precise and factual.

Input JSON:
{JSON_DATA}

Your summary:
"""
]

def main():
    # 1. 初始化 LLM
    print(f"Loading model from {MODEL_PATH}...")
    llm = LLM(model=MODEL_PATH, tensor_parallel_size=TENSOR_PARALLEL_SIZE)
    sampling_params = SamplingParams(temperature=TEMPERATURE, max_tokens=MAX_TOKENS)

    # 2. 准备输出目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 3. 获取所有 JSON 文件
    json_files = glob.glob(os.path.join(INPUT_DIR, "*.json"))
    print(f"Found {len(json_files)} JSON files.")

    for file_path in json_files:
        file_name = os.path.basename(file_path)
        output_path = os.path.join(OUTPUT_DIR, f"{file_name}")
        
        print(f"Processing file: {file_name}...")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 假设 data 是一个 dict，类似 {"api_name": {...}, "api_name_2": {...}}
            # 如果 data 本身就是 list，请修改下面的 items() 逻辑
            
            prompts = []
            metadata = [] # 用于记录 prompt 对应哪个 API 和哪个模板

            # 遍历文件中的每一个 API
            # 如果你的 JSON 根是 List，请用 `for api_item in data:`
            # 如果你的 JSON 根是 Dict，如下所示：
            for api_name, api_content in data.items():
                
                # 将该 API 的 JSON 转为字符串
                json_str = json.dumps(api_content, indent=2, ensure_ascii=False)

                # 为该 API 生成 3 个 prompt
                for i, template in enumerate(PROMPT_TEMPLATES):
                    # 使用 replace 避免 format 对 json 内部 {} 的报错
                    prompt_text = template.replace("{JSON_DATA}", json_str)

                    prompt_text = llm.get_tokenizer().apply_chat_template(
                        [
                            {"role": "user", "content": prompt_text}
                        ], 
                        tokenize=False, 
                        add_generation_prompt=True
                    )
                    
                    prompts.append(prompt_text)
                    metadata.append({
                        "api_name": api_name,
                        "template_id": i,
                        "original_content": api_content
                    })

            if not prompts:
                print(f"No APIs found in {file_name}, skipping.")
                continue

            # 4. 执行批量推理
            print(f"Generating {len(prompts)} responses for {file_name}...")
            outputs = llm.generate(prompts, sampling_params)

            # 5. 整理结果
            # 我们将结果重组为：每个 API 包含原始数据和 3 个生成的描述
            results_map = {} 

            for i, output in enumerate(outputs):
                meta = metadata[i]
                generated_text = output.outputs[0].text.strip()
                api_name = meta['api_name']

                if api_name not in results_map:
                    results_map[api_name] = {
                        "api_name": api_name,
                        "original_json": meta['original_content'],
                        "generations": {}
                    }
                
                # 保存生成结果，key 为 template_0, template_1, template_2
                results_map[api_name]["generations"][f"template_{meta['template_id']}"] = generated_text

            # 将 map 转回 list 以便保存 (或者保持 dict，看你需要)
            final_output = list(results_map.values())

            # 6. 保存到新文件
            with open(output_path, 'w', encoding='utf-8') as f_out:
                json.dump(final_output, f_out, indent=2, ensure_ascii=False)
            
            print(f"Saved processed data to {output_path}")

        except Exception as e:
            print(f"Error processing {file_name}: {e}")

if __name__ == "__main__":
    main()