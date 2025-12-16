import os
import json
import glob
import random
import pandas as pd
import argparse
from vllm import LLM, SamplingParams

# ================= 1. vLLM 生成用的 Prompt (Teacher Prompts) =================
GENERATION_PROMPTS = [
    # 模板 0: 叙述性
    """Transform the following API JSON into a coherent, natural language paragraph describing how to use it.
CRITICAL: You MUST explicitly mention the API name ("{API_NAME}") and its purpose at the beginning.
Input JSON:
{JSON_DATA}
Your description:""",
    
    # 模板 1: 技术参考
    """You are a technical documentation expert. Convert the provided API JSON definition into a comprehensive technical reference.
Rules:
1. **Identity**: Explicitly state the API Name.
2. **Purpose**: Explain functionality.
3. **Inputs**: Detail parameters.
Input JSON:
{JSON_DATA}
Your technical reference:""",
    
    # 模板 2: 结构化摘要
    """Convert the provided API definition into a structured natural language summary.
Format:
1. **Tool Identifier**: The exact API name string.
2. **Intent**: User goal.
3. **Action**: Parameters.
Input JSON:
{JSON_DATA}
Your summary:"""
]

# ================= 2. 提问模板 =================

# [Type A] Name -> Usage
NAME_QUERY_TEMPLATES = [
    "How do I use the `{name}` API?",
    "What are the arguments for `{name}`?",
    "Please provide the usage guide for `{name}`.",
    "Generate technical documentation for `{name}`.",
    "Explain the schema for `{name}`.",
    "I need help using `{name}`.",
    "What does the `{name}` function do?"
]

# [Type B] Description -> Usage (引用模式，不再强行造句)
INTENT_QUERY_TEMPLATES = [
    # 直接引用描述，询问名字和用法
    "Which API has the description: \"{description}\"?",
    "I am looking for the tool described as: \"{description}\". How do I use it?",
    "Find the API with the following functionality: \"{description}\".",
    "What is the name and usage of the API that matches this description: \"{description}\"?",
    "Given the description \"{description}\", which tool should I call?",
    "Identify the API defined by: \"{description}\" and explain its parameters.",
    # 模拟检索场景
    "Search for the tool with description: \"{description}\"."
]

# [Type C] Description -> Raw JSON (引用模式)
INTENT_TO_JSON_TEMPLATES = [
    "Show me the raw JSON definition for the tool described as: \"{description}\".",
    "Return the JSON schema for the API with the description: \"{description}\".",
    "What is the underlying definition for the tool having this description: \"{description}\"?"
]

# [Type D] Name -> Raw JSON
NAME_TO_JSON_TEMPLATES = [
    "Show me the raw JSON definition for `{name}`.",
    "Output the original definition of `{name}`.",
    "Get the JSON schema for `{name}`."
]

def main(args):
    print(f"Loading model from {args.model}...")
    llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size)
    sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    json_files = glob.glob(os.path.join(args.input, "*.json"))
    
    for file_path in json_files:
        file_name = os.path.basename(file_path)
        output_path = os.path.join(args.output, f"{file_name.replace('.json', '.parquet')}")
        print(f"Processing file: {file_name}...")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            prompts = []
            metadata = [] 

            # --- 阶段 1: 准备 vLLM 输入 ---
            for api_name, api_content in data.items():
                json_str = json.dumps(api_content, indent=2, ensure_ascii=False)
                
                # 清洗 description
                raw_desc = api_content.get('description', '').strip()
                if raw_desc.endswith('.'): raw_desc = raw_desc[:-1]
                clean_desc = raw_desc if raw_desc else f"use the {api_name} tool"

                for i, gen_template in enumerate(GENERATION_PROMPTS):
                    prompt_text = gen_template.replace("{JSON_DATA}", json_str).replace("{API_NAME}", api_name)
                    
                    full_prompt = llm.get_tokenizer().apply_chat_template(
                        [{"role": "user", "content": prompt_text}], 
                        tokenize=False, 
                        add_generation_prompt=True
                    )
                    
                    prompts.append(full_prompt)
                    metadata.append({
                        "api_name": api_name,
                        "description": clean_desc,
                        "template_id": i
                    })
            
            if not prompts: continue

            print(f"Generating {len(prompts)} responses...")
            outputs = llm.generate(prompts, sampling_params)

            # --- 阶段 2: 构造混合数据 ---
            data_rows = []
            
            for i, output in enumerate(outputs):
                meta = metadata[i]
                generated_answer = output.outputs[0].text.strip()
                
                # 1. 名字 -> 自然语言解释 (循环 N 次)
                for _ in range(args.num_name_nl_pairs):
                    template = random.choice(NAME_QUERY_TEMPLATES)
                    q_text = safe_format(template, name=meta['api_name'])
                    data_rows.append({"prompt": q_text, "response": generated_answer, "source": "name_query"})

                # 2. 意图 -> 自然语言解释 (循环 N 次) -> 【重点】
                for _ in range(args.num_intent_nl_pairs):
                    template = random.choice(INTENT_QUERY_TEMPLATES)
                    q_text = safe_format(template, description=meta['description'])
                    data_rows.append({"prompt": q_text, "response": generated_answer, "source": "intent_query"})

            # --- 阶段 3: 原始 JSON 数据的混合 ---
            for api_name, api_content in data.items():
                json_str = json.dumps(api_content, ensure_ascii=False)
                raw_desc = api_content.get('description', '').strip()
                if raw_desc.endswith('.'): raw_desc = raw_desc[:-1]
                clean_desc = raw_desc if raw_desc else f"use {api_name}"

                # 3. 名字 -> Raw JSON (循环 N 次)
                for _ in range(args.num_name_json_pairs):
                    template = random.choice(NAME_TO_JSON_TEMPLATES)
                    q_text = safe_format(template, name=api_name)
                    data_rows.append({"prompt": q_text, "response": json_str, "source": "raw_json_name"})

                # 4. 意图 -> Raw JSON (循环 N 次)
                for _ in range(args.num_intent_json_pairs):
                    template = random.choice(INTENT_TO_JSON_TEMPLATES)
                    q_text = safe_format(template, description=clean_desc)
                    data_rows.append({"prompt": q_text, "response": json_str, "source": "raw_json_intent"})

            # --- 阶段 4: 保存 ---
            df = pd.DataFrame(data_rows)
            # 随机打乱
            df = df.sample(frac=1).reset_index(drop=True)
            
            print(f"Generated {len(df)} diverse samples for {args.input}")
            df.to_parquet(output_path, index=False)
            print(f"Saved to {output_path}")

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

def safe_format(template, **kwargs):
    result = template
    for k, v in kwargs.items():
        result = result.replace("{" + k + "}", str(v))
    return result

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--input", type=str, default="data/api_infos",
                       help="Input JSON file path")
    parser.add_argument("--output", type=str, default="data/toolbench_stage1",
                       help="Output file path")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                       help="Model path")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                       help="Tensor parallel size")
    parser.add_argument("--max_tokens", type=int, default=2048,
                       help="Max tokens")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Temperature")
    parser.add_argument("--num_name_nl_pairs", type=int, default=1,
                       help="Num name NL pairs")
    parser.add_argument("--num_intent_nl_pairs", type=int, default=1,
                       help="Num intent NL pairs")
    parser.add_argument("--num_name_json_pairs", type=int, default=1,
                       help="Num name JSON pairs")
    parser.add_argument("--num_intent_json_pairs", type=int, default=1,
                       help="Num intent JSON pairs")
    args = parser.parse_args()
    
    main(args)