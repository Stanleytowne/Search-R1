import asyncio
import json
import logging
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import AsyncOpenAI
import uvicorn
from tenacity import retry, stop_after_attempt, wait_exponential

JUDGE_PROMPT = """Giving the query and the corresponding execution trajectory (including thoughts, tool calls, and observations), evaluate the `answer_status` based on these rules:

1. **Solved**: The tool calls were successful. The final answer is strictly grounded in the real "Observation" data and fully addresses the query.
2. **Partially Solved**: The model used real "Observation" data, but the task is only halfway finished or the final answer missed some details from the observations.
3. **Unsolved**: 
    - The model fabricated information not found in the Observations.
    - The tool calls failed, and the model failed to solve the query or made up a result.
    - The answer is incorrect or irrelevant.

Output a JSON object with the following fields:
- "reason": A very brief explanation (less than 20 words).
- "answer_status": One of ["Solved", "Partially Solved", "Unsolved"].

<Target_Query>
{query}
</Target_Query>

<Model_Execution_Trajectory>
{trajectory}
</Model_Execution_Trajectory>
"""

# 设置日志，方便排查哪个环节慢
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# 限制最大并发数，建议根据你的 OpenAI Tier 等级调整（如 10-20）
MAX_CONCURRENT_REQUESTS = 20
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

client = AsyncOpenAI(
    api_key="YOUR_OPENAI_API_KEY",
    base_url=None
)

class JudgeRequest(BaseModel):
    queries: List[str]
    trajectories: List[str]

class JudgeResponse(BaseModel):
    scores: List[float]

# 使用 tenacity 增加自动重试逻辑
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def call_openai_with_retry(prompt: str):
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        timeout=30.0 # 稍微增加一点超时
    )
    return response

async def get_single_score(query: str, trajectory: str):
    # 使用信号量控制并发
    async with semaphore:
        prompt = JUDGE_PROMPT.format(query=query, trajectory=trajectory)
        try:
            response = await call_openai_with_retry(prompt)
            res_json = json.loads(response.choices[0].message.content)
            status = res_json.get("answer_status", "Unsolved")
            
            status_map = {"Solved": 1.0, "Partially Solved": 0.5, "Unsolved": 0.0}
            return status_map.get(status, 0.0)
        except Exception as e:
            logger.error(f"Error calling LLM Judge after retries: {e}")
            return 0.0

@app.post("/evaluate_batch", response_model=JudgeResponse)
async def evaluate_batch(request: JudgeRequest):
    if len(request.queries) != len(request.trajectories):
        raise HTTPException(status_code=400, detail="Length mismatch")
    
    # 这里的 gather 会受到 semaphore 的保护，不会瞬间压垮网络
    tasks = [get_single_score(q, t) for q, t in zip(request.queries, request.trajectories)]
    scores = await asyncio.gather(*tasks)
    return JudgeResponse(scores=scores)

if __name__ == "__main__":
    # 使用 0.0.0.0 确保容器/宿主机间通信正常
    uvicorn.run(app, host="0.0.0.0", port=12346)