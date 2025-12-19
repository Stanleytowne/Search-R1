import asyncio
import json
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import AsyncOpenAI
import uvicorn

JUDGE_PROMPT = """Giving the query and the corresponding execution trajectory (including thoughts, tool calls, and observations), evaluate the `answer_status` based on these rules:

1. **Solved**: The tool calls were successful, and the final answer completely and accurately addresses all parts of the user's query.
2. **Partially Solved**: The model made progress and successfully called some tools, but the final answer is incomplete, only addresses part of the query, or contains minor inaccuracies.
3. **Unsolved**: The tool calls failed (errors), the model gave up, or the final answer is clearly incorrect, hallucinated, or irrelevant to the tool observations.

Output a JSON object with the following fields:
- "reason": A very brief explanation (less than 20 words).
- "answer_status": One of ["Solved", "Partially Solved", "Unsolved"].

Query:
{query}

Answer Trajectory:
{trajectory}
"""

app = FastAPI()

client = AsyncOpenAI(
    api_key="YOUR_OPENAI_API_KEY",
    base_url=None
)

class JudgeRequest(BaseModel):
    queries: List[str]
    trajectories: List[str]

class JudgeResponse(BaseModel):
    scores: List[float]

async def get_single_score(query: str, trajectory: str):
    prompt = JUDGE_PROMPT.format(query=query, trajectory=trajectory)
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            timeout=20.0
        )
        res_json = json.loads(response.choices[0].message.content)
        status = res_json.get("answer_status", "Unsolved")
        
        status_map = {
            "Solved": 1.0,
            "Partially Solved": 0.5,
            "Unsolved": 0.0
        }
        return status_map.get(status, 0.0) # 默认为 0 比较保险
    except Exception as e:
        print(f"Error calling LLM Judge: {e}")
        return 0.0

@app.post("/evaluate_batch", response_model=JudgeResponse)
async def evaluate_batch(request: JudgeRequest):
    if len(request.queries) != len(request.trajectories):
        raise HTTPException(status_code=400, detail="Length mismatch")
    
    # 并发处理 Batch
    tasks = [get_single_score(q, t) for q, t in zip(request.queries, request.trajectories)]
    scores = await asyncio.gather(*tasks)
    return JudgeResponse(scores=scores)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=1234)