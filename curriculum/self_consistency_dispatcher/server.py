from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import asyncio
import httpx
import logging
from collections import Counter

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Self-Consistency Dispatcher", description="Dispatcher for calculating self-consistency of Math Agent solutions")

import os

MATH_AGENT_URL = os.getenv("MATH_AGENT_URL", "http://localhost:8000/solve")

class DispatchRequest(BaseModel):
    question: str
    n: int = 10
    max_turns: int = 5
    model: str = "gemini-2.0-flash"

class DispatchResponse(BaseModel):
    question: str
    majority_answer: Optional[str]
    self_consistency_score: float
    total_samples: int
    all_answers: List[Optional[str]]
    raw_responses: List[Dict[str, Any]]

async def call_math_agent(client: httpx.AsyncClient, question: str, max_turns: int, model: str) -> Dict[str, Any]:
    payload = {
        "problem": question,
        "max_turns": max_turns,
        "model": model
    }
    try:
        response = await client.post(MATH_AGENT_URL, json=payload, timeout=180.0)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error calling Math Agent: {e}")
        return {"error": str(e), "final_answer": None, "raw_reasoning": ""}

@app.post("/dispatch", response_model=DispatchResponse)
async def dispatch(request: DispatchRequest):
    logger.info(f"Dispatching question: {request.question} (n={request.n})")
    
    async with httpx.AsyncClient() as client:
        tasks = [call_math_agent(client, request.question, request.max_turns, request.model) for _ in range(request.n)]
        results = await asyncio.gather(*tasks)
    
    answers = [r.get("final_answer") for r in results]
    
    # Filter out None and errors if necessary, but keep for score calculation $n$
    valid_answers = [a for a in answers if a is not None]
    
    if not valid_answers:
        return DispatchResponse(
            question=request.question,
            majority_answer=None,
            self_consistency_score=0.0,
            total_samples=request.n,
            all_answers=answers,
            raw_responses=results
        )
    
    # Standard majority voting
    counts = Counter(valid_answers)
    majority_answer, majority_count = counts.most_common(1)[0]
    
    # Self-consistency score based on Agent0: p(x) = count(majority) / n
    score = majority_count / request.n
    
    return DispatchResponse(
        question=request.question,
        majority_answer=majority_answer,
        self_consistency_score=score,
        total_samples=request.n,
        all_answers=answers,
        raw_responses=results
    )

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
