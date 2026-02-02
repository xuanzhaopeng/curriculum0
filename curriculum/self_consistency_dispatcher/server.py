from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import asyncio
import httpx
import logging
import json
from collections import Counter

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Self-Consistency Dispatcher", description="Dispatcher for calculating self-consistency of Math Agent solutions")

import os

MATH_AGENT_URL = os.getenv("MATH_AGENT_URL", "tcp://localhost:8000")

class DispatchRequest(BaseModel):
    question: str
    n: int = 10
    max_turns: int = 5
    model: str = "qwen-flash-us"

class DispatchResponse(BaseModel):
    question: str
    majority_answer: Optional[str]
    self_consistency_score: float
    total_samples: int
    all_answers: List[Optional[str]]
    raw_responses: List[Dict[str, Any]]
    tool_calls: List[int] = []
    

import struct

async def call_math_agent(question: str, max_turns: int, model: str) -> Dict[str, Any]:
    payload = {
        "problem": question,
        "max_turns": max_turns,
        "model": model
    }
    
    # Parse MATH_AGENT_URL to get host/port
    # Expected format: tcp://host:port or just host:port
    # For backward compatibility, if it starts with http, we might need logic?
    # User instruction says MIGRATE to TCP. So we assume it's a TCP address.
    # Default: localhost:8000
    
    agent_addr = MATH_AGENT_URL.replace("tcp://", "").replace("http://", "")
    
    # Strip any path suffix (e.g. /solve) if present
    if "/" in agent_addr:
        agent_addr = agent_addr.split("/")[0]

    if ":" in agent_addr:
        host, port_str = agent_addr.split(":")
        port = int(port_str)
    else:
        host = agent_addr
        port = 8000
    
    try:
        async def _tcp_interaction():
            reader = None
            writer = None
            try:
                reader, writer = await asyncio.open_connection(host, port)
                
                # Serialize
                req_bytes = json.dumps(payload).encode('utf-8')
                req_len = len(req_bytes)
                
                # Send 4 bytes len + body
                writer.write(struct.pack('>I', req_len))
                writer.write(req_bytes)
                await writer.drain()
                
                # Read Response
                # 1. Read 4 bytes len
                header_data = await reader.readexactly(4)
                resp_len = struct.unpack('>I', header_data)[0]
                
                # 2. Read body
                body_data = await reader.readexactly(resp_len)
                resp_json = json.loads(body_data.decode('utf-8'))
                return resp_json
            finally:
                if writer:
                    try:
                        writer.close()
                        await writer.wait_closed()
                    except Exception:
                        pass # Ignore errors during close

        # Set strict timeout of 1600 seconds for the entire interaction (300s x 5 turns + overhead)
        return await asyncio.wait_for(_tcp_interaction(), timeout=1600.0)
        
        logger.error(f"Timeout (1600s) calling Math Agent via TCP at {host}:{port}")
        return {"error": "timeout", "final_answer": None, "raw_reasoning": "Timeout waiting for Math Agent"}
    except Exception as e:
        logger.error(f"Error calling Math Agent via TCP: {e}")
        return {"error": str(e), "final_answer": None, "raw_reasoning": ""}

@app.post("/dispatch", response_model=DispatchResponse)
async def dispatch(request: DispatchRequest):
    logger.info(f"Dispatching question: {request.question} (n={request.n})")
        
    # TCP client creates a new connection per request usually, or we could pool.
    # For now, simple open_connection per call is fine for this scale.
    # We no longer need httpx client context
    
    tasks = [call_math_agent(request.question, request.max_turns, request.model) for _ in range(request.n)]
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
            raw_responses=results,
            tool_calls=[r.get("tool_calls", 0) for r in results]
        )
    
    # Use math equivalence to group answers instead of simple string comparison
    from mathruler.grader import grade_answer
    
    unique_groups = [] # List of (representative_answer, count)
    
    for ans in valid_answers:
        found = False
        for i in range(len(unique_groups)):
            group_representative, count = unique_groups[i]
            is_negative_answer = ('no ' in ans.lower() or 'not ' in ans.lower()) and ('no ' in group_representative.lower() or 'not ' in group_representative.lower())
            if ans == group_representative or is_negative_answer or grade_answer(ans, group_representative) or grade_answer(group_representative, ans):
                unique_groups[i] = (group_representative, count + 1)
                found = True
                break
        if not found:
            unique_groups.append((ans, 1))

    if not unique_groups:
        return DispatchResponse(
            question=request.question,
            majority_answer=None,
            self_consistency_score=0.0,
            total_samples=request.n,
            all_answers=answers,
            raw_responses=results,
            tool_calls=[r.get("tool_calls", 0) for r in results]
        )

    # Sort groups by count descending
    unique_groups.sort(key=lambda x: x[1], reverse=True)
    majority_answer, majority_count = unique_groups[0]
    
    # Self-consistency score based on Agent0: p(x) = count(majority) / n
    score = majority_count / request.n
    
    return DispatchResponse(
        question=request.question,
        majority_answer=majority_answer,
        self_consistency_score=score,
        total_samples=request.n,
        all_answers=answers,
        raw_responses=results,
        tool_calls=[r.get("tool_calls", 0) for r in results]
    )

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
