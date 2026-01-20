from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import uvicorn
import os
import logging
from .agent import MathAgent

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Math Agent API", description="HTTP interface for the Math Agent using OpenAI Protocol")

class ProblemRequest(BaseModel):
    problem: str
    max_turns: Optional[int] = 5
    sandbox_url: Optional[str] = "http://localhost:8080"
    model: Optional[str] = "gemini-2.5-flash"
    base_url: Optional[str] = "https://generativelanguage.googleapis.com/v1beta/openai/"
    api_key: Optional[str] = None # Can be passed in request or use env

class SolutionResponse(BaseModel):
    problem: str
    raw_reasoning: str
    final_answer: Optional[str]

@app.post("/solve", response_model=SolutionResponse)
async def solve_problem(request: ProblemRequest, x_api_key: Optional[str] = Header(None)):
    try:
        # Resolve API Key
        api_key = request.api_key or x_api_key or os.getenv("LLM_API_KEY") or os.getenv("GEMINI_API_KEY")
        
        if not api_key:
            raise HTTPException(status_code=401, detail="API Key is required.")

        # Initialize agent
        agent = MathAgent(
            api_key=api_key,
            base_url=request.base_url,
            model=request.model,
            sandbox_url=request.sandbox_url
        )
        
        logger.info(f"Solving problem: {request.problem}")
        result_dict = agent.solve(problem=request.problem, max_turns=request.max_turns)
        
        return SolutionResponse(
            problem=request.problem, 
            raw_reasoning=result_dict["raw_reasoning"],
            final_answer=result_dict["final_answer"]
        )
    except Exception as e:
        logger.error(f"Error solving problem: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    # Note: When running with uvicorn directly, use --env-file or set env vars
    uvicorn.run(app, host="0.0.0.0", port=port)
