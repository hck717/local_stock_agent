"""Chat API routes."""
from fastapi import APIRouter, HTTPException
from typing import Optional

from app.agent.schemas import ChatRequest, ChatResponse
from app.agent.orchestrator import stock_agent

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process a stock analysis query.
    
    Accepts natural language stock questions and returns:
    - Natural language answer
    - Data tables
    - Charts (as HTML file paths)
    - Warnings and disclaimers
    
    Optional parameters:
    - provider: LLM provider type (ollama, openai)
    - model: Specific model name
    """
    try:
        result = stock_agent.process_request(
            message=request.message,
            session_id=request.session_id,
            provider=request.provider,
            model=request.model,
            provider_config=request.provider_config,
            require_approval=request.require_approval,
            approved=request.approved
        )
        
        if result.get("status") == "error":
            error_code = result.get("error_code", "ERROR")
            if error_code in ["NO_DATA", "RATE_LIMIT_OR_SOURCE_ERROR"]:
                return result
            raise HTTPException(status_code=400, detail=result)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
