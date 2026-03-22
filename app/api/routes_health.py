"""Health check routes."""
from fastapi import APIRouter

from app.agent.schemas import HealthResponse
from app.services.ollama_client import ollama_client
from app.services.providers import gateway
from app.tools.cache_tool import get_cache_stats

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("/", response_model=HealthResponse)
async def health_check():
    """
    Check health of all system components.
    """
    ollama_status = ollama_client.check_health()
    
    return HealthResponse(
        api="ok",
        ollama=ollama_status.get("status", "unknown"),
        yfinance="ok"
    )


@router.get("/providers")
async def providers_health():
    """
    Check health of all LLM providers.
    """
    return gateway.health_check_all()


@router.get("/cache")
async def cache_status():
    """
    Get cache statistics.
    """
    return get_cache_stats()


@router.get("/ollama")
async def ollama_health():
    """
    Detailed Ollama health check.
    """
    return ollama_client.check_health()
