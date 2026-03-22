"""Provider management routes."""
from fastapi import APIRouter, HTTPException
from typing import Optional, List, Dict, Any

from app.services.providers import gateway

router = APIRouter(prefix="/providers", tags=["Providers"])


@router.get("/")
async def list_providers() -> List[Dict[str, Any]]:
    """
    List all available LLM providers and their models.
    """
    return gateway.list_providers()


@router.get("/{provider_type}/models")
async def list_models(provider_type: str) -> List[Dict[str, Any]]:
    """
    List all available models for a provider.
    
    Args:
        provider_type: Provider type (ollama, openai)
    """
    providers = gateway.list_providers()
    for p in providers:
        if p.get("type") == provider_type:
            return p.get("models", [])
    
    raise HTTPException(status_code=404, detail=f"Provider {provider_type} not found")


@router.get("/{provider_type}/health")
async def provider_health(provider_type: str) -> Dict[str, Any]:
    """
    Check health status of a specific provider.
    
    Args:
        provider_type: Provider type (ollama, openai)
    """
    result = gateway.health_check_all()
    if provider_type in result:
        return {"provider": provider_type, "status": result[provider_type]}
    
    raise HTTPException(status_code=404, detail=f"Provider {provider_type} not found or not initialized")


@router.post("/set")
async def set_provider(provider: str, model: Optional[str] = None) -> Dict[str, Any]:
    """
    Set the active LLM provider and model.
    
    Args:
        provider: Provider type (ollama, openai)
        model: Optional model name (uses default if not specified)
    """
    success = gateway.set_provider(provider, model)
    if success:
        return {
            "status": "ok",
            "provider": provider,
            "model": gateway.current_model,
            "message": f"Set provider to {provider} with model {gateway.current_model}"
        }
    
    return {
        "status": "error",
        "message": f"Failed to set provider {provider}",
        "available": [p.get("type") for p in gateway.list_providers()]
    }


@router.get("/{provider_type}/models/{model}/capabilities")
async def get_model_capabilities(provider_type: str, model: str) -> Dict[str, Any]:
    """
    Get capabilities for a specific model.
    
    Args:
        provider_type: Provider type
        model: Model name
    """
    caps = gateway.get_capabilities(provider_type, model)
    if caps:
        return {
            "provider": provider_type,
            "model": model,
            "capabilities": caps
        }
    
    raise HTTPException(status_code=404, detail=f"Model {model} not found for provider {provider_type}")
