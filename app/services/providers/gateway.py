"""LLM Provider Gateway - unified interface for all providers."""
from typing import Dict, Any, Optional, List
import json

from app.services.providers.base import (
    ProviderType, ModelInfo, GenerationSettings, Message,
    registry, BaseLLMProvider
)


class LLMGateway:
    """Unified gateway for all LLM providers."""
    
    def __init__(self):
        self._current_provider: Optional[ProviderType] = None
        self._current_model: Optional[str] = None
        self._settings: GenerationSettings = GenerationSettings()
    
    @property
    def current_provider(self) -> Optional[ProviderType]:
        return self._current_provider
    
    @property
    def current_model(self) -> Optional[str]:
        return self._current_model
    
    @property
    def settings(self) -> GenerationSettings:
        return self._settings
    
    def initialize_providers(self):
        """Initialize all registered providers."""
        from app.services.providers.ollama_provider import register_ollama
        from app.services.providers.openai_provider import register_openai
        from app.services.providers.anthropic_provider import register_anthropic
        from app.services.providers.gemini_provider import register_gemini
        from app.services.providers.deepseek_provider import register_deepseek
        
        register_ollama()
        register_openai()
        register_anthropic()
        register_gemini()
        register_deepseek()
    
    def list_providers(self) -> List[Dict[str, Any]]:
        """List all available providers."""
        return registry.list_providers()
    
    def set_provider(self, provider_type: str, model: Optional[str] = None) -> bool:
        """Set the current provider and model."""
        try:
            ptype = ProviderType(provider_type)
            provider = registry.get(ptype)
            
            if not provider:
                return False
            
            self._current_provider = ptype
            
            if model:
                self._current_model = model
            else:
                models = provider.list_models()
                self._current_model = models[0].id if models else None
            
            return True
            
        except ValueError:
            return False
    
    def set_default_provider(self) -> bool:
        """Set the default provider from config."""
        provider, model = registry.get_default()
        if provider:
            self._current_provider = registry._default_provider
            self._current_model = model
            return True
        return False
    
    def generate(
        self,
        messages: List[Message],
        provider_type: Optional[str] = None,
        model: Optional[str] = None,
        settings: Optional[GenerationSettings] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate a response using the specified or current provider."""
        target_provider = self._current_provider
        target_model = self._current_model
        
        if provider_type:
            try:
                target_provider = ProviderType(provider_type)
            except ValueError:
                return {"status": "error", "message": f"Unknown provider: {provider_type}"}
        
        if model:
            target_model = model
        
        if not target_provider or not target_model:
            return {"status": "error", "message": "No provider or model selected"}
        
        provider = registry.get(target_provider)
        if not provider:
            return {"status": "error", "message": f"Provider {target_provider} not available"}
        
        if api_key:
            provider.api_key = api_key
        
        if base_url:
            provider.base_url = base_url
        
        gen_settings = settings or self._settings
        
        return provider.generate(messages, target_model, gen_settings)
    
    def stream(
        self,
        messages: List[Message],
        provider_type: Optional[str] = None,
        model: Optional[str] = None,
        settings: Optional[GenerationSettings] = None
    ):
        """Generate a streaming response."""
        target_provider = self._current_provider
        target_model = self._current_model
        
        if provider_type:
            try:
                target_provider = ProviderType(provider_type)
            except ValueError:
                yield "Error: Unknown provider"
                return
        
        if model:
            target_model = model
        
        if not target_provider or not target_model:
            yield "Error: No provider or model selected"
            return
        
        provider = registry.get(target_provider)
        if not provider:
            yield f"Error: Provider {target_provider} not available"
            return
        
        gen_settings = settings or self._settings
        gen_settings.stream = True
        
        for chunk in provider.stream(messages, target_model, gen_settings):
            yield chunk
    
    def health_check_all(self) -> Dict[str, Any]:
        """Check health of all providers."""
        results = {}
        for ptype in ProviderType:
            provider = registry.get(ptype)
            if provider:
                results[ptype.value] = provider.health_check()
        return results
    
    def get_capabilities(self, provider_type: str, model: str) -> Optional[Dict[str, Any]]:
        """Get capabilities for a specific model."""
        try:
            ptype = ProviderType(provider_type)
            provider = registry.get(ptype)
            if provider:
                info = provider.get_model_info(model)
                if info:
                    caps = info.capabilities
                    return {
                        "streaming": caps.streaming,
                        "tool_calling": caps.tool_calling,
                        "json_mode": caps.json_mode,
                        "vision": caps.vision,
                        "reasoning": caps.reasoning
                    }
        except ValueError:
            pass
        return None


gateway = LLMGateway()
