"""Base LLM Provider interface and models."""
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Generator
from dataclasses import dataclass, field
from enum import Enum


class ProviderType(Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    DEEPSEEK = "deepseek"
    PERPLEXITY = "perplexity"


@dataclass
class ModelCapabilities:
    streaming: bool = True
    tool_calling: bool = False
    json_mode: bool = False
    vision: bool = False
    reasoning: bool = False
    max_tokens: int = 4096


@dataclass
class ModelInfo:
    id: str
    name: str
    provider: str
    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)


@dataclass
class GenerationSettings:
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    reasoning_mode: bool = False
    stream: bool = True
    json_mode: bool = False


@dataclass
class Message:
    role: str
    content: str
    name: Optional[str] = None


class BaseLLMProvider(ABC):
    """Base class for all LLM providers."""
    
    provider_type: ProviderType = ProviderType.OLLAMA
    display_name: str = "Unknown"
    
    @abstractmethod
    def list_models(self) -> List[ModelInfo]:
        """List available models for this provider."""
        pass
    
    @abstractmethod
    def generate(
        self,
        messages: List[Message],
        model: str,
        settings: GenerationSettings
    ) -> Dict[str, Any]:
        """Generate a response from the model."""
        pass
    
    @abstractmethod
    def stream(
        self,
        messages: List[Message],
        model: str,
        settings: GenerationSettings
    ) -> Generator[str, None, None]:
        """Generate a streaming response from the model."""
        pass
    
    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """Check if the provider is available and healthy."""
        pass
    
    def get_model_info(self, model: str) -> Optional[ModelInfo]:
        """Get information about a specific model."""
        models = self.list_models()
        for m in models:
            if m.id == model:
                return m
        return None
    
    def supports_tools(self, model: str) -> bool:
        """Check if model supports tool calling."""
        info = self.get_model_info(model)
        return info.capabilities.tool_calling if info else False
    
    def supports_json_mode(self, model: str) -> bool:
        """Check if model supports JSON mode."""
        info = self.get_model_info(model)
        return info.capabilities.json_mode if info else False
    
    def supports_streaming(self, model: str) -> bool:
        """Check if model supports streaming."""
        info = self.get_model_info(model)
        return info.capabilities.streaming if info else True


class ProviderRegistry:
    """Registry for all LLM providers."""
    
    def __init__(self):
        self._providers: Dict[ProviderType, BaseLLMProvider] = {}
        self._default_provider: Optional[ProviderType] = None
        self._default_model: Optional[str] = None
    
    def register(
        self,
        provider: ProviderType,
        instance: BaseLLMProvider,
        set_default: bool = False
    ):
        """Register a provider."""
        self._providers[provider] = instance
        if set_default or not self._default_provider:
            self._default_provider = provider
            models = instance.list_models()
            if models:
                self._default_model = models[0].id
    
    def get(self, provider: ProviderType) -> Optional[BaseLLMProvider]:
        """Get a provider instance."""
        return self._providers.get(provider)
    
    def list_providers(self) -> List[Dict[str, Any]]:
        """List all registered providers."""
        result = []
        for ptype, provider in self._providers.items():
            models = provider.list_models()
            result.append({
                "type": ptype.value,
                "name": provider.display_name,
                "default_model": self._default_model if ptype == self._default_provider else (models[0].id if models else None),
                "models": [m.id for m in models],
                "available": provider.health_check().get("status") == "ok"
            })
        return result
    
    def get_default(self) -> tuple[Optional[BaseLLMProvider], Optional[str]]:
        """Get the default provider and model."""
        if self._default_provider:
            return self._providers.get(self._default_provider), self._default_model
        return None, None
    
    def set_default(self, provider: ProviderType, model: Optional[str] = None):
        """Set the default provider and model."""
        if provider in self._providers:
            self._default_provider = provider
            if model:
                self._default_model = model
            else:
                models = self._providers[provider].list_models()
                self._default_model = models[0].id if models else None


registry = ProviderRegistry()
