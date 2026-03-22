"""Ollama LLM provider adapter."""
import json
from typing import List, Dict, Any, Generator
import httpx

from app.services.providers.base import (
    BaseLLMProvider, ProviderType, ModelInfo, ModelCapabilities,
    GenerationSettings, Message, registry
)


class OllamaProvider(BaseLLMProvider):
    """Ollama local LLM provider."""
    
    provider_type = ProviderType.OLLAMA
    display_name = "Ollama (Local)"
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.client = httpx.Client(timeout=120.0)
    
    def list_models(self) -> List[ModelInfo]:
        """List available models in Ollama."""
        try:
            response = self.client.get(f"{self.base_url}/api/tags")
            if response.status_code != 200:
                return []
            
            data = response.json()
            models = []
            
            for model in data.get("models", []):
                model_id = model.get("name", "")
                if model_id:
                    caps = ModelCapabilities(
                        streaming=True,
                        tool_calling=False,
                        json_mode=True,
                        reasoning="deepseek-r1" in model_id.lower() or "r1" in model_id.lower(),
                    )
                    models.append(ModelInfo(
                        id=model_id,
                        name=model_id,
                        provider=self.provider_type.value,
                        capabilities=caps
                    ))
            
            if not models:
                models.append(ModelInfo(
                    id="deepseek-r1:7b",
                    name="DeepSeek R1 7B",
                    provider=self.provider_type.value,
                    capabilities=ModelCapabilities(
                        streaming=True,
                        json_mode=True,
                        reasoning=True
                    )
                ))
            
            return models
            
        except Exception:
            return [ModelInfo(
                id="deepseek-r1:7b",
                name="DeepSeek R1 7B",
                provider=self.provider_type.value,
                capabilities=ModelCapabilities(
                    streaming=True,
                    json_mode=True,
                    reasoning=True
                )
            )]
    
    def generate(
        self,
        messages: List[Message],
        model: str,
        settings: GenerationSettings
    ) -> Dict[str, Any]:
        """Generate a response using Ollama."""
        try:
            formatted_messages = []
            for msg in messages:
                formatted_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
            
            request_data = {
                "model": model,
                "messages": formatted_messages,
                "stream": False,
                "options": {
                    "temperature": settings.temperature,
                    "num_predict": settings.max_tokens
                }
            }
            
            if settings.json_mode:
                request_data["format"] = "json"
            
            response = self.client.post(
                f"{self.base_url}/api/chat",
                json=request_data
            )
            
            if response.status_code != 200:
                return {
                    "status": "error",
                    "message": f"Ollama returned status {response.status_code}"
                }
            
            result = response.json()
            content = result.get("message", {}).get("content", "")
            
            return {
                "status": "success",
                "content": content,
                "model": model,
                "provider": self.provider_type.value
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
    
    def stream(
        self,
        messages: List[Message],
        model: str,
        settings: GenerationSettings
    ) -> Generator[str, None, None]:
        """Generate a streaming response."""
        formatted_messages = []
        for msg in messages:
            formatted_messages.append({
                "role": msg.role,
                "content": msg.content
            })
        
        request_data = {
            "model": model,
            "messages": formatted_messages,
            "stream": True,
            "options": {
                "temperature": settings.temperature,
                "num_predict": settings.max_tokens
            }
        }
        
        if settings.json_mode:
            request_data["format"] = "json"
        
        try:
            with self.client.stream("POST", f"{self.base_url}/api/chat", json=request_data) as response:
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            content = data.get("message", {}).get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            yield f"Error: {str(e)}"
    
    def health_check(self) -> Dict[str, Any]:
        """Check Ollama health."""
        try:
            response = self.client.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                return {"status": "ok", "message": "Ollama is running"}
            return {"status": "error", "message": f"Status: {response.status_code}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}


def register_ollama():
    """Register Ollama provider."""
    ollama = OllamaProvider()
    registry.register(ProviderType.OLLAMA, ollama, set_default=True)
