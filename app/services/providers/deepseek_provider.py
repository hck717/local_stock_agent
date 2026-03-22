"""DeepSeek LLM provider adapter."""
import os
from typing import List, Dict, Any, Generator, Optional
import httpx

from app.services.providers.base import (
    BaseLLMProvider, ProviderType, ModelInfo, ModelCapabilities,
    GenerationSettings, Message, registry
)


class DeepSeekProvider(BaseLLMProvider):
    """DeepSeek LLM provider."""
    
    provider_type = ProviderType.DEEPSEEK
    display_name = "DeepSeek"
    
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.deepseek.com"):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.base_url = base_url
        self.client = httpx.Client(timeout=120.0)
    
    def list_models(self) -> List[ModelInfo]:
        """List available DeepSeek models."""
        if not self.api_key:
            return self._get_default_models()
        
        return self._get_default_models()
    
    def _get_default_models(self) -> List[ModelInfo]:
        """Get default model list."""
        return [
            ModelInfo(
                id="deepseek-chat",
                name="DeepSeek Chat",
                provider=self.provider_type.value,
                capabilities=ModelCapabilities(
                    streaming=True,
                    tool_calling=True,
                    json_mode=True,
                    vision=False,
                    reasoning=True,
                    max_tokens=8192
                )
            ),
            ModelInfo(
                id="deepseek-reasoner",
                name="DeepSeek Reasoner",
                provider=self.provider_type.value,
                capabilities=ModelCapabilities(
                    streaming=True,
                    tool_calling=False,
                    json_mode=True,
                    vision=False,
                    reasoning=True,
                    max_tokens=8192
                )
            ),
        ]
    
    def generate(
        self,
        messages: List[Message],
        model: str,
        settings: GenerationSettings
    ) -> Dict[str, Any]:
        """Generate a response using DeepSeek."""
        if not self.api_key:
            return {"status": "error", "message": "DeepSeek API key not configured"}
        
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
                "temperature": settings.temperature,
                "max_tokens": settings.max_tokens,
                "stream": False
            }
            
            if settings.json_mode:
                request_data["response_format"] = {"type": "json_object"}
            
            response = self.client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json=request_data
            )
            
            if response.status_code != 200:
                return {
                    "status": "error",
                    "message": f"DeepSeek returned status {response.status_code}: {response.text}"
                }
            
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            return {
                "status": "success",
                "content": content,
                "model": model,
                "provider": self.provider_type.value
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def stream(
        self,
        messages: List[Message],
        model: str,
        settings: GenerationSettings
    ) -> Generator[str, None, None]:
        """Generate a streaming response."""
        if not self.api_key:
            yield "Error: DeepSeek API key not configured"
            return
        
        formatted_messages = []
        for msg in messages:
            formatted_messages.append({
                "role": msg.role,
                "content": msg.content
            })
        
        request_data = {
            "model": model,
            "messages": formatted_messages,
            "temperature": settings.temperature,
            "max_tokens": settings.max_tokens,
            "stream": True
        }
        
        try:
            with self.client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json=request_data
            ) as response:
                for line in response.iter_lines():
                    if line and line.startswith("data: "):
                        if line.strip() == "data: [DONE]":
                            break
                        import json
                        try:
                            data = json.loads(line[6:])
                            content = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            yield f"Error: {str(e)}"
    
    def health_check(self) -> Dict[str, Any]:
        """Check DeepSeek API health."""
        if not self.api_key:
            return {"status": "warning", "message": "API key not configured"}
        
        return {"status": "ok", "message": "DeepSeek API is configured"}


def register_deepseek():
    """Register DeepSeek provider."""
    deepseek = DeepSeekProvider()
    registry.register(ProviderType.DEEPSEEK, deepseek)
