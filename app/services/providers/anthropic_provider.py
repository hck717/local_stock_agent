"""Anthropic LLM provider adapter."""
import os
from typing import List, Dict, Any, Generator, Optional
import httpx

from app.services.providers.base import (
    BaseLLMProvider, ProviderType, ModelInfo, ModelCapabilities,
    GenerationSettings, Message, registry
)


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude LLM provider."""
    
    provider_type = ProviderType.ANTHROPIC
    display_name = "Anthropic (Claude)"
    
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.anthropic.com"):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.base_url = base_url
        self.client = httpx.Client(timeout=120.0)
    
    def list_models(self) -> List[ModelInfo]:
        """List available Anthropic models."""
        if not self.api_key:
            return self._get_default_models()
        
        return self._get_default_models()
    
    def _get_default_models(self) -> List[ModelInfo]:
        """Get default model list."""
        return [
            ModelInfo(
                id="claude-sonnet-4-20250514",
                name="Claude Sonnet 4",
                provider=self.provider_type.value,
                capabilities=ModelCapabilities(
                    streaming=True,
                    tool_calling=True,
                    json_mode=False,
                    vision=True,
                    reasoning=True,
                    max_tokens=8192
                )
            ),
            ModelInfo(
                id="claude-3-5-sonnet-20241022",
                name="Claude 3.5 Sonnet",
                provider=self.provider_type.value,
                capabilities=ModelCapabilities(
                    streaming=True,
                    tool_calling=True,
                    json_mode=False,
                    vision=True,
                    reasoning=True,
                    max_tokens=8192
                )
            ),
            ModelInfo(
                id="claude-3-5-haiku-20241022",
                name="Claude 3.5 Haiku",
                provider=self.provider_type.value,
                capabilities=ModelCapabilities(
                    streaming=True,
                    tool_calling=False,
                    json_mode=False,
                    vision=True,
                    reasoning=True,
                    max_tokens=4096
                )
            ),
            ModelInfo(
                id="claude-3-opus-20240229",
                name="Claude 3 Opus",
                provider=self.provider_type.value,
                capabilities=ModelCapabilities(
                    streaming=True,
                    tool_calling=True,
                    json_mode=False,
                    vision=True,
                    reasoning=True,
                    max_tokens=4096
                )
            ),
        ]
    
    def generate(
        self,
        messages: List[Message],
        model: str,
        settings: GenerationSettings
    ) -> Dict[str, Any]:
        """Generate a response using Anthropic."""
        if not self.api_key:
            return {"status": "error", "message": "Anthropic API key not configured"}
        
        try:
            formatted_messages = []
            for msg in messages:
                if msg.role == "system":
                    formatted_messages.append({
                        "role": "user",
                        "content": f"<system>\n{msg.content}\n</system>"
                    })
                else:
                    formatted_messages.append({
                        "role": msg.role,
                        "content": msg.content
                    })
            
            request_data = {
                "model": model,
                "messages": formatted_messages,
                "max_tokens": settings.max_tokens,
                "temperature": settings.temperature,
            }
            
            response = self.client.post(
                f"{self.base_url}/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01",
                    "anthropic-dangerous-direct-browser-access": "true"
                },
                json=request_data
            )
            
            if response.status_code != 200:
                return {
                    "status": "error",
                    "message": f"Anthropic returned status {response.status_code}: {response.text}"
                }
            
            result = response.json()
            content = result.get("content", [{}])[0].get("text", "")
            
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
            yield "Error: Anthropic API key not configured"
            return
        
        formatted_messages = []
        for msg in messages:
            if msg.role == "system":
                formatted_messages.append({
                    "role": "user",
                    "content": f"<system>\n{msg.content}\n</system>"
                })
            else:
                formatted_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        request_data = {
            "model": model,
            "messages": formatted_messages,
            "max_tokens": settings.max_tokens,
            "temperature": settings.temperature,
            "stream": True
        }
        
        try:
            with self.client.stream(
                "POST",
                f"{self.base_url}/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01"
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
                            content = data.get("content", [{}])[0].get("text", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            yield f"Error: {str(e)}"
    
    def health_check(self) -> Dict[str, Any]:
        """Check Anthropic API health."""
        if not self.api_key:
            return {"status": "warning", "message": "API key not configured"}
        
        return {"status": "ok", "message": "Anthropic API is configured"}


def register_anthropic():
    """Register Anthropic provider."""
    anthropic = AnthropicProvider()
    registry.register(ProviderType.ANTHROPIC, anthropic)
