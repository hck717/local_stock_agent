"""OpenAI LLM provider adapter."""
import os
from typing import List, Dict, Any, Generator, Optional
import httpx

from app.services.providers.base import (
    BaseLLMProvider, ProviderType, ModelInfo, ModelCapabilities,
    GenerationSettings, Message, registry
)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider."""
    
    provider_type = ProviderType.OPENAI
    display_name = "OpenAI"
    
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.openai.com/v1"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url
        self.client = httpx.Client(timeout=120.0)
    
    def list_models(self) -> List[ModelInfo]:
        """List available OpenAI models."""
        if not self.api_key:
            return self._get_default_models()
        
        try:
            response = self.client.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
            
            if response.status_code != 200:
                return self._get_default_models()
            
            data = response.json()
            models = []
            
            gpt_models = ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "o1", "o1-mini", "o1-preview"]
            
            for model in data.get("data", []):
                model_id = model.get("id", "")
                if any(g in model_id.lower() for g in gpt_models):
                    caps = ModelCapabilities(
                        streaming=True,
                        tool_calling=True,
                        json_mode=True,
                        vision="vision" in model_id.lower(),
                        reasoning="o1" in model_id.lower(),
                    )
                    models.append(ModelInfo(
                        id=model_id,
                        name=model_id,
                        provider=self.provider_type.value,
                        capabilities=caps
                    ))
            
            if not models:
                return self._get_default_models()
            
            return models
            
        except Exception:
            return self._get_default_models()
    
    def _get_default_models(self) -> List[ModelInfo]:
        """Get default model list when API is unavailable."""
        return [
            ModelInfo(
                id="gpt-4o",
                name="GPT-4o",
                provider=self.provider_type.value,
                capabilities=ModelCapabilities(
                    streaming=True,
                    tool_calling=True,
                    json_mode=True,
                    vision=True,
                )
            ),
            ModelInfo(
                id="gpt-4o-mini",
                name="GPT-4o Mini",
                provider=self.provider_type.value,
                capabilities=ModelCapabilities(
                    streaming=True,
                    tool_calling=True,
                    json_mode=True,
                    vision=True,
                )
            ),
            ModelInfo(
                id="o1-preview",
                name="o1 Preview",
                provider=self.provider_type.value,
                capabilities=ModelCapabilities(
                    streaming=False,
                    tool_calling=False,
                    json_mode=False,
                    reasoning=True,
                )
            ),
        ]
    
    def generate(
        self,
        messages: List[Message],
        model: str,
        settings: GenerationSettings
    ) -> Dict[str, Any]:
        """Generate a response using OpenAI."""
        if not self.api_key:
            return {"status": "error", "message": "OpenAI API key not configured"}
        
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
                    "message": f"OpenAI returned status {response.status_code}: {response.text}"
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
            yield "Error: OpenAI API key not configured"
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
        """Check OpenAI API health."""
        if not self.api_key:
            return {"status": "warning", "message": "API key not configured"}
        
        try:
            response = self.client.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
            if response.status_code == 200:
                return {"status": "ok", "message": "OpenAI API is accessible"}
            return {"status": "error", "message": f"Status: {response.status_code}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}


def register_openai():
    """Register OpenAI provider."""
    openai = OpenAIProvider()
    registry.register(ProviderType.OPENAI, openai)
