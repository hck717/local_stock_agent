"""Google Gemini LLM provider adapter."""
import os
from typing import List, Dict, Any, Generator, Optional
import httpx

from app.services.providers.base import (
    BaseLLMProvider, ProviderType, ModelInfo, ModelCapabilities,
    GenerationSettings, Message, registry
)


class GeminiProvider(BaseLLMProvider):
    """Google Gemini LLM provider."""
    
    provider_type = ProviderType.GEMINI
    display_name = "Google Gemini"
    
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://generativelanguage.googleapis.com/v1beta"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.base_url = base_url
        self.client = httpx.Client(timeout=120.0)
    
    def list_models(self) -> List[ModelInfo]:
        """List available Gemini models."""
        if not self.api_key:
            return self._get_default_models()
        
        try:
            response = self.client.get(
                f"{self.base_url}/models?key={self.api_key}"
            )
            
            if response.status_code != 200:
                return self._get_default_models()
            
            data = response.json()
            models = []
            
            for model in data.get("models", []):
                model_id = model.get("name", "").replace("models/", "")
                caps = ModelCapabilities(
                    streaming=True,
                    tool_calling="functionCalling" in model.get("supportedGenerationMethods", []),
                    json_mode=True,
                    vision=True,
                    reasoning="thinking" in model_id.lower(),
                    max_tokens=32768
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
        """Get default model list."""
        return [
            ModelInfo(
                id="gemini-2.0-flash-exp",
                name="Gemini 2.0 Flash (Experimental)",
                provider=self.provider_type.value,
                capabilities=ModelCapabilities(
                    streaming=True,
                    tool_calling=True,
                    json_mode=True,
                    vision=True,
                    reasoning=True,
                    max_tokens=32768
                )
            ),
            ModelInfo(
                id="gemini-1.5-flash",
                name="Gemini 1.5 Flash",
                provider=self.provider_type.value,
                capabilities=ModelCapabilities(
                    streaming=True,
                    tool_calling=True,
                    json_mode=True,
                    vision=True,
                    reasoning=True,
                    max_tokens=8192
                )
            ),
            ModelInfo(
                id="gemini-1.5-flash-8b",
                name="Gemini 1.5 Flash 8B",
                provider=self.provider_type.value,
                capabilities=ModelCapabilities(
                    streaming=True,
                    tool_calling=True,
                    json_mode=True,
                    vision=True,
                    reasoning=True,
                    max_tokens=8192
                )
            ),
            ModelInfo(
                id="gemini-1.5-pro",
                name="Gemini 1.5 Pro",
                provider=self.provider_type.value,
                capabilities=ModelCapabilities(
                    streaming=True,
                    tool_calling=True,
                    json_mode=True,
                    vision=True,
                    reasoning=True,
                    max_tokens=32768
                )
            ),
        ]
    
    def generate(
        self,
        messages: List[Message],
        model: str,
        settings: GenerationSettings
    ) -> Dict[str, Any]:
        """Generate a response using Gemini."""
        if not self.api_key:
            return {"status": "error", "message": "Gemini API key not configured"}
        
        try:
            formatted_contents = []
            for msg in messages:
                if msg.role == "system":
                    formatted_contents.append({
                        "role": "user",
                        "parts": [{"text": f"<system>\n{msg.content}\n</system>"}]
                    })
                else:
                    role = "user" if msg.role == "user" else "model"
                    formatted_contents.append({
                        "role": role,
                        "parts": [{"text": msg.content}]
                    })
            
            request_data = {
                "contents": formatted_contents,
                "generationConfig": {
                    "temperature": settings.temperature,
                    "maxOutputTokens": settings.max_tokens,
                }
            }
            
            response = self.client.post(
                f"{self.base_url}/models/{model}:generateContent?key={self.api_key}",
                headers={"Content-Type": "application/json"},
                json=request_data
            )
            
            if response.status_code != 200:
                return {
                    "status": "error",
                    "message": f"Gemini returned status {response.status_code}: {response.text}"
                }
            
            result = response.json()
            content = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            
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
            yield "Error: Gemini API key not configured"
            return
        
        formatted_contents = []
        for msg in messages:
            if msg.role == "system":
                formatted_contents.append({
                    "role": "user",
                    "parts": [{"text": f"<system>\n{msg.content}\n</system>"}]
                })
            else:
                role = "user" if msg.role == "user" else "model"
                formatted_contents.append({
                    "role": role,
                    "parts": [{"text": msg.content}]
                })
        
        request_data = {
            "contents": formatted_contents,
            "generationConfig": {
                "temperature": settings.temperature,
                "maxOutputTokens": settings.max_tokens,
            }
        }
        
        try:
            with self.client.stream(
                "POST",
                f"{self.base_url}/models/{model}:streamGenerateContent?key={self.api_key}&alt=sse",
                headers={"Content-Type": "application/json"},
                json=request_data
            ) as response:
                for line in response.iter_lines():
                    if line:
                        import json
                        try:
                            data = json.loads(line)
                            content = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            yield f"Error: {str(e)}"
    
    def health_check(self) -> Dict[str, Any]:
        """Check Gemini API health."""
        if not self.api_key:
            return {"status": "warning", "message": "API key not configured"}
        
        try:
            response = self.client.get(
                f"{self.base_url}/models?key={self.api_key}"
            )
            if response.status_code == 200:
                return {"status": "ok", "message": "Gemini API is accessible"}
            return {"status": "error", "message": f"Status: {response.status_code}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}


def register_gemini():
    """Register Gemini provider."""
    gemini = GeminiProvider()
    registry.register(ProviderType.GEMINI, gemini)
