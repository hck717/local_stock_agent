"""Ollama LLM client service."""
import json
import httpx
from typing import Optional, Dict, Any, List

from app.config import config


class OllamaClient:
    """Client for interacting with Ollama API."""
    
    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None):
        self.base_url = base_url or config.OLLAMA_BASE_URL
        self.model = model or config.OLLAMA_MODEL
        self.client = httpx.Client(timeout=120.0)
    
    def _make_request(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make a request to Ollama API."""
        url = f"{self.base_url}/api/{endpoint}"
        response = self.client.post(url, json=data)
        response.raise_for_status()
        return response.json()
    
    def check_health(self) -> Dict[str, Any]:
        """Check if Ollama is running and model is available."""
        try:
            tags_response = self.client.get(f"{self.base_url}/api/tags")
            if tags_response.status_code != 200:
                return {"status": "error", "message": "Ollama not responding"}
            
            models = tags_response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            
            if any(self.model in name for name in model_names):
                return {"status": "ok", "model": self.model}
            else:
                return {
                    "status": "warning",
                    "message": f"Model {self.model} not found. Available: {model_names}"
                }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        json_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Generate a response from the model.
        
        Args:
            prompt: User prompt
            system: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            json_mode: Whether to request JSON output
        
        Returns:
            dict with response and metadata
        """
        request_data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        if system:
            request_data["system"] = system
        
        if json_mode:
            request_data["format"] = "json"
        
        try:
            result = self._make_request("generate", request_data)
            return {
                "status": "success",
                "response": result.get("response", ""),
                "done": result.get("done", True),
                "model": self.model
            }
        except httpx.HTTPError as e:
            return {"status": "error", "message": str(e)}
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> Dict[str, Any]:
        """
        Generate a chat response.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        
        Returns:
            dict with response and metadata
        """
        request_data = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        try:
            result = self._make_request("chat", request_data)
            return {
                "status": "success",
                "response": result.get("message", {}).get("content", ""),
                "done": result.get("done", True),
                "model": self.model
            }
        except httpx.HTTPError as e:
            return {"status": "error", "message": str(e)}
    
    def generate_with_context(
        self,
        prompt: str,
        context: Optional[List[int]] = None,
        system: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate with continuation context.
        
        Args:
            prompt: User prompt
            context: Previous context for continuation
            system: Optional system prompt
        
        Returns:
            dict with response and updated context
        """
        request_data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        
        if context:
            request_data["context"] = context
        
        if system:
            request_data["system"] = system
        
        try:
            result = self._make_request("generate", request_data)
            return {
                "status": "success",
                "response": result.get("response", ""),
                "context": result.get("context"),
                "done": result.get("done", True)
            }
        except httpx.HTTPError as e:
            return {"status": "error", "message": str(e)}
    
    def close(self):
        """Close the client connection."""
        self.client.close()


ollama_client = OllamaClient()
