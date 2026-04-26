"""Module for instantiating LLM providers via a Factory pattern."""

import os
import httpx
import logging
from typing import Protocol

from google import genai

logger = logging.getLogger(__name__)

class LLMInterface(Protocol):
    def generate(self, prompt: str) -> str:
        """Generates a text response given a prompt string."""
        ...

class GroqLLM:
    """Wrapper for Groq API."""
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.api_key = os.environ.get("GROQ_API_KEY")
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is missing.")

    def generate(self, prompt: str) -> str:
        """Invokes Groq HTTP endpoint."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 1024
        }
        with httpx.Client(timeout=30.0) as client:
            response = client.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

class GeminiLLM:
    """Wrapper for Google Gemini API."""
    def __init__(self, model_name: str):
        self.model_name = model_name
        if not os.environ.get("GEMINI_API_KEY"):
            raise ValueError("GEMINI_API_KEY is missing.")
        self.client = genai.Client()

    def generate(self, prompt: str) -> str:
        """Invokes Gemini SDK."""
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        return response.text or ""

def get_llm(role: str, provider: str, model_name: str) -> LLMInterface:
    """
    Dependency Injection factory for Language Models.
    Allows dynamic switching between providers for the Asymmetric architecture.
    
    Args:
        role (str): The intended role of the LLM (e.g., 'generator', 'critic').
        provider (str): The model provider ('groq', 'google').
        model_name (str): The specific model version (e.g., 'gemini-2.5-flash').
        
    Returns:
        LLMInterface: An object exposing a generate(prompt) method.
        
    Raises:
        ValueError: If an unsupported provider is requested.
    """
    logger.info("[LLMFactory] Initializing %s LLM: %s via %s", role, model_name, provider)
    
    provider_lower = provider.lower()
    if provider_lower == "groq":
        return GroqLLM(model_name)
    elif provider_lower == "google":
        return GeminiLLM(model_name)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
