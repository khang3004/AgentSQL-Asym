"""Module for instantiating LLM providers via a Factory pattern."""

import os
import httpx
import logging
from typing import Protocol

from google import genai
from google.genai.errors import APIError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class LLMInterface(Protocol):
    def generate(self, prompt: str) -> str:
        """Generates a text response given a prompt string."""
        ...

class RateLimitError(Exception):
    pass

class KeyRotator:
    """Provides keys in a Round-Robin fashion."""
    def __init__(self, prefix: str):
        self.keys = []
        i = 1
        while True:
            key = os.environ.get(f"{prefix}_{i}")
            if not key:
                if i == 1:
                    fallback = os.environ.get(prefix)
                    if fallback and not any(x in fallback.lower() for x in ["your_", "placeholder", "_here"]):
                        self.keys.append(fallback)
                break
            if not any(x in key.lower() for x in ["your_", "placeholder", "_here"]):
                self.keys.append(key)
            i += 1
            
        if not self.keys and os.environ.get(prefix):
             fallback_raw = os.environ.get(prefix)
             if fallback_raw and not any(x in fallback_raw.lower() for x in ["your_", "placeholder", "_here"]):
                  self.keys.append(fallback_raw)
             
        self.current_idx = 0

    def get_key(self) -> str:
        if not self.keys:
            return ""
        key = self.keys[self.current_idx]
        self.current_idx = (self.current_idx + 1) % len(self.keys)
        return key

groq_rotator = KeyRotator("GROQ_API_KEY")
gemini_rotator = KeyRotator("GEMINI_API_KEY")

class OllamaLLM:
    """Fallback to local Ollama (The M3 Pro / PC Savior)."""
    def __init__(self, model_name: str = "llama3.1"):
        self.model_name = model_name
        self.api_url = "http://host.docker.internal:11434/api/generate"

    def generate(self, prompt: str) -> str:
        logger.info("[OllamaLLM] Using local fallback for prompt.")
        try:
            with httpx.Client(timeout=120.0) as client:
                response = client.post(
                    self.api_url, 
                    json={"model": self.model_name, "prompt": prompt, "stream": False}
                )
                response.raise_for_status()
                return response.json().get("response", "")
        except Exception as e:
            logger.error(f"[OllamaLLM] Fallback failed: {e}")
            return f"Error: Local fallback failed - {e}"

def fallback_to_ollama(func):
    """Decorator to fallback to Ollama if all keys/retries fail."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"All retries failed with error: {e}. Falling back to local Ollama.")
            prompt = kwargs.get('prompt') or args[1]
            return OllamaLLM().generate(prompt)
    return wrapper

class GroqLLM:
    """Wrapper for Groq API with Key Rotation and Rate Limit handling."""
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"

    @fallback_to_ollama
    @retry(
        retry=retry_if_exception_type(RateLimitError),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def generate(self, prompt: str) -> str:
        api_key = groq_rotator.get_key()
        if not api_key:
            raise Exception("No GROQ_API_KEY found in .env")
            
        headers = {
            "Authorization": f"Bearer {api_key}",
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
            if response.status_code == 429:
                logger.warning("Groq HTTP 429 hit. Rotating key and retrying.")
                raise RateLimitError("Groq 429 Too Many Requests")
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                logger.error(f"Groq API error: {response.text}")
                raise e
            return response.json()["choices"][0]["message"]["content"]

class GeminiLLM:
    """Wrapper for Google Gemini API with Key Rotation and Rate Limit handling."""
    def __init__(self, model_name: str):
        self.model_name = model_name

    @fallback_to_ollama
    @retry(
        retry=retry_if_exception_type(RateLimitError),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def generate(self, prompt: str) -> str:
        api_key = gemini_rotator.get_key()
        if not api_key:
            raise Exception("No GEMINI_API_KEY found in .env")
            
        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            return response.text or ""
        except APIError as e:
            if getattr(e, 'code', None) == 429 or "429" in str(e):
                logger.warning("Gemini HTTP 429 hit. Rotating key and retrying.")
                raise RateLimitError("Gemini 429 Too Many Requests")
            raise

def get_llm(role: str, provider: str = None, model_name: str = None) -> LLMInterface:
    """
    Dependency Injection factory for Language Models.
    Enforces Asymmetric Configuration:
    - Generator: Llama-4 via Groq
    - Corrector/Critic: Gemini 2.5 Flash via Google
    """
    role_lower = role.lower()
    
    # Use provided arguments, fallback to defaults if not provided
    if not provider:
        provider = "groq" if role_lower == "generator" else "google"
    if not model_name:
        model_name = "llama3-70b-8192" if role_lower == "generator" else "gemini-2.5-flash"

    logger.info("[LLMFactory] Initializing %s LLM: %s via %s", role, model_name, provider)
    
    if provider == "groq":
        return GroqLLM(model_name)
    elif provider == "google":
        return GeminiLLM(model_name)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
