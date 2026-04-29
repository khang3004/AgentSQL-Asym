"""Module for instantiating LLM providers via a Factory pattern.

This module provides a unified interface for different Language Models (LLMs)
used in the AgentSQL pipeline. It implements an Asymmetric framework where
the generator and critic roles have distinct error handling and fallback strategies.
"""

import os
import httpx
import logging
from typing import Protocol, Any, Callable

from google import genai
from google.genai.errors import APIError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class LLMInterface(Protocol):
    """Protocol defining the standard interface for all LLM wrappers."""
    
    def generate(self, prompt: str) -> str:
        """Generates a text response given a prompt string.
        
        Args:
            prompt (str): The input prompt for the LLM.
            
        Returns:
            str: The generated text response.
        """
        ...

class RateLimitError(Exception):
    """Exception raised when an LLM provider's rate limit is exceeded."""
    pass

class ServiceUnavailableError(Exception):
    """Exception raised when an LLM provider service is temporarily unavailable (e.g., 503)."""
    pass

class KeyRotator:
    """Provides API keys in a Round-Robin fashion from environment variables."""
    
    def __init__(self, prefix: str) -> None:
        """Initializes the KeyRotator by scanning environment variables.
        
        Args:
            prefix (str): The prefix of the environment variables to scan (e.g., 'GROQ_API_KEY').
        """
        self.keys: list[str] = []
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
             
        self.current_idx: int = 0

    def get_key(self) -> str:
        """Retrieves the next available API key.
        
        Returns:
            str: The API key, or an empty string if no valid keys are found.
        """
        if not self.keys:
            return ""
        key = self.keys[self.current_idx]
        self.current_idx = (self.current_idx + 1) % len(self.keys)
        return key

groq_rotator = KeyRotator("GROQ_API_KEY")
gemini_rotator = KeyRotator("GEMINI_API_KEY")

class OllamaLLM:
    """Fallback LLM using a local Ollama instance."""
    
    def __init__(self, model_name: str = "llama3.1") -> None:
        """Initializes the Ollama local LLM.
        
        Args:
            model_name (str): The name of the local Ollama model to use.
        """
        self.model_name = model_name
        self.api_url = "http://host.docker.internal:11434/api/generate"

    def generate(self, prompt: str) -> str:
        """Generates a text response using the local Ollama instance.
        
        Args:
            prompt (str): The input prompt.
            
        Returns:
            str: The generated text response, or an error string if the fallback fails.
        """
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

def fallback_to_ollama(func: Callable) -> Callable:
    """Decorator to fallback to Ollama ONLY if all keys are exhausted.
    
    Args:
        func (Callable): The generation function to wrap.
        
    Returns:
        Callable: The wrapped function with fallback logic.
    """
    def wrapper(*args: Any, **kwargs: Any) -> str:
        try:
            return func(*args, **kwargs)
        except RateLimitError as e:
            logger.error(f"All API keys exhausted (Rate Limit): {e}. Falling back to local Ollama.")
            prompt = kwargs.get('prompt') or args[1]
            return OllamaLLM().generate(prompt)
    return wrapper

class GroqLLM:
    """Wrapper for Groq API acting as the Generator.
    
    Implements key rotation and rate limit handling. Falls back to a local Ollama
    instance only when all API keys have exhausted their quotas (RateLimitError).
    """
    
    def __init__(self, model_name: str) -> None:
        """Initializes the Groq LLM wrapper.
        
        Args:
            model_name (str): The Groq model name to use.
        """
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
        """Generates SQL using the Groq API.
        
        Args:
            prompt (str): The prompt containing the schema and user question.
            
        Raises:
            Exception: If no API key is found.
            RateLimitError: If a 429 Too Many Requests error occurs.
            httpx.HTTPStatusError: For other HTTP errors (e.g., 400, 404).
            
        Returns:
            str: The generated SQL response.
        """
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
    """Wrapper for Google Gemini API acting as the Critic.
    
    Implements robust error handling with exponential backoff for 429 (Quota)
    and 503 (Service Unavailable) errors. Does NOT fallback to a local model.
    """
    
    def __init__(self, model_name: str) -> None:
        """Initializes the Gemini LLM wrapper.
        
        Args:
            model_name (str): The Gemini model name to use.
        """
        self.model_name = model_name

    @retry(
        retry=retry_if_exception_type((RateLimitError, ServiceUnavailableError)),
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        reraise=True
    )
    def generate(self, prompt: str) -> str:
        """Generates correction feedback using the Gemini API.
        
        Args:
            prompt (str): The prompt containing the failed SQL and schema.
            
        Raises:
            Exception: If no API key is found.
            RateLimitError: If a 429 Too Many Requests error occurs.
            ServiceUnavailableError: If a 503 Service Unavailable error occurs.
            
        Returns:
            str: The generated feedback response.
        """
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
            error_code = getattr(e, 'code', None)
            error_str = str(e)
            
            if error_code == 429 or "429" in error_str:
                logger.warning("Gemini HTTP 429 hit. Rotating key and retrying.")
                raise RateLimitError("Gemini 429 Too Many Requests")
            elif error_code == 503 or "503" in error_str:
                logger.warning("Gemini HTTP 503 Service Unavailable hit. Backing off and retrying.")
                raise ServiceUnavailableError("Gemini 503 Service Unavailable")
                
            logger.error(f"Gemini API Error: {error_str}")
            raise e

class ResilientCriticLLM:
    """Resilient wrapper for Critic role that falls back to Groq 70B on persistent Google failures.
    
    Implements cross-provider graceful degradation.
    """
    
    def __init__(self, primary_model_name: str) -> None:
        """Initializes the resilient critic wrapper.
        
        Args:
            primary_model_name (str): The primary Gemini model name to use.
        """
        self.primary_llm = GeminiLLM(primary_model_name)
        self.fallback_llm = GroqLLM("llama-3.3-70b-versatile")

    def generate(self, prompt: str) -> str:
        """Generates correction feedback, falling back to Groq if Google completely fails.
        
        Args:
            prompt (str): The input prompt.
            
        Returns:
            str: The generated feedback response.
        """
        try:
            return self.primary_llm.generate(prompt)
        except Exception as e:
            logger.warning(
                f"CRITICAL: Primary Critic (Google) failed persistently with error: {e}. "
                f"Gracefully degrading to Groq (llama-3.3-70b-versatile)."
            )
            return self.fallback_llm.generate(prompt)


def get_llm(role: str, provider: str = None, model_name: str = None) -> LLMInterface:
    """Dependency Injection factory for Language Models.
    
    Args:
        role (str): The role of the LLM in the pipeline ('generator' or 'critic').
        provider (str, optional): The LLM provider. Defaults to 'groq' for generator, 'google' for critic.
        model_name (str, optional): The model name. Defaults to provider-specific models.
        
    Raises:
        ValueError: If an unsupported provider is specified.
        
    Returns:
        LLMInterface: An instantiated LLM wrapper matching the requested role and provider.
    """
    role_lower = role.lower()
    
    # Use provided arguments, fallback to defaults if not provided
    if not provider:
        provider = "groq" if role_lower == "generator" else "google"
    if not model_name:
        model_name = "llama-3.1-8b-instant" if role_lower == "generator" else "gemini-2.5-flash"

    logger.info("[LLMFactory] Initializing %s LLM: %s via %s", role, model_name, provider)
    
    if provider == "groq":
        return GroqLLM(model_name)
    elif provider == "google":
        if role_lower == "critic":
            return ResilientCriticLLM(model_name)
        return GeminiLLM(model_name)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
