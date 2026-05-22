"""Module for instantiating LLM providers via a Factory pattern.

This module provides a unified interface for Language Models used in the AgentSQL
pipeline. It implements an Asymmetric, Groq-only architecture where:
  - Generator role: openai/gpt-oss-120b (primary) → llama-4-scout-17b (fallback)
  - Corrector role: openai/gpt-oss-20b (primary) → llama-4-scout-17b (fallback)

Key-rotation strategy: On a Groq 429, cycle to the next API key IMMEDIATELY
(zero sleep) and retry. Only after all N keys have been tried does the caller
escalate to the scout fallback model.
"""

import os
import logging
from typing import Protocol

import httpx
from dotenv import load_dotenv
from langsmith import traceable


logger = logging.getLogger(__name__)

load_dotenv()

# ---------------------------------------------------------------------------
# Custom exception types
# ---------------------------------------------------------------------------

class RateLimitError(Exception):
    """Raised when a Groq API key returns HTTP 429 Too Many Requests."""
    pass


class AllKeysExhaustedError(Exception):
    """Raised when every Groq API key in the pool has hit its rate limit."""
    pass


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

class LLMInterface(Protocol):
    """Standard interface for all LLM wrappers in the pipeline."""

    def generate(self, prompt: str) -> str:
        """Generates a text response given a prompt string.

        Args:
            prompt: The input prompt for the LLM.

        Returns:
            The generated text response.
        """
        ...


# ---------------------------------------------------------------------------
# Key Manager (Groq-only)
# ---------------------------------------------------------------------------

class KeyManager:
    """Round-Robin API key manager that reads keys from environment variables.

    Scans for ``GROQ_API_KEY_1``, ``GROQ_API_KEY_2``, … (and a bare
    ``GROQ_API_KEY`` as a single-key fallback) and cycles through them.
    ``rotate()`` advances to the next key; ``exhausted`` becomes True once
    every key has been tried in the current round.
    """

    def __init__(self, prefix: str = "GROQ_API_KEY") -> None:
        self.keys: list[str] = []
        i = 1
        while True:
            key = os.environ.get(f"{prefix}_{i}")
            if not key:
                break
            if not any(x in key.lower() for x in ["your_", "placeholder", "_here"]):
                self.keys.append(key)
            i += 1

        # Bare fallback (single-key setup)
        if not self.keys:
            bare = os.environ.get(prefix, "")
            if bare and not any(x in bare.lower() for x in ["your_", "placeholder", "_here"]):
                self.keys.append(bare)

        self._idx: int = 0
        self._tries: int = 0  # how many distinct keys used in current round

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def total_keys(self) -> int:
        return len(self.keys)

    @property
    def exhausted(self) -> bool:
        """True once every key has been tried at least once in this round."""
        return self._tries >= self.total_keys

    def current_key(self) -> str:
        """Return the currently active API key (does NOT advance the cursor)."""
        if not self.keys:
            return ""
        return self.keys[self._idx]

    def rotate(self) -> str:
        """Advance to the next key and return it.  Marks one more key as tried."""
        if not self.keys:
            return ""
        self._idx = (self._idx + 1) % self.total_keys
        self._tries += 1
        logger.info("[KeyManager] Rotated to key index %d.", self._idx)
        return self.keys[self._idx]

    def reset_tries(self) -> None:
        """Call after a successful round so ``exhausted`` resets correctly."""
        self._tries = 0


# Singleton key managers – shared across all LLM instances
groq_key_manager = KeyManager("GROQ_API_KEY")
gemini_key_manager = KeyManager("GEMINI_API_KEY")


# ---------------------------------------------------------------------------
# Base Groq & Gemini HTTP callers
# ---------------------------------------------------------------------------

_GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
_GROQ_TIMEOUT = 60.0


def _call_groq(model_name: str, prompt: str, api_key: str, max_tokens: int = 2048, temperature: float = 0.0) -> str:
    """Low-level Groq REST call.  Raises ``RateLimitError`` on 429.

    Args:
        model_name: Groq model identifier.
        prompt: User prompt string.
        api_key: Groq API key to use.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0.0 = deterministic).

    Raises:
        RateLimitError: On HTTP 429.
        httpx.HTTPStatusError: On any other HTTP error.

    Returns:
        The generated text content.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    with httpx.Client(timeout=_GROQ_TIMEOUT) as client:
        response = client.post(_GROQ_API_URL, headers=headers, json=payload)

    if response.status_code == 429:
        logger.warning("[Groq] HTTP 429 on model=%s key_idx=%d. Rotating key.", model_name, groq_key_manager._idx)
        raise RateLimitError(f"Groq 429 on model {model_name}")

    try:
        response.raise_for_status()
    except httpx.HTTPStatusError:
        logger.error("[Groq] HTTP error %d: %s", response.status_code, response.text[:400])
        raise

    return response.json()["choices"][0]["message"]["content"]


def _call_gemini(model_name: str, prompt: str, api_key: str, max_tokens: int = 2048, temperature: float = 0.0) -> str:
    """Low-level native Google Gemini REST call. Raises ``RateLimitError`` on 429.

    Args:
        model_name: Gemini model identifier.
        prompt: User prompt string.
        api_key: Gemini API key to use.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0.0 = deterministic).

    Raises:
        RateLimitError: On HTTP 429.
        httpx.HTTPStatusError: On any other HTTP error.

    Returns:
        The generated text content.
    """
    clean_model = model_name.split("/")[-1]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{clean_model}:generateContent?key={api_key}"
    headers = {
        "Content-Type": "application/json",
    }
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens
        }
    }
    with httpx.Client(timeout=60.0) as client:
        response = client.post(url, headers=headers, json=payload)

    if response.status_code == 429:
        logger.warning("[Gemini] HTTP 429 on model=%s key_idx=%d. Rotating key.", model_name, gemini_key_manager._idx)
        raise RateLimitError(f"Gemini 429 on model {model_name}")

    try:
        response.raise_for_status()
    except httpx.HTTPStatusError:
        logger.error("[Gemini] HTTP error %d: %s", response.status_code, response.text[:400])
        raise

    try:
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as e:
        logger.error("[Gemini] Unexpected response format: %s", response.text[:400])
        raise RuntimeError("Failed to parse Gemini API response") from e


# ---------------------------------------------------------------------------
# LLM Providers (Groq & Gemini)
# ---------------------------------------------------------------------------

class GroqLLM:
    """Wrapper for a single Groq model with instant key rotation on 429.

    Retries up to ``len(keys)`` times with *zero* sleep between each attempt.
    Each retry calls ``groq_key_manager.rotate()`` before the next attempt so
    a fresh key is always used.  If all keys are exhausted the
    ``AllKeysExhaustedError`` propagates upward.
    """

    def __init__(self, model_name: str, max_tokens: int = 2048) -> None:
        self.model_name = model_name
        self.max_tokens = max_tokens

    @traceable(run_type="llm")
    def generate(self, prompt: str) -> str:
        """Generate text.  Rotates key immediately on 429, raises after all exhausted.

        Args:
            prompt: Input prompt.

        Raises:
            AllKeysExhaustedError: When every key has returned 429.
            Exception: On non-rate-limit errors.

        Returns:
            Generated text string.
        """
        if not groq_key_manager.keys:
            raise RuntimeError("No GROQ_API_KEY_* entries found in environment.")

        groq_key_manager.reset_tries()
        last_error: Exception | None = None

        # Try every key in the pool exactly once per call
        for attempt in range(groq_key_manager.total_keys):
            api_key = groq_key_manager.current_key()
            try:
                result = _call_groq(self.model_name, prompt, api_key, self.max_tokens)
                groq_key_manager.reset_tries()
                return result
            except RateLimitError as e:
                last_error = e
                groq_key_manager.rotate()
                logger.info("[GroqLLM] Attempt %d/%d exhausted – trying next key.", attempt + 1, groq_key_manager.total_keys)
            except Exception:
                raise  # Non-rate-limit errors bubble up immediately

        raise AllKeysExhaustedError(
            f"All {groq_key_manager.total_keys} Groq keys returned 429 for model '{self.model_name}'."
        ) from last_error


class GeminiLLM:
    """Wrapper for a single Gemini model with instant key rotation on 429.

    Retries up to ``len(keys)`` times with *zero* sleep between each attempt.
    Each retry calls ``gemini_key_manager.rotate()`` before the next attempt so
    a fresh key is always used.  If all keys are exhausted the
    ``AllKeysExhaustedError`` propagates upward.
    """

    def __init__(self, model_name: str, max_tokens: int = 2048) -> None:
        self.model_name = model_name
        self.max_tokens = max_tokens

    @traceable(run_type="llm")
    def generate(self, prompt: str) -> str:
        """Generate text.  Rotates key immediately on 429, raises after all exhausted.

        Args:
            prompt: Input prompt.

        Raises:
            AllKeysExhaustedError: When every key has returned 429.
            Exception: On non-rate-limit errors.

        Returns:
            Generated text string.
        """
        if not gemini_key_manager.keys:
            raise RuntimeError("No GEMINI_API_KEY_* entries found in environment.")

        gemini_key_manager.reset_tries()
        last_error: Exception | None = None

        # Try every key in the pool exactly once per call
        for attempt in range(gemini_key_manager.total_keys):
            api_key = gemini_key_manager.current_key()
            try:
                result = _call_gemini(self.model_name, prompt, api_key, self.max_tokens)
                gemini_key_manager.reset_tries()
                return result
            except RateLimitError as e:
                last_error = e
                gemini_key_manager.rotate()
                logger.info("[GeminiLLM] Attempt %d/%d exhausted – trying next key.", attempt + 1, gemini_key_manager.total_keys)
            except Exception:
                raise  # Non-rate-limit errors bubble up immediately

        raise AllKeysExhaustedError(
            f"All {gemini_key_manager.total_keys} Gemini keys returned 429 for model '{self.model_name}'."
        ) from last_error


# ---------------------------------------------------------------------------
# Resilient Wrappers (Primary with Fallback)
# ---------------------------------------------------------------------------

def get_fallback_models() -> list[str]:
    """Helper to dynamically resolve the fallback model list.

    1. First priority: os.getenv("FALLBACK_MODEL") (comma-separated list)
    2. Second priority: default to ["google/gemma-4-31b", "meta-llama/llama-4-scout-17b-16e-instruct"]
    """
    env_fallback = os.getenv("FALLBACK_MODEL")
    if env_fallback:
        models = [m.strip() for m in env_fallback.split(",") if m.strip()]
        if models:
            return models
    return ["google/gemma-4-31b", "meta-llama/llama-4-scout-17b-16e-instruct"]


def get_fallback_providers() -> list[str]:
    """Helper to dynamically resolve the fallback provider list.

    1. First priority: os.getenv("FALLBACK_PROVIDER") (comma-separated list)
    2. Second priority: empty list (will auto-infer from model names)
    """
    env_provider = os.getenv("FALLBACK_PROVIDER")
    if env_provider:
        providers = [p.strip().lower() for p in env_provider.split(",") if p.strip()]
        if providers:
            return providers
    return []


def get_fallback_model() -> str:
    """Helper to dynamically resolve the fallback model name (returns the first one)."""
    models = get_fallback_models()
    return models[0] if models else "google/gemma-4-31b"


def create_base_llm(model_name: str, max_tokens: int = 2048, provider: str | None = None) -> LLMInterface:
    """Helper to instantiate the correct base LLM (Groq or Gemini) based on model name or explicit provider."""
    resolved_provider = provider
    if not resolved_provider:
        model_lower = model_name.lower()
        if "gemini" in model_lower or "google" in model_lower or "gemma" in model_lower:
            resolved_provider = "google"
        else:
            resolved_provider = "groq"
            
    resolved_provider = resolved_provider.lower()
    if resolved_provider in ("google", "gemini"):
        return GeminiLLM(model_name, max_tokens)
    else:
        return GroqLLM(model_name, max_tokens)


class ResilientGroqLLM:
    """Two-tier Groq wrapper: primary model -> dynamic fallback list on full key exhaustion.

    When all Groq keys fail on the primary model (``AllKeysExhaustedError``),
    falls back sequentially through the configured fallback models.
    """

    def __init__(self, primary_model: str, max_tokens: int = 2048) -> None:
        self.primary_llm = GroqLLM(primary_model, max_tokens)
        self.primary_model = primary_model
        self.max_tokens = max_tokens

    @traceable(run_type="llm")
    def generate(self, prompt: str) -> str:
        """Generate text, transparently degrading to fallback on key exhaustion.

        Args:
            prompt: Input prompt.

        Returns:
            Generated text string.
        """
        try:
            return self.primary_llm.generate(prompt)
        except AllKeysExhaustedError as e:
            fallback_models = get_fallback_models()
            fallback_providers = get_fallback_providers()
            logger.warning(
                "[ResilientGroqLLM] All keys exhausted for primary model '%s': %s. "
                "Starting fallback chain: %s with providers %s",
                self.primary_model,
                e,
                fallback_models,
                fallback_providers,
            )
            
            last_err = e
            for i, model_name in enumerate(fallback_models):
                if model_name == self.primary_model:
                    continue
                # Determine provider for this fallback model
                provider = None
                if i < len(fallback_providers):
                    provider = fallback_providers[i]
                try:
                    logger.info("[ResilientGroqLLM] Trying fallback model '%s' with provider '%s'...", model_name, provider or "auto-infer")
                    fallback_llm = create_base_llm(model_name, self.max_tokens, provider)
                    return fallback_llm.generate(prompt)
                except AllKeysExhaustedError as fe:
                    logger.warning(
                        "[ResilientGroqLLM] Fallback model '%s' also failed with key exhaustion: %s",
                        model_name,
                        fe,
                    )
                    last_err = fe
                except Exception as ex:
                    logger.error(
                        "[ResilientGroqLLM] Fallback model '%s' failed with unexpected error: %s",
                        model_name,
                        ex,
                    )
                    raise ex
            raise AllKeysExhaustedError(
                f"All primary and fallback models exhausted for primary model '{self.primary_model}'."
            ) from last_err


class ResilientGeminiLLM:
    """Two-tier Gemini wrapper: primary model -> dynamic fallback list on full key exhaustion.

    When all Gemini keys fail on the primary model (``AllKeysExhaustedError``),
    falls back sequentially through the configured fallback models.
    """

    def __init__(self, primary_model: str, max_tokens: int = 2048) -> None:
        self.primary_llm = GeminiLLM(primary_model, max_tokens)
        self.primary_model = primary_model
        self.max_tokens = max_tokens

    @traceable(run_type="llm")
    def generate(self, prompt: str) -> str:
        """Generate text, transparently degrading to fallback on key exhaustion.

        Args:
            prompt: Input prompt.

        Returns:
            Generated text string.
        """
        try:
            return self.primary_llm.generate(prompt)
        except AllKeysExhaustedError as e:
            fallback_models = get_fallback_models()
            fallback_providers = get_fallback_providers()
            logger.warning(
                "[ResilientGeminiLLM] All keys exhausted for primary model '%s': %s. "
                "Starting fallback chain: %s with providers %s",
                self.primary_model,
                e,
                fallback_models,
                fallback_providers,
            )
            
            last_err = e
            for i, model_name in enumerate(fallback_models):
                if model_name == self.primary_model:
                    continue
                # Determine provider for this fallback model
                provider = None
                if i < len(fallback_providers):
                    provider = fallback_providers[i]
                try:
                    logger.info("[ResilientGeminiLLM] Trying fallback model '%s' with provider '%s'...", model_name, provider or "auto-infer")
                    fallback_llm = create_base_llm(model_name, self.max_tokens, provider)
                    return fallback_llm.generate(prompt)
                except AllKeysExhaustedError as fe:
                    logger.warning(
                        "[ResilientGeminiLLM] Fallback model '%s' also failed with key exhaustion: %s",
                        model_name,
                        fe,
                    )
                    last_err = fe
                except Exception as ex:
                    logger.error(
                        "[ResilientGeminiLLM] Fallback model '%s' failed with unexpected error: %s",
                        model_name,
                        ex,
                    )
                    raise ex
            raise AllKeysExhaustedError(
                f"All primary and fallback models exhausted for primary model '{self.primary_model}'."
            ) from last_err


# ---------------------------------------------------------------------------
# Public factory function
# ---------------------------------------------------------------------------

def get_llm(role: str, model_name: str | None = None, provider: str | None = None) -> LLMInterface:
    """Dependency-injection factory for LLM instances.

    Args:
        role: Pipeline role – ``'generator'`` or ``'corrector'`` (or ``'critic'``
              as a legacy alias for corrector).
        model_name: Override the default model for the role.
        provider: Provider name (``'groq'`` or ``'google'``). If not set, it is
                  inferred from model name or environment.

    Raises:
        ValueError: If an unrecognised role is passed.

    Returns:
        An ``LLMInterface``-compatible wrapper.
    """
    role_lower = role.lower()

    _DEFAULTS: dict[str, tuple[str, int]] = {
        "generator": ("openai/gpt-oss-120b", 2048),
        "corrector": ("openai/gpt-oss-20b", 1024),
        "critic":    ("openai/gpt-oss-20b", 1024),   # legacy alias
        # Reflection uses the scout model: blazing fast, always available.
        "reflector": ("meta-llama/llama-4-scout-17b-16e-instruct", 512),
    }

    if role_lower not in _DEFAULTS:
        raise ValueError(
            f"Unsupported LLM role: '{role}'. Valid roles: {list(_DEFAULTS.keys())}"
        )

    default_model, default_max_tokens = _DEFAULTS[role_lower]
    resolved_model = model_name or default_model

    # Determine provider
    resolved_provider = provider
    if not resolved_provider:
        if resolved_model and ("gemini" in resolved_model.lower() or "google" in resolved_model.lower() or "gemma" in resolved_model.lower()):
            resolved_provider = "google"
        elif role_lower == "generator":
            resolved_provider = os.environ.get("GENERATOR_PROVIDER", "groq")
        elif role_lower in ("critic", "corrector"):
            resolved_provider = os.environ.get("CRITIC_PROVIDER", "google" if ("gemini" in resolved_model.lower() or "gemma" in resolved_model.lower()) else "groq")
        else:
            resolved_provider = "groq"

    resolved_provider = resolved_provider.lower()

    if resolved_provider in ("google", "gemini"):
        logger.info("[LLMFactory] Initialising role='%s' model='%s' (Gemini)", role, resolved_model)
        return ResilientGeminiLLM(primary_model=resolved_model, max_tokens=default_max_tokens)
    else:
        logger.info("[LLMFactory] Initialising role='%s' model='%s' (Groq)", role, resolved_model)
        return ResilientGroqLLM(primary_model=resolved_model, max_tokens=default_max_tokens)
