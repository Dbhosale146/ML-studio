from ...core.config import Settings
from .base import LLM
from .mock import MockLLM


def make_llm(settings: Settings) -> LLM:
    """Gemini when LLM_PROVIDER=gemini and a key is configured, otherwise the offline MockLLM."""
    if settings.llm_enabled:
        from .gemini import GeminiLLM

        return GeminiLLM(settings.gemini_api_key, settings.gemini_model, settings.llm_timeout_s)
    return MockLLM()


__all__ = ["LLM", "MockLLM", "make_llm"]
