"""Compatibility shim for older imports that expect ``rag.phase2.llm``."""

from rag.llm import LLMConfigurationError, get_genai_client

__all__ = ["LLMConfigurationError", "get_genai_client"]
