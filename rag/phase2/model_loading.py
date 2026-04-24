"""Compatibility shim for older imports that expect ``rag.phase2.model_loading``."""

from rag.model_loading import quiet_transformer_loading

__all__ = ["quiet_transformer_loading"]
