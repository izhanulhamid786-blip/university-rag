"""Compatibility shim for older imports that expect ``rag.phase2.memory``."""

from rag.memory import rewrite_query

__all__ = ["rewrite_query"]
