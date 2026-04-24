"""Compatibility shim for older imports that expect ``rag.phase2.pipeline``."""

from rag.pipeline import (
    TextChunk,
    app_status,
    run,
    run_with_metadata,
    warmup_local_models,
)

__all__ = [
    "TextChunk",
    "app_status",
    "run",
    "run_with_metadata",
    "warmup_local_models",
]
