"""RAG package for the university assistant.

Phase 2 modules can stay tucked under ``rag/phase2`` while legacy imports
such as ``rag.model_loading`` and ``rag.pipeline`` continue to work.
"""

from pathlib import Path


_PHASE2_DIR = Path(__file__).resolve().parent / "phase2"
if _PHASE2_DIR.is_dir():
    phase2_path = str(_PHASE2_DIR)
    if phase2_path not in __path__:
        __path__.append(phase2_path)
