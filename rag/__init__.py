"""RAG package for the university assistant.

Core modules now live directly under ``rag``. The ``rag.phase2`` package keeps
small compatibility shims so older imports continue to work during the
transition.
"""

from pathlib import Path


_PHASE2_DIR = Path(__file__).resolve().parent / "phase2"
if _PHASE2_DIR.is_dir():
    phase2_path = str(_PHASE2_DIR)
    if phase2_path not in __path__:
        __path__.append(phase2_path)
