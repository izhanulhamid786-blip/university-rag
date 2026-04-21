import logging
import sys
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

from sentence_transformers import CrossEncoder

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.settings import get_settings


log = logging.getLogger(__name__)
MIN_RERANK_SCORE = -2.3
_TRANSFORMER_LOAD_LOGGERS = (
    "transformers.core_model_loading",
    "transformers.integrations.peft",
    "transformers.integrations.tensor_parallel",
    "transformers.modeling_utils",
    "transformers.utils.loading_report",
)


@contextmanager
def quiet_transformer_loading():
    original_levels = []
    try:
        for name in _TRANSFORMER_LOAD_LOGGERS:
            logger = logging.getLogger(name)
            original_levels.append((logger, logger.level))
            if logger.level < logging.ERROR:
                logger.setLevel(logging.ERROR)
        yield
    finally:
        for logger, level in original_levels:
            logger.setLevel(level)


@lru_cache(maxsize=1)
def _reranker() -> CrossEncoder:
    settings = get_settings()
    with quiet_transformer_loading():
        return CrossEncoder(
            settings.rerank_model,
            local_files_only=settings.local_files_only,
        )


def preload_reranker() -> None:
    _reranker()


def rerank(query: str, chunks: list[dict], top_k: int | None = None) -> list[dict]:
    settings = get_settings()
    limit = top_k or settings.rerank_top_k
    if not chunks:
        return []
    if len(chunks) <= 1:
        return chunks[:limit]

    try:
        pairs = [(query, chunk["text"]) for chunk in chunks]
        scores = _reranker().predict(pairs)
    except Exception as exc:
        log.warning("Reranker unavailable, falling back to retrieval order: %s", exc)
        return chunks[:limit]

    ranked = sorted(zip(chunks, scores), key=lambda item: float(item[1]), reverse=True)
    filtered = [item for item in ranked if float(item[1]) >= MIN_RERANK_SCORE]
    if filtered:
        ranked = filtered
    else:
        log.info(
            "All rerank scores fell below %.2f for query %r; using relative rerank order as fallback.",
            MIN_RERANK_SCORE,
            query,
        )

    results = []
    for chunk, score in ranked[:limit]:
        item = dict(chunk)
        item["rerank_score"] = float(score)
        results.append(item)
    return results


if __name__ == "__main__":
    from rag.retriever import hybrid_retrieve

    query = "phd admission process"
    reranked = rerank(query, hybrid_retrieve(query, k=15), top_k=5)
    for index, item in enumerate(reranked, start=1):
        print(f"[{index}] {item.get('title', 'Untitled')} | rerank={item.get('rerank_score', 0):.3f}")
