import logging
import re
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
TABLE_FACT_QUERY_RE = re.compile(
    r"\b(form|forms|application|registration|roll|number|numbers|selected|selection|eligible|"
    r"eligibility|candidate|candidates|list|table)\b",
    re.IGNORECASE,
)
FORM_NUMBER_RE = re.compile(r"\b(?:CUK\d{4,}|form\s*(?:no|nos|number|numbers)|application\s*(?:no|number|numbers))\b", re.I)
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


def _retrieval_prior(query: str, chunk: dict) -> float:
    if not TABLE_FACT_QUERY_RE.search(query or ""):
        return 0.0

    prior = min(0.8, 1.5 * float(chunk.get("final_score", 0.0)))
    text = " ".join([chunk.get("title") or "", chunk.get("text") or ""])
    if chunk.get("has_table"):
        prior += 0.12
    if FORM_NUMBER_RE.search(text):
        prior += 0.16
    return prior


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

    ranked = sorted(
        zip(chunks, scores),
        key=lambda item: float(item[1]) + _retrieval_prior(query, item[0]),
        reverse=True,
    )
    filtered = [item for item in ranked if float(item[1]) + _retrieval_prior(query, item[0]) >= MIN_RERANK_SCORE]
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
        item["rerank_final_score"] = float(score) + _retrieval_prior(query, chunk)
        results.append(item)
    return results


if __name__ == "__main__":
    from rag.retriever import hybrid_retrieve

    query = "phd admission process"
    reranked = rerank(query, hybrid_retrieve(query, k=15), top_k=5)
    for index, item in enumerate(reranked, start=1):
        print(f"[{index}] {item.get('title', 'Untitled')} | rerank={item.get('rerank_score', 0):.3f}")
