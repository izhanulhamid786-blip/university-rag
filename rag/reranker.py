import logging
import re
import sys
import textwrap
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

from sentence_transformers import CrossEncoder

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.settings import get_settings
from rag.model_loading import (
    clear_broken_proxy_env,
    recover_closed_huggingface_session,
    reset_huggingface_session_if_closed,
)


log = logging.getLogger(__name__)
MIN_RERANK_SCORE = -2.3
TOKEN_RE = re.compile(r"\w+")
EXACT_LOOKUP_WORDS = {
    "who",
    "is",
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "professor",
    "er",
    "contact",
    "contacts",
    "email",
    "emails",
    "phone",
    "phones",
    "mobile",
    "number",
    "numbers",
    "office",
    "designation",
    "details",
    "detail",
    "about",
    "of",
    "the",
    "a",
    "an",
}
DIRECTORY_CATEGORIES = {"contact", "faculty", "departments", "administration", "about"}
DIRECTORY_TEXT_HINTS = {
    "assistant professor",
    "associate professor",
    "professor",
    "teacher",
    "faculty",
    "department",
    "dean",
    "director",
    "coordinator",
    "head",
    "hod",
    "registrar",
    "email",
    "contact",
    "phone",
    "mobile",
}
INCIDENTAL_TEXT_HINTS = {
    "attended",
    "workshop",
    "course",
    "programme",
    "event",
    "lecture",
    "resource person",
    "newsletter",
    "seminar",
    "conference",
}
TABLE_FACT_QUERY_RE = re.compile(
    r"\b(form|forms|application|registration|roll|number|numbers|selected|selection|eligible|"
    r"eligibility|candidate|candidates|list|table)\b",
    re.IGNORECASE,
)
FORM_NUMBER_RE = re.compile(r"\b(?:CUK\d{4,}|form\s*(?:no|nos|number|numbers)|application\s*(?:no|number|numbers))\b", re.I)
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE_RE = re.compile(r"(?:\+91[\s-]?)?(?:\d[\s-]?){10,13}")
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
    clear_broken_proxy_env()
    for attempt in range(2):
        reset_huggingface_session_if_closed()
        try:
            with quiet_transformer_loading():
                return CrossEncoder(
                    settings.rerank_model,
                    local_files_only=settings.local_files_only,
                )
        except RuntimeError as exc:
            if attempt == 0 and recover_closed_huggingface_session(exc):
                continue
            raise

    raise RuntimeError("Failed to load reranking model.")


def preload_reranker() -> None:
    _reranker()


def _query_tokens(query: str) -> list[str]:
    return TOKEN_RE.findall((query or "").lower())


def _looks_like_exact_lookup(query: str) -> bool:
    tokens = _query_tokens(query)
    if not tokens:
        return False

    entity_tokens = [token for token in tokens if token not in EXACT_LOOKUP_WORDS and len(token) > 2]
    if not (2 <= len(entity_tokens) <= 5):
        return False

    return (
        query.strip().lower().startswith(("who is", "who's", "what is", "what's"))
        or bool(set(tokens) & {"contact", "email", "phone", "designation", "details"})
    )


def _has_exact_entity(query: str, chunk: dict) -> bool:
    entity_tokens = [token for token in _query_tokens(query) if token not in EXACT_LOOKUP_WORDS and len(token) > 2]
    if not entity_tokens:
        return False

    haystack = " ".join([chunk.get("title") or "", chunk.get("text") or ""]).lower()
    return all(token in haystack for token in entity_tokens)


def _body_text(chunk: dict) -> str:
    text = chunk.get("text") or ""
    if "\n\n" in text:
        return text.split("\n\n", 1)[1]
    return text


def _has_phone_contact(text: str) -> bool:
    for match in PHONE_RE.finditer(text):
        digits = re.sub(r"\D", "", match.group(0))
        if digits.startswith("91") and len(digits) > 10:
            digits = digits[2:]
        if len(digits) == 10 and digits[0] in {"6", "7", "8", "9"}:
            return True
    return False


def _entity_window(query: str, chunk: dict, radius: int = 500) -> str:
    entity_tokens = [token for token in _query_tokens(query) if token not in EXACT_LOOKUP_WORDS and len(token) > 2]
    text = _body_text(chunk).lower()
    positions = [text.find(token) for token in entity_tokens if text.find(token) >= 0]
    if not positions:
        return text[: radius * 2]

    start = max(0, min(positions) - radius)
    end = min(len(text), max(positions) + radius)
    return text[start:end]


def _exact_lookup_prior(query: str, chunk: dict) -> float:
    if not _looks_like_exact_lookup(query):
        return 0.0

    category = (chunk.get("category") or "").lower()
    if not _has_exact_entity(query, chunk):
        return -1.5

    prior = 0.0
    entity_context = _entity_window(query, chunk)
    has_contact = bool(EMAIL_RE.search(entity_context) or _has_phone_contact(entity_context))
    has_directory_hint = category in DIRECTORY_CATEGORIES or any(hint in entity_context for hint in DIRECTORY_TEXT_HINTS)
    has_incidental_hint = any(hint in entity_context for hint in INCIDENTAL_TEXT_HINTS)

    if has_contact:
        prior += 3.0
    if category == "contact":
        prior += 0.5
    if has_directory_hint:
        prior += 0.85
    if category in {"academics", "general"} and has_incidental_hint and not has_contact:
        prior -= 2.25
    if has_incidental_hint and not has_directory_hint:
        prior -= 0.75
    return prior


def _retrieval_prior(query: str, chunk: dict) -> float:
    exact_prior = _exact_lookup_prior(query, chunk)
    if not TABLE_FACT_QUERY_RE.search(query or ""):
        return exact_prior

    prior = min(0.8, 1.5 * float(chunk.get("final_score", 0.0)))
    text = " ".join([chunk.get("title") or "", chunk.get("text") or ""])
    if chunk.get("has_table"):
        prior += 0.12
    if FORM_NUMBER_RE.search(text):
        prior += 0.16
    return prior + exact_prior


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


DEBUG_RESULT_LIMIT = 3
DEBUG_PREVIEW_CHARS = 450


def _debug_preview(text: str, limit: int = DEBUG_PREVIEW_CHARS) -> str:
    compact = re.sub(r"\s+", " ", (text or "")).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _debug_indent(text: str, *, width: int = 120, indent: str = "    ") -> str:
    return textwrap.fill(
        text,
        width=width,
        initial_indent=indent,
        subsequent_indent=indent,
        break_long_words=False,
        break_on_hyphens=False,
    )


def _debug_print_result(index: int, item: dict, *, preview_chars: int = DEBUG_PREVIEW_CHARS) -> None:
    matched_by = ", ".join(item.get("matched_by", [])) or "unknown"
    retrieval_score = float(item.get("final_score", 0.0))
    rerank_score = float(item.get("rerank_score", 0.0))
    preview = _debug_indent(_debug_preview(item.get("text", ""), preview_chars))
    print(
        f"\n[{index}] {item.get('title', 'Untitled')}\n"
        f"    Category: {item.get('category', 'general')}\n"
        f"    Score: {rerank_score:.3f} (retrieval={retrieval_score:.4f})\n"
        f"    Matched by: {matched_by}\n"
        f"    URL: {item.get('source_url') or 'N/A'}\n"
        f"    Preview:\n"
        f"{preview}"
    )


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    from rag.retriever import hybrid_retrieve

    query = " ".join(sys.argv[1:]).strip() or "phd admission process"

    candidates = hybrid_retrieve(query, k=15)
    reranked = rerank(query, candidates, top_k=DEBUG_RESULT_LIMIT)
    print(f"\nTop {min(DEBUG_RESULT_LIMIT, len(reranked))} reranked results for: {query!r}")
    print(f"Candidates considered: {len(candidates)}")
    for index, item in enumerate(reranked, start=1):
        _debug_print_result(index, item)
