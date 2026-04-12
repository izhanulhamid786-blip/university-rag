import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from google.genai.errors import ClientError

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.llm import LLMConfigurationError, get_genai_client
from rag.memory import rewrite_query
from rag.prompt import build_prompt
from rag.reranker import preload_reranker, rerank
from rag.retriever import collection_status, get_collection, hybrid_retrieve, preload_embedder
from rag.settings import get_settings


log = logging.getLogger(__name__)
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
EMERGENCY_GENERATOR_MODELS = (
    "gemini-2.5-flash-lite",
    "gemini-flash-latest",
    "gemini-3-flash-preview",
)


@dataclass
class TextChunk:
    text: str


def _message_stream(message: str):
    yield TextChunk(message)


def _chunk_body(chunk: dict) -> str:
    text = (chunk.get("text") or "").strip()
    if "\n\n" in text:
        return text.split("\n\n", 1)[1].strip()
    return text


def _snippet(text: str, limit: int = 240) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _local_fallback_answer(query: str, chunks: list[dict], *, quota_hit: bool = False) -> str:
    lines = []
    if quota_hit:
        lines.append(
            "Gemini quota is exhausted right now, so here is a retrieval-only answer from the indexed documents."
        )
    else:
        lines.append(
            "The generation service is unavailable right now, so here is the best answer I can assemble from the indexed documents."
        )

    if query:
        lines.append(f"Question: {query}")

    if not chunks:
        lines.append("I couldn't find enough supporting documents to answer confidently.")
        return "\n\n".join(lines)

    lines.append("Most relevant documents:")
    for index, chunk in enumerate(chunks[:3], start=1):
        title = chunk.get("title") or f"Source {index}"
        category = chunk.get("category", "general")
        body = _snippet(_chunk_body(chunk))
        lines.append(f"[{index}] {title} ({category})")
        if body:
            lines.append(body)

    lines.append("Open the source links below for the exact official notices and PDFs.")
    return "\n\n".join(lines)


def _generation_models() -> list[str]:
    settings = get_settings()
    models = []
    for model_name in (
        settings.generator_model,
        settings.fallback_generator_model,
        *EMERGENCY_GENERATOR_MODELS,
    ):
        if model_name and model_name not in models:
            models.append(model_name)
    return models


def _query_tokens(query: str) -> list[str]:
    return TOKEN_RE.findall((query or "").lower())


def _looks_like_exact_lookup(query: str) -> bool:
    tokens = _query_tokens(query)
    if not tokens:
        return False

    entity_tokens = [token for token in tokens if token not in EXACT_LOOKUP_WORDS and len(token) > 2]
    if not (2 <= len(entity_tokens) <= 5):
        return False

    has_lookup_shape = (
        query.strip().lower().startswith(("who is", "who's", "what is", "what's"))
        or bool(set(tokens) & {"contact", "email", "phone", "designation", "details"})
    )
    return has_lookup_shape


def _rerank_candidate_limit(query: str, total_candidates: int) -> int:
    settings = get_settings()
    base_limit = min(total_candidates, max(settings.rerank_top_k, settings.rerank_candidate_k))
    if _looks_like_exact_lookup(query):
        return min(base_limit, max(settings.rerank_top_k, 8))
    return base_limit


def warmup_local_models() -> None:
    try:
        get_collection(required=False)
    except Exception as exc:
        log.warning("Collection warmup skipped: %s", exc)

    for loader, label in ((preload_embedder, "embedder"), (preload_reranker, "reranker")):
        try:
            loader()
            log.info("Warmup complete: %s", label)
        except Exception as exc:
            log.warning("Warmup failed for %s: %s", label, exc)


def _is_quota_error(exc: Exception) -> bool:
    if isinstance(exc, ClientError):
        code = getattr(exc, "status_code", None)
        if code is None:
            code = getattr(exc, "code", None)
        if code == 429:
            return True
    message = str(exc).lower()
    return "429" in message or "resource_exhausted" in message or "quota" in message


def _generation_stream(prompt: str, *, query: str, fallback_chunks: list[dict]):
    try:
        client = get_genai_client(required=True)
        last_error = None
        for model_name in _generation_models():
            try:
                log.info("Generating with model: %s", model_name)
                for item in client.models.generate_content_stream(
                    model=model_name,
                    contents=prompt,
                ):
                    yield item
                return
            except Exception as exc:
                last_error = exc
                if _is_quota_error(exc):
                    log.warning("Generation quota hit for model %s: %s", model_name, exc)
                    continue
                raise
        if last_error is not None:
            raise last_error
    except LLMConfigurationError as exc:
        log.warning("Generator configuration issue: %s", exc)
        yield TextChunk(str(exc))
    except Exception as exc:
        log.exception("Generation failed")
        yield TextChunk(
            _local_fallback_answer(
                query,
                fallback_chunks,
                quota_hit=_is_quota_error(exc),
            )
        )


def _build_sources(chunks: list[dict]) -> list[dict]:
    sources = []
    seen = set()
    for chunk in chunks:
        key = chunk.get("source_url") or chunk.get("source_path") or chunk.get("source")
        if not key or key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "label": chunk.get("title") or key,
                "url": chunk.get("source_url"),
                "path": chunk.get("source_path"),
                "category": chunk.get("category", "general"),
            }
        )
    return sources


def app_status() -> dict:
    settings = get_settings()
    kb = collection_status()
    return {
        "generator_configured": bool(settings.google_api_key),
        "knowledge_base_ready": kb["ready"],
        "collection_name": kb["collection_name"],
        "chunk_count": kb["count"],
        "message": kb["message"],
    }


def run(query: str, history: list[dict]):
    clean_query = (query or "").strip()
    if not clean_query:
        return _message_stream("Please ask a question about the university."), []

    started = time.perf_counter()
    smart_query = rewrite_query(clean_query, history)
    log.info("Smart query: %s", smart_query)
    after_rewrite = time.perf_counter()

    candidates = hybrid_retrieve(smart_query)
    log.info("Retrieved %s candidates", len(candidates))
    after_retrieve = time.perf_counter()
    if not candidates:
        return _message_stream(
            "I don't have that information. Please contact the university office directly."
        ), []

    settings = get_settings()
    rerank_limit = _rerank_candidate_limit(smart_query, len(candidates))
    rerank_input = candidates[:rerank_limit]
    top_chunks = rerank(smart_query, rerank_input, top_k=settings.rerank_top_k)
    after_rerank = time.perf_counter()
    sources = _build_sources(top_chunks)
    prompt = build_prompt(smart_query, top_chunks, history)
    after_prompt = time.perf_counter()
    log.info(
        "Pipeline timings | rewrite=%.3fs retrieve=%.3fs rerank=%.3fs prompt=%.3fs total=%.3fs candidates=%s rerank_input=%s",
        after_rewrite - started,
        after_retrieve - after_rewrite,
        after_rerank - after_retrieve,
        after_prompt - after_rerank,
        after_prompt - started,
        len(candidates),
        len(rerank_input),
    )
    return _generation_stream(prompt, query=smart_query, fallback_chunks=top_chunks), sources


if __name__ == "__main__":
    stream, sources = run("What is the admission process at CUK?", [])
    for item in stream:
        if getattr(item, "text", ""):
            print(item.text, end="", flush=True)
    print("\n\nSources:", sources)
