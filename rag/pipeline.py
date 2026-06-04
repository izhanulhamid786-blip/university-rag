import logging
import json
import re
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.llm import GeneratorConnectionError, generate_text
from rag.memory import rewrite_query
from rag.prompt import build_prompt
from rag.reranker import preload_reranker, rerank
from rag.retriever import collection_status, get_collection, hybrid_retrieve, preload_embedder
from rag.settings import get_settings
from rag.text_cleanup import clean_text_artifacts


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
GREETING_QUERIES = {
    "hi",
    "hello",
    "hey",
    "yo",
    "good morning",
    "good afternoon",
    "good evening",
}
OFFICIAL_PUBLIC_ZONE_URL = "https://cukashmir.ac.in/#/publiczone"





def _chunk_body(chunk: dict) -> str:
    text = clean_text_artifacts(chunk.get("text") or "")
    if "\n\n" in text:
        return text.split("\n\n", 1)[1].strip()
    return text


def _snippet(text: str, limit: int = 240) -> str:
    cleaned = clean_text_artifacts(text)
    if _has_table_shape(cleaned):
        return _table_snippet(cleaned, limit=limit)

    compact = " ".join(cleaned.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _has_table_shape(text: str) -> bool:
    return any(line.count("|") >= 2 for line in text.splitlines())


def _table_snippet(text: str, limit: int = 240) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    selected = []
    for line in lines:
        if line.count("|") >= 2:
            selected.append(line)
        elif selected:
            break
        elif len(line) <= 120:
            selected = [line]

        candidate = "\n".join(selected)
        if len(candidate) >= limit:
            break

    snippet = "\n".join(selected).strip() or " ".join(text.split())
    if len(snippet) <= limit:
        return snippet
    return snippet[: limit - 3].rstrip() + "..."


def _source_host(url: str | None) -> str:
    if not url:
        return ""
    return urlparse(url).netloc.lower()


def _is_broken_source_url(url: str | None) -> bool:
    if not url:
        return True
    url = str(url).strip()
    if not url:
        return True
    parsed = urlparse(url)
    return not (parsed.scheme in {"http", "https"} and parsed.netloc)


def _is_api_source_url(url: str | None) -> bool:
    return _source_host(url) == "cukapi.disgenweb.in"


@lru_cache(maxsize=512)
def _source_page_from_path(source_path: str | None) -> str:
    if not source_path:
        return ""

    try:
        path = Path(source_path).resolve()
        root = Path(get_settings().data_dir).resolve()
        if root not in path.parents and path != root:
            return ""
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return ""

    source_page = str(payload.get("source_page") or "").strip()
    if source_page.startswith(("http://", "https://")):
        return source_page
    return ""


def _clickable_source_url(chunk: dict) -> str | None:
    source_url = chunk.get("source_url")
    source_path = chunk.get("source_path")
    public_page = _source_page_from_path(source_path)
    
    text = chunk.get("text", "")
    import re
    file_match = re.search(r"FileUrl:\s*(https?://[^\n\r]+)", text)
    if file_match:
        return file_match.group(1).strip()
        
    link_match = re.search(r"Link:\s*(https?://[^\n\r]+)", text)
    if link_match:
        return link_match.group(1).strip()

    # Serve local PDFs through API static mount
    if not source_url and source_path and source_path.endswith(".pdf"):
        import urllib.parse
        from pathlib import Path
        try:
            path_obj = Path(source_path)
            if "manual" in path_obj.parts:
                return f"/files/manual/{urllib.parse.quote(path_obj.name)}"
            elif "pdfs" in path_obj.parts:
                return f"/files/pdfs/{urllib.parse.quote(path_obj.name)}"
        except Exception:
            pass

    if _is_api_source_url(source_url):
        return public_page or OFFICIAL_PUBLIC_ZONE_URL
    if not _is_broken_source_url(source_url):
        return source_url
    return public_page or None


def _local_fallback_answer(
    query: str,
    chunks: list[dict],
    *,
    quota_hit: bool = False,
    generator_error: str | None = None,
) -> str:
    lines = []
    provider_label = get_settings().generator_provider.title()
    if quota_hit:
        lines.append(
            f"**{provider_label} quota exhausted**\n"
            f"The {provider_label} API quota is exhausted right now, so this is a concise evidence summary from the indexed documents."
        )
    elif generator_error:
        lines.append(
            "**Answer generator error**\n"
            f"The answer generator failed with {generator_error}, so this is a concise evidence summary from the indexed documents."
        )
    else:
        lines.append(
            "**Answer quality note**\n"
            "The local answer generator is unavailable right now, so this is a concise evidence summary from the indexed documents."
        )

    if query:
        lines.append(f"**Question**\n{query}")

    if not chunks:
        lines.append("I couldn't find enough supporting documents to answer confidently.")
        return "\n\n".join(lines)

    lines.append("**Most relevant official material**")
    for index, chunk in enumerate(chunks[:3], start=1):
        title = clean_text_artifacts(chunk.get("title") or f"Source {index}")
        category = chunk.get("category", "general")
        body = _snippet(_chunk_body(chunk))
        lines.append(f"- [{index}] **{title}** ({category})")
        if body:
            lines.append(f"  {body}")

    lines.append("**Recommended next step**\nOpen the evidence cards below for the exact official notices and PDFs.")
    return "\n\n".join(lines)


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


def _is_greeting(query: str) -> bool:
    normalized = re.sub(r"[^\w\s]", "", (query or "").strip().lower())
    normalized = " ".join(normalized.split())
    return normalized in GREETING_QUERIES


def _merge_candidates(primary: list[dict], secondary: list[dict]) -> list[dict]:
    merged = []
    seen = set()
    for item in primary + secondary:
        key = item.get("chunk_id") or item.get("id") or (
            item.get("source_url"),
            item.get("source_path"),
            item.get("chunk_index"),
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged


def _rerank_candidate_limit(query: str, total_candidates: int) -> int:
    settings = get_settings()
    base_limit = min(total_candidates, max(settings.rerank_top_k, 80))
    if _looks_like_exact_lookup(query):
        return min(base_limit, max(settings.rerank_top_k, 80))
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
                "citation": len(sources) + 1,
                "label": clean_text_artifacts(chunk.get("title") or key),
                "url": _clickable_source_url(chunk),
                "raw_url": chunk.get("source_url"),
                "path": chunk.get("source_path"),
                "category": chunk.get("category", "general"),
                "preview": _snippet(_chunk_body(chunk), limit=220),
                "matched_by": chunk.get("matched_by", []),
                "rerank_score": chunk.get("rerank_score"),
                "final_score": chunk.get("final_score"),
            }
        )
    return sources


def app_status() -> dict:
    settings = get_settings()
    kb = collection_status()
    return {
        "generator_configured": bool(settings.generator_api_key),
        "generator_provider": settings.generator_provider,
        "generator_model": settings.generator_model,
        "knowledge_base_ready": kb["ready"],
        "collection_name": kb["collection_name"],
        "chunk_count": kb["count"],
        "vector_db_dir": str(settings.vector_db_dir),
        "message": kb["message"],
    }


def run_with_metadata(
    query: str,
    history: list[dict],
    *,
    answer_style: str = "balanced",
):
    clean_query = (query or "").strip()
    metadata = {
        "query": clean_query,
        "rewritten_query": clean_query,
        "candidate_count": 0,
        "rerank_input_count": 0,
        "selected_chunk_count": 0,
        "timings_ms": {},
    }
    if not clean_query:
        return "Please ask a question about the university.", [], metadata
    if _is_greeting(clean_query):
        return "Hello. Ask me about CUK admissions, notices, departments, faculty, contacts, or indexed university documents.", [], metadata

    started = time.perf_counter()
    smart_query = rewrite_query(clean_query, history)
    metadata["rewritten_query"] = smart_query
    log.info("Smart query: %s", smart_query)
    after_rewrite = time.perf_counter()

    candidates = hybrid_retrieve(smart_query)
    if smart_query != clean_query:
        original_candidates = hybrid_retrieve(clean_query)
        candidates = _merge_candidates(candidates, original_candidates)
    metadata["candidate_count"] = len(candidates)
    log.info("Retrieved %s candidates", len(candidates))
    after_retrieve = time.perf_counter()
    if not candidates:
        metadata["timings_ms"] = {
            "rewrite": round((after_rewrite - started) * 1000, 1),
            "retrieve": round((after_retrieve - after_rewrite) * 1000, 1),
            "total": round((after_retrieve - started) * 1000, 1),
        }
        return "I don't have that information. Please contact the university office directly.", [], metadata

    settings = get_settings()
    rerank_limit = _rerank_candidate_limit(smart_query, len(candidates))
    rerank_input = candidates[:rerank_limit]
    metadata["rerank_input_count"] = len(rerank_input)
    top_chunks = rerank(smart_query, rerank_input, top_k=settings.rerank_top_k)
    metadata["selected_chunk_count"] = len(top_chunks)
    after_rerank = time.perf_counter()
    sources = _build_sources(top_chunks)
    prompt = build_prompt(smart_query, top_chunks, history, answer_style=answer_style)
    after_prompt = time.perf_counter()
    metadata["retrieved_contexts"] = [_chunk_body(chunk) for chunk in top_chunks]
    metadata["retrieved_context_sources"] = [
        {
            "title": chunk.get("title") or "Untitled",
            "url": chunk.get("source_url"),
            "path": chunk.get("source_path"),
            "category": chunk.get("category", "general"),
        }
        for chunk in top_chunks
    ]
    metadata["timings_ms"] = {
        "rewrite": round((after_rewrite - started) * 1000, 1),
        "retrieve": round((after_retrieve - after_rewrite) * 1000, 1),
        "rerank": round((after_rerank - after_retrieve) * 1000, 1),
        "prompt": round((after_prompt - after_rerank) * 1000, 1),
        "total": round((after_prompt - started) * 1000, 1),
    }
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
    try:
        log.info("Generating with %s model: %s", settings.generator_provider, settings.generator_model)
        answer = generate_text(prompt, stream=False)
        answer = clean_text_artifacts(answer)
    except GeneratorConnectionError as exc:
        log.warning("%s generation failed; using local fallback: %s", settings.generator_provider, exc)
        answer = _local_fallback_answer(
            smart_query,
            top_chunks,
            quota_hit=getattr(exc, "quota_exhausted", False),
            generator_error=getattr(exc, "error_name", exc.__class__.__name__),
        )

    return answer, sources, metadata


def run(
    query: str,
    history: list[dict],
    *,
    answer_style: str = "balanced",
):
    answer, sources, _ = run_with_metadata(query, history, answer_style=answer_style)
    return answer, sources


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    query = " ".join(sys.argv[1:]).strip()
    if not query and sys.stdin.isatty():
        query = input("Question: ").strip()
    query = query or "What is the admission process at CUK?"

    answer, sources = run(query, [])
    if answer:
        print(answer)
    if sources:
        print("\n\nSources:")
        for source in sources:
            label = source.get("label") or "Untitled source"
            citation = source.get("citation", "?")
            location = source.get("url") or source.get("path") or ""
            print(f"[{citation}] {label}")
            if location:
                print(f"    {location}")
