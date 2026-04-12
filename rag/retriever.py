import json
import logging
import re
import sys
from functools import lru_cache
from pathlib import Path
from urllib.parse import unquote, urlparse

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.settings import get_settings


log = logging.getLogger(__name__)
TOKEN_RE = re.compile(r"\w+")
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE_RE = re.compile(r"(?:\+91[\s-]?)?(?:\d[\s-]?){10,13}")
RRF_K = 60
STOP_WORDS = {
    "a",
    "an",
    "and",
    "at",
    "by",
    "central",
    "for",
    "how",
    "i",
    "in",
    "is",
    "kashmir",
    "me",
    "of",
    "on",
    "or",
    "the",
    "to",
    "university",
    "what",
    "when",
    "where",
}
PERSON_QUERY_STOP_WORDS = {
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
    "staff",
    "teacher",
    "teachers",
    "faculty",
    "member",
    "members",
    "department",
    "departments",
    "school",
    "schools",
    "professor",
    "prof",
    "dean",
    "director",
    "coordinator",
    "head",
    "hod",
    "details",
    "detail",
    "information",
    "info",
    "there",
    "many",
    "count",
    "total",
}
LINK_WORDS = {
    "link",
    "url",
    "website",
    "apply",
    "application",
    "form",
    "download",
    "notification",
    "notice",
    "pdf",
    "prospectus",
}
STAFF_QUERY_WORDS = {
    "staff",
    "teacher",
    "teachers",
    "faculty",
    "professor",
    "prof",
    "dean",
    "director",
    "coordinator",
    "coordinators",
    "head",
    "hod",
    "registrar",
    "officer",
    "employee",
    "employees",
    "contact",
    "contacts",
    "email",
    "emails",
    "phone",
    "phones",
    "mobile",
    "designation",
}
CONTACT_QUERY_WORDS = {"contact", "contacts", "email", "emails", "phone", "phones", "mobile", "office", "address"}
COUNT_QUERY_WORDS = {"count", "counts", "many", "number", "numbers", "total", "totals"}
STAFF_TEXT_HINTS = {
    "professor",
    "prof.",
    "assistant professor",
    "associate professor",
    "dean",
    "director",
    "coordinator",
    "head of department",
    "hod",
    "registrar",
    "controller of examinations",
    "email",
    "contact",
    "phone",
    "mobile",
}
STAFF_URL_HINTS = {
    "administration",
    "faculty",
    "department",
    "departlist",
    "departmentlist",
    "contact",
    "staff",
    "employee",
    "school",
    "dean",
    "director",
    "coordinator",
}
LOW_VALUE_STAFF_HINTS = {
    "ordinance",
    "statute",
    "act",
    "policy",
    "minutes",
    "agenda",
    "newsletter",
    "result",
    "admission",
    "notice",
    "notification",
    "circular",
    "aqar",
    "naac",
    "nirf",
    "tender",
}
COUNT_TEXT_HINTS = {
    "total",
    "number of",
    "faculty members",
    "teaching staff",
    "non-teaching",
    "sanctioned",
    "filled",
    "vacant",
}
INTENT_CATEGORY_HINTS = {
    "admissions": {"admission", "admissions", "apply", "application", "cuet", "eligibility", "prospectus"},
    "fees": {"fee", "fees", "refund", "payment", "scholarship"},
    "examinations": {"exam", "examination", "datesheet", "date", "schedule", "timetable"},
    "results": {"result", "results", "grade", "marks"},
    "recruitment": {"job", "jobs", "recruitment", "vacancy", "advertisement"},
    "departments": {"department", "departments", "faculty", "programme", "programmes", "school", "syllabus"},
    "contact": {"contact", "phone", "email", "address", "office"},
}
UNIVERSITY_HOSTS = {
    "cukashmir.ac.in",
    "www.cukashmir.ac.in",
    "results.cukashmir.in",
    "www.results.cukashmir.in",
    "cukapi.disgenweb.in",
}
UTILITY_URL_HINTS = {"screenreaderaccess", "event-gallery", "implink"}
NOISY_TEXT_HINTS = {
    "page not found",
    "server error",
    "404 not found",
    "the page you're looking for does not seem to exist",
}
NOISY_SECTION_HINTS = {"quick links", "go to home"}
DEPARTMENT_URL_HINTS = {"departlist", "departmentlist"}


class KnowledgeBaseNotReadyError(RuntimeError):
    pass


_BM25_CACHE = {
    "count": None,
    "bm25": None,
    "texts": None,
    "metadatas": None,
    "ids": None,
}


@lru_cache(maxsize=1)
def _embedder() -> SentenceTransformer:
    settings = get_settings()
    return SentenceTransformer(
        settings.embed_model,
        local_files_only=settings.local_files_only,
    )


def preload_embedder() -> None:
    _embedder()


@lru_cache(maxsize=1)
def _client() -> chromadb.PersistentClient:
    settings = get_settings()
    return chromadb.PersistentClient(path=str(settings.vector_db_dir))


def get_collection(required: bool = False):
    settings = get_settings()
    try:
        collection = _client().get_collection(settings.collection_name)
    except Exception as exc:
        if required:
            raise KnowledgeBaseNotReadyError(
                f"Collection '{settings.collection_name}' not found. Run `python -m ingest.build_db --reset` first."
            ) from exc
        return None

    if required and collection.count() == 0:
        raise KnowledgeBaseNotReadyError(
            f"Collection '{settings.collection_name}' is empty. Run `python -m ingest.build_db --reset` first."
        )
    return collection


def collection_status() -> dict:
    settings = get_settings()
    collection = get_collection(required=False)
    if collection is None:
        return {
            "ready": False,
            "collection_name": settings.collection_name,
            "count": 0,
            "message": "Vector collection is missing. Build it with `python -m ingest.build_db --reset`.",
        }

    count = collection.count()
    return {
        "ready": count > 0,
        "collection_name": settings.collection_name,
        "count": count,
        "message": "" if count > 0 else "Vector collection is empty. Build it with `python -m ingest.build_db --reset`.",
    }


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _significant_query_tokens(query: str) -> list[str]:
    tokens = [token for token in _tokenize(query) if len(token) > 2 and token not in STOP_WORDS]
    return tokens or _tokenize(query)


def _focus_query_tokens(query_tokens: set[str]) -> set[str]:
    return {
        token
        for token in query_tokens
        if len(token) > 2 and token not in PERSON_QUERY_STOP_WORDS and token not in STOP_WORDS
    }


def _is_link_intent(query: str) -> bool:
    tokens = set(_tokenize(query))
    return bool(tokens & LINK_WORDS)


def _is_staff_intent(query_tokens: set[str]) -> bool:
    return bool(query_tokens & STAFF_QUERY_WORDS)


def _is_contact_intent(query_tokens: set[str]) -> bool:
    return bool(query_tokens & CONTACT_QUERY_WORDS)


def _is_count_intent(query_tokens: set[str]) -> bool:
    return "how" in query_tokens and "many" in query_tokens or bool(query_tokens & COUNT_QUERY_WORDS)


def _coerce_bool(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _is_university_url(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return host in UNIVERSITY_HOSTS or host.endswith(".cukashmir.ac.in")


def _is_utility_url(url: str) -> bool:
    lower_url = url.lower()
    return any(hint in lower_url for hint in UTILITY_URL_HINTS)


def _canonical_url(url: str | None) -> str:
    if not url:
        return ""
    parsed = urlparse(url.strip())
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    path = unquote(parsed.path or "").rstrip("/")
    query = unquote(parsed.query or "")
    fragment = unquote(parsed.fragment or "")
    parts = [host, path]
    if query:
        parts.append(f"?{query}")
    if fragment:
        parts.append(f"#{fragment}")
    return "".join(parts)


def _display_title(title: str | None, source_url: str | None) -> str:
    current = (title or "").strip()
    source_name = ""
    if source_url:
        source_name = unquote(Path(urlparse(source_url).path).stem)
        source_name = re.sub(r"^\d{6,}[-_\s]*", "", source_name)
        source_name = re.sub(r"[_\-]+", " ", source_name)
        source_name = re.sub(r"\s+", " ", source_name).strip()

    lower_current = current.lower()
    current_is_generic = (
        not current
        or lower_current.startswith("microsoft word")
        or lower_current.startswith("microsoft powerpoint")
        or lower_current.startswith("microsoft excel")
        or "%20" in current
        or bool(re.match(r"^\d{6,}[-_\s]", current))
    )

    if current_is_generic and source_name:
        return source_name
    return current or source_name or "Untitled"


def _intent_categories(query_tokens: set[str]) -> set[str]:
    return {
        category
        for category, keywords in INTENT_CATEGORY_HINTS.items()
        if query_tokens & keywords
    }


def _source_key(item: dict) -> str:
    source_url = _canonical_url(item.get("source_url"))
    if source_url:
        return source_url
    title = (item.get("title") or "").strip().lower()
    category = (item.get("category") or "").strip().lower()
    text = (item.get("text") or "").strip().lower()[:160]
    return f"{title}::{category}::{text}"


def _metadata_overlap_score(item: dict, query_tokens: set[str]) -> float:
    haystack = " ".join(
        [
            item.get("title") or "",
            item.get("category") or "",
            item.get("source_url") or "",
            (item.get("text") or "")[:600],
        ]
    ).lower()
    return 0.006 * sum(1 for token in query_tokens if token in haystack)


def _entity_match_score(item: dict, query_tokens: set[str]) -> float:
    focus_tokens = _focus_query_tokens(query_tokens)
    if not focus_tokens:
        return 0.0

    haystack = " ".join(
        [
            item.get("title") or "",
            item.get("source_url") or "",
            (item.get("text") or "")[:800],
        ]
    ).lower()
    overlap = sum(1 for token in focus_tokens if token in haystack)
    if overlap == 0:
        return 0.0
    return min(0.04, 0.012 * overlap)


def _contact_signal_score(item: dict) -> float:
    text = " ".join(
        [
            item.get("title") or "",
            item.get("source_url") or "",
            (item.get("text") or "")[:1200],
        ]
    )
    score = 0.0
    if EMAIL_RE.search(text):
        score += 0.035
    if PHONE_RE.search(text):
        score += 0.025
    score += min(0.03, 0.012 * float(item.get("contact_field_count", 0)))
    return score


def _staff_query_score(
    item: dict,
    query_tokens: set[str],
    *,
    staff_intent: bool,
    contact_intent: bool,
    count_intent: bool,
) -> float:
    if not staff_intent:
        return 0.0

    score = 0.0
    category = (item.get("category") or "").lower()
    text = (item.get("text") or "").lower()
    url = (item.get("source_url") or "").lower()
    title = (item.get("title") or "").lower()

    if category in {"contact", "faculty", "departments", "about"}:
        score += 0.045
    if any(hint in url for hint in STAFF_URL_HINTS):
        score += 0.045
    if any(hint in title or hint in text for hint in STAFF_TEXT_HINTS):
        score += 0.03

    if contact_intent:
        score += _contact_signal_score(item)
        if any(hint in url for hint in LOW_VALUE_STAFF_HINTS) and not EMAIL_RE.search(text):
            score -= 0.055
        if category in {"results", "admissions", "recruitment", "tenders"} and not EMAIL_RE.search(text):
            score -= 0.03

    if count_intent:
        if item.get("has_table"):
            score += 0.035
        if float(item.get("table_row_count", 0)) >= 2:
            score += 0.02
        if any(hint in text for hint in COUNT_TEXT_HINTS):
            score += 0.03

    if any(hint in text or hint in url for hint in LOW_VALUE_STAFF_HINTS) and not (
        EMAIL_RE.search(text) or PHONE_RE.search(text)
    ):
        score -= 0.03

    if query_tokens & {"teacher", "teachers", "faculty"} and category == "departments":
        score += 0.015

    return score


def _heuristic_score(item: dict, query_tokens: set[str], intent_categories: set[str]) -> float:
    score = 0.0
    category = (item.get("category") or "").lower()
    text = (item.get("text") or "").lower()
    url = (item.get("source_url") or "").lower()
    staff_intent = _is_staff_intent(query_tokens)
    contact_intent = _is_contact_intent(query_tokens)
    count_intent = _is_count_intent(query_tokens)

    score += _metadata_overlap_score(item, query_tokens)
    score += _entity_match_score(item, query_tokens)

    if intent_categories:
        if category in intent_categories:
            score += 0.02
        elif category == "general":
            score -= 0.01

    if any(hint in text for hint in NOISY_TEXT_HINTS):
        score -= 0.12

    if any(hint in text for hint in NOISY_SECTION_HINTS) and "link" not in query_tokens:
        score -= 0.025

    if any(hint in url for hint in DEPARTMENT_URL_HINTS) and not (query_tokens & INTENT_CATEGORY_HINTS["departments"]):
        score -= 0.03

    score += _staff_query_score(
        item,
        query_tokens,
        staff_intent=staff_intent,
        contact_intent=contact_intent,
        count_intent=count_intent,
    )

    return score


def _unpack(meta: dict) -> dict:
    links = []
    try:
        links = json.loads(meta.get("links", "[]"))
    except Exception:
        pass

    unpacked = dict(meta)
    unpacked["title"] = _display_title(unpacked.get("title"), unpacked.get("source_url"))
    unpacked["links"] = links
    unpacked["has_links"] = _coerce_bool(meta.get("has_links"))
    unpacked["ocr"] = _coerce_bool(meta.get("ocr"))
    unpacked["chunk_index"] = int(meta.get("chunk_index", 0))
    unpacked["chunk_total"] = int(meta.get("chunk_total", 0))
    unpacked["has_table"] = _coerce_bool(meta.get("has_table"))
    unpacked["table_row_count"] = int(meta.get("table_row_count", 0))
    unpacked["contact_field_count"] = int(meta.get("contact_field_count", 0))
    return unpacked


def _get_bm25():
    collection = get_collection(required=True)
    count = collection.count()
    if _BM25_CACHE["bm25"] is not None and _BM25_CACHE["count"] == count:
        return (
            _BM25_CACHE["bm25"],
            _BM25_CACHE["texts"],
            _BM25_CACHE["metadatas"],
            _BM25_CACHE["ids"],
        )

    snapshot = collection.get(include=["documents", "metadatas"])
    texts = snapshot.get("documents", [])
    metadatas = snapshot.get("metadatas", [])
    ids = snapshot.get("ids", [])
    tokenized = [_tokenize(text) for text in texts]
    bm25 = BM25Okapi(tokenized) if tokenized else None

    _BM25_CACHE.update(
        {
            "count": count,
            "bm25": bm25,
            "texts": texts,
            "metadatas": metadatas,
            "ids": ids,
        }
    )
    return bm25, texts, metadatas, ids


def _dense_search(query: str, k: int) -> list[dict]:
    collection = get_collection(required=True)
    embedding = _embedder().encode([query], normalize_embeddings=True).tolist()
    results = collection.query(
        query_embeddings=embedding,
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    rows = []
    ids = results.get("ids", [[]])[0]
    for chunk_id, text, meta, distance in zip(
        ids,
        results.get("documents", [[]])[0],
        results.get("metadatas", [[]])[0],
        results.get("distances", [[]])[0],
    ):
        row = {
            "chunk_id": chunk_id,
            "text": text,
            "dense_score": max(0.0, 1 - float(distance)),
            **_unpack(meta),
        }
        rows.append(row)
    return rows


def _bm25_search(query: str, k: int) -> list[dict]:
    bm25, texts, metadatas, ids = _get_bm25()
    if bm25 is None:
        return []

    scores = bm25.get_scores(_tokenize(query))
    top_indices = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[:k]

    rows = []
    for index in top_indices:
        if scores[index] <= 0:
            continue
        row = {
            "chunk_id": ids[index],
            "text": texts[index],
            "bm25_score": float(scores[index]),
            **_unpack(metadatas[index]),
        }
        rows.append(row)
    return rows


def hybrid_retrieve(query: str, k: int | None = None) -> list[dict]:
    settings = get_settings()
    limit = k or settings.retrieval_k
    candidate_limit = max(limit * 4, 20)
    if not query.strip():
        return []

    try:
        dense = _dense_search(query, candidate_limit)
        sparse = _bm25_search(query, candidate_limit)
    except KnowledgeBaseNotReadyError:
        return []

    merged = {}
    link_intent = _is_link_intent(query)
    query_tokens = set(_significant_query_tokens(query))
    intent_categories = _intent_categories(query_tokens)
    for source_name, results in (("dense", dense), ("bm25", sparse)):
        for rank, item in enumerate(results, start=1):
            chunk_id = item["chunk_id"]
            current = merged.get(chunk_id)
            if current is None:
                current = {
                    **item,
                    "rrf_score": 0.0,
                    "dense_score": item.get("dense_score", 0.0),
                    "bm25_score": item.get("bm25_score", 0.0),
                }
                merged[chunk_id] = current
            else:
                current["dense_score"] = max(current.get("dense_score", 0.0), item.get("dense_score", 0.0))
                current["bm25_score"] = max(current.get("bm25_score", 0.0), item.get("bm25_score", 0.0))

            current["rrf_score"] += 1.0 / (RRF_K + rank)
            current["matched_by"] = sorted(set(current.get("matched_by", [])) | {source_name})

    ranked = list(merged.values())
    for item in ranked:
        if link_intent and item.get("has_links"):
            item["rrf_score"] += 0.02
        item["heuristic_score"] = _heuristic_score(item, query_tokens, intent_categories)
        item["final_score"] = item["rrf_score"] + item["heuristic_score"]

    deduped = {}
    for item in ranked:
        key = _source_key(item)
        current = deduped.get(key)
        if current is None:
            deduped[key] = item
            continue
        if (
            item["final_score"] > current["final_score"]
            or (
                item["final_score"] == current["final_score"]
                and item.get("chunk_index", 0) < current.get("chunk_index", 0)
            )
        ):
            deduped[key] = item

    ranked = list(deduped.values())
    ranked.sort(
        key=lambda item: (
            item.get("final_score", item["rrf_score"]),
            item.get("dense_score", 0.0),
            item.get("bm25_score", 0.0),
        ),
        reverse=True,
    )
    return ranked[:limit]


def retrieve_links(query: str, k: int = 10) -> list[str]:
    query_words = _tokenize(query)
    allow_external = "external" in query_words
    allow_utility = bool({"screenreader", "accessibility", "accessible"} & set(query_words))
    seen = set()
    urls = []
    for result in hybrid_retrieve(query, k=k):
        for link in result.get("links", []):
            url = link.get("url") if isinstance(link, dict) else link
            if not allow_external and url and not _is_university_url(url):
                continue
            if not allow_utility and url and _is_utility_url(url):
                continue
            anchor_text = link.get("anchor_text", "") if isinstance(link, dict) else str(link)
            categories = " ".join(link.get("categories", [])) if isinstance(link, dict) else ""
            haystack = f"{anchor_text} {url} {categories}".lower()
            if query_words and not any(word in haystack for word in query_words):
                continue
            if not url or url in seen:
                continue
            seen.add(url)
            urls.append(url)
        source_url = result.get("source_url")
        source_haystack = f"{result.get('title', '')} {source_url or ''} {result.get('category', '')}".lower()
        if query_words and not any(word in source_haystack for word in query_words):
            continue
        if source_url and not allow_external and not _is_university_url(source_url):
            continue
        if source_url and not allow_utility and _is_utility_url(source_url):
            continue
        if source_url and source_url not in seen:
            seen.add(source_url)
            urls.append(source_url)
    return urls[:k]


if __name__ == "__main__":
    query = "admission process central university kashmir"
    results = hybrid_retrieve(query)
    print(f"\nTop {min(3, len(results))} results for '{query}':")
    for index, result in enumerate(results[:3], start=1):
        print(f"\n[{index}] {result.get('title', 'Untitled')} | score={result['rrf_score']:.4f}")
        print(f"     {result['text'][:180]}...")
        if result.get("source_url"):
            print(f"     URL: {result['source_url']}")
