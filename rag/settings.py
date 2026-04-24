import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
load_dotenv(ENV_FILE, override=True)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value.strip()
    return None


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    if not value:
        return default

    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


@dataclass(frozen=True)
class Settings:
    project_root: Path
    base_dir: Path
    env_file: Path
    app_file: Path
    crawler_file: Path
    rag_dir: Path
    ingest_dir: Path
    tests_dir: Path
    data_dir: Path
    structured_dir: Path
    manual_dir: Path
    pdf_dir: Path
    crawl_index_path: Path
    visited_urls_path: Path
    doc_hashes_path: Path
    vector_db_dir: Path
    collection_name: str
    embed_model: str
    rerank_model: str
    generator_model: str
    fallback_generator_model: str | None
    google_api_key: str | None
    retrieval_k: int
    rerank_top_k: int
    rerank_candidate_k: int
    max_context_chars: int
    include_manual_raw: bool
    include_legacy_root_raw: bool
    local_files_only: bool


def get_settings() -> Settings:
    data_dir = _env_path("RAG_DATA_DIR", PROJECT_ROOT / "data")
    return Settings(
        project_root=PROJECT_ROOT,
        base_dir=PROJECT_ROOT,
        env_file=ENV_FILE,
        app_file=PROJECT_ROOT / "app.py",
        crawler_file=PROJECT_ROOT / "crawler.py",
        rag_dir=PROJECT_ROOT / "rag",
        ingest_dir=PROJECT_ROOT / "ingest",
        tests_dir=PROJECT_ROOT / "tests",
        data_dir=data_dir,
        structured_dir=_env_path("RAG_STRUCTURED_DIR", data_dir / "structured"),
        manual_dir=_env_path("RAG_MANUAL_DIR", data_dir / "manual"),
        pdf_dir=_env_path("RAG_PDF_DIR", data_dir / "pdfs"),
        crawl_index_path=_env_path("RAG_CRAWL_INDEX_PATH", data_dir / "index.json"),
        visited_urls_path=_env_path("RAG_VISITED_URLS_PATH", data_dir / ".visited_urls.txt"),
        doc_hashes_path=_env_path("RAG_DOC_HASHES_PATH", data_dir / ".doc_hashes.json"),
        vector_db_dir=_env_path("RAG_VECTOR_DB_DIR", PROJECT_ROOT / "vector_db"),
        collection_name=os.getenv("RAG_COLLECTION", "university_kb"),
        embed_model=os.getenv("RAG_EMBED_MODEL", "BAAI/bge-base-en-v1.5"),
        rerank_model=os.getenv("RAG_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        generator_model=os.getenv("GENERATOR_MODEL", "gemini-2.5-flash"),
        fallback_generator_model=(
            _first_env("FALLBACK_GENERATOR_MODEL", "GENERATOR_FALLBACK_MODEL")
            or "gemini-2.5-flash-lite"
        ),
        google_api_key=_first_env("GOOGLE_API_KEY", "GEMINI_API_KEY"),
        retrieval_k=max(5, _env_int("RAG_RETRIEVAL_K", 20)),
        rerank_top_k=max(1, _env_int("RAG_RERANK_TOP_K", 5)),
        rerank_candidate_k=max(4, _env_int("RAG_RERANK_CANDIDATES", 12)),
        max_context_chars=max(4000, _env_int("RAG_MAX_CONTEXT_CHARS", 12000)),
        include_manual_raw=_env_bool("RAG_INCLUDE_MANUAL_RAW", True),
        include_legacy_root_raw=_env_bool("RAG_INCLUDE_LEGACY_ROOT_RAW", True),
        local_files_only=_env_bool("RAG_LOCAL_FILES_ONLY", False),
    )
