import logging
import os
import sys
from functools import lru_cache
from pathlib import Path

from google import genai

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.settings import get_settings


log = logging.getLogger(__name__)
BROKEN_LOCAL_PROXY_VALUES = {
    "http://127.0.0.1:9",
    "https://127.0.0.1:9",
}
PROXY_ENV_VARS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


class LLMConfigurationError(RuntimeError):
    pass


def _clear_broken_proxy_env() -> None:
    cleared = []
    for name in PROXY_ENV_VARS:
        value = (os.getenv(name) or "").strip()
        if value.lower() in BROKEN_LOCAL_PROXY_VALUES:
            os.environ.pop(name, None)
            cleared.append(name)
    if cleared:
        log.warning("Cleared broken proxy variables for Gemini requests: %s", ", ".join(sorted(cleared)))


@lru_cache(maxsize=2)
def _cached_genai_client(api_key: str):
    return genai.Client(api_key=api_key)


def get_genai_client(required: bool = False):
    settings = get_settings()
    if not settings.google_api_key:
        if required:
            raise LLMConfigurationError(
                "Missing GOOGLE_API_KEY or GEMINI_API_KEY. Add it to .env before starting the app."
            )
        return None

    try:
        _clear_broken_proxy_env()
        return _cached_genai_client(settings.google_api_key)
    except Exception as exc:
        log.exception("Failed to initialize Google GenAI client")
        if required:
            raise LLMConfigurationError(f"Could not initialize Google GenAI client: {exc}") from exc
        return None
