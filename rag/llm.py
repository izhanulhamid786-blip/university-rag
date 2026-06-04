import json
import logging
import os
import sys
from pathlib import Path

import requests
from requests import exceptions as requests_exceptions

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.settings import get_settings


log = logging.getLogger(__name__)


class GeneratorConnectionError(RuntimeError):
    def __init__(self, message: str, *, error_name: str = "GeneratorError", quota_exhausted: bool = False):
        super().__init__(message)
        self.error_name = error_name
        self.quota_exhausted = quota_exhausted


GroqConnectionError = GeneratorConnectionError


def _is_quota_error(status_code: int | None, message: str) -> bool:
    haystack = f"{status_code or ''} {message}".lower()
    return any(
        marker in haystack
        for marker in (
            "quota",
            "rate limit",
            "rate_limit",
            "too many requests",
            "429",
        )
    )


def _error_name(status_code: int | None, message: str, fallback: str = "GeneratorError", provider: str = "Generator") -> str:
    if status_code == 401:
        return f"{provider}AuthenticationError"
    if status_code == 403:
        return f"{provider}PermissionError"
    if status_code == 404:
        return f"{provider}ModelNotFound"
    if status_code == 429 or _is_quota_error(status_code, message):
        return f"{provider}QuotaExceeded"
    return fallback


def groq_generate(prompt: str, *, stream: bool = True):
    settings = get_settings()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise GeneratorConnectionError(
            "Missing GROQ_API_KEY. Add it to your .env file.",
            error_name="GroqAuthenticationError",
        )

    url = f"{os.getenv('GROQ_BASE_URL', 'https://api.groq.com/openai/v1')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.generator_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "stream": stream,
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60, stream=stream)
        if response.status_code != 200:
            try:
                err_data = response.json()
                msg = err_data.get("error", {}).get("message", "")
            except:
                msg = response.text
            raise GeneratorConnectionError(
                f"Groq API returned error {response.status_code}: {msg}",
                error_name=_error_name(response.status_code, msg, provider="Groq"),
                quota_exhausted=_is_quota_error(response.status_code, msg),
            )
    except requests_exceptions.Timeout as exc:
        raise GeneratorConnectionError(
            f"Groq read timeout: {exc}",
            error_name="GroqReadTimeout",
        ) from exc
    except requests.RequestException as exc:
        raise GeneratorConnectionError(
            f"Groq request failed for model '{settings.generator_model}': {exc}",
            error_name=exc.__class__.__name__,
            quota_exhausted=_is_quota_error(None, str(exc)),
        ) from exc

    if not stream:
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            return ""
        return (choices[0].get("message") or {}).get("content", "") or ""

    def chunks():
        try:
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if data == "[DONE]":
                    break
                payload = json.loads(data)
                choices = payload.get("choices") or []
                if not choices:
                    continue
                text = (choices[0].get("delta") or {}).get("content", "")
                if text:
                    yield text
        except (requests.RequestException, json.JSONDecodeError) as exc:
            raise GeneratorConnectionError(
                f"Groq did not return a complete response for model '{settings.generator_model}'.",
                error_name=exc.__class__.__name__,
                quota_exhausted=_is_quota_error(None, str(exc)),
            ) from exc

    return chunks()


def generate_text(prompt: str, *, stream: bool = True):
    return groq_generate(prompt, stream=stream)
