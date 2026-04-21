import logging
import re
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.llm import get_genai_client
from rag.settings import get_settings


log = logging.getLogger(__name__)
NON_PERSON_PHRASES = {
    "central university",
    "university office",
    "school of",
    "media studies",
    "department of",
    "associate professor",
    "assistant professor",
    "professor",
    "dean",
    "director",
    "coordinator",
    "controller of examinations",
    "finance officer",
    "registrar",
    "chief vigilance officer",
    "academic affairs",
}
PERSON_WITH_TITLE_RE = re.compile(
    r"\b(?:Prof\.?|Professor|Dr\.?|Mr\.?|Mrs\.?|Ms\.?)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b"
)
PERSON_NAME_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b")
REFERENCE_WORD_RE = re.compile(
    r"\b(he|she|his|her|hers|him|they|them|their|theirs)\b",
    re.IGNORECASE,
)
CONTEXTUAL_WORD_RE = re.compile(
    r"\b(it|its|this|that|these|those|there|same|former|latter|mentioned|above|below)\b",
    re.IGNORECASE,
)
FOLLOW_UP_PREFIX_RE = re.compile(r"^\s*(and|also|what about|how about|then|now)\b", re.IGNORECASE)


def _extract_recent_person(history: list[dict]) -> str | None:
    for turn in reversed(history[-4:]):
        for field in ("bot", "user"):
            text = (turn.get(field) or "").strip()
            if not text:
                continue

            titled = PERSON_WITH_TITLE_RE.findall(text)
            if titled:
                return titled[-1].strip()

            plain = [
                match.strip()
                for match in PERSON_NAME_RE.findall(text)
                if not any(
                    blocked in match.lower()
                    for blocked in NON_PERSON_PHRASES
                )
            ]
            if plain:
                return plain[-1]
    return None


def _local_rewrite(query: str, history: list[dict]) -> str:
    if not history or not REFERENCE_WORD_RE.search(query):
        return query

    person = _extract_recent_person(history)
    if not person:
        return query

    rewritten = query
    rewritten = re.sub(r"\bhis\b", f"{person}'s", rewritten, flags=re.IGNORECASE)
    rewritten = re.sub(r"\bher\b", f"{person}'s", rewritten, flags=re.IGNORECASE)
    rewritten = re.sub(r"\btheir\b", f"{person}'s", rewritten, flags=re.IGNORECASE)
    rewritten = re.sub(r"\bhim\b", person, rewritten, flags=re.IGNORECASE)
    rewritten = re.sub(r"\bthem\b", person, rewritten, flags=re.IGNORECASE)
    rewritten = re.sub(r"\bhe\b", person, rewritten, flags=re.IGNORECASE)
    rewritten = re.sub(r"\bshe\b", person, rewritten, flags=re.IGNORECASE)
    rewritten = re.sub(r"\bthey\b", person, rewritten, flags=re.IGNORECASE)
    return rewritten


def rewrite_query(query: str, history: list[dict]) -> str:
    if not history:
        return query

    local_rewrite = _local_rewrite(query, history)
    if REFERENCE_WORD_RE.search(query):
        return local_rewrite

    should_try_llm = bool(CONTEXTUAL_WORD_RE.search(query) or FOLLOW_UP_PREFIX_RE.search(query))
    if not should_try_llm:
        return local_rewrite

    client = get_genai_client(required=False)
    if client is None:
        return local_rewrite

    recent_turns = history[-4:]
    turns = "\n".join(f"User: {turn['user']}\nAssistant: {turn['bot']}" for turn in recent_turns)
    prompt = f"""Given this conversation:
{turns}

Rewrite the new question so it is fully self-contained and preserves the user's intent.
If the new question uses a reference like "he", "she", "his", "her", "they", or "their",
replace it with the exact person or subject mentioned most recently in the conversation.
Do not switch to a different person just because another person might match the topic.
Return only the rewritten question.

Question: {query}
"""

    try:
        response = client.models.generate_content(
            model=get_settings().generator_model,
            contents=prompt,
        )
        rewritten = (response.text or "").strip()
        return rewritten or local_rewrite
    except Exception as exc:
        log.warning("Query rewrite failed, using original query: %s", exc)
        return local_rewrite
