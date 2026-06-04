import logging
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.llm import generate_text

log = logging.getLogger(__name__)

ROLE_KEYS = {
    "assistant": ("assistant", "bot", "answer"),
    "user": ("user", "human", "question"),
}

def _message_text(turn: dict, role: str) -> str:
    for key in ROLE_KEYS[role]:
        value = turn.get(key)
        if value:
            return str(value).strip()
    if str(turn.get("role", "")).lower() == role and turn.get("content"):
        return str(turn["content"]).strip()
    return ""

def _format_history_for_condense(history: list[dict], max_turns: int) -> str:
    lines = []
    for turn in history[-max_turns:]:
        user_text = _message_text(turn, "user")
        assistant_text = _message_text(turn, "assistant")
        if user_text:
            lines.append(f"User: {user_text}")
        if assistant_text:
            lines.append(f"Assistant: {assistant_text}")
    return "\n".join(lines)


def condense_question(
    chat_history: list[dict],
    new_user_query: str,
    *,
    max_history_turns: int = 4,
) -> str:
    """Rewrite a follow-up into a standalone retrieval query using the LLM.
    
    This replaces brittle regex heuristics with an intelligent contextualizer.
    """
    query = (new_user_query or "").strip()
    if not query or not chat_history:
        return query

    turns = _format_history_for_condense(chat_history, max_history_turns)
    if not turns:
        return query

    prompt = f"""You are a smart context-resolver for a university search engine.
Read the recent conversation, then look at the user's latest follow-up question.

Conversation History:
{turns}

Follow-up Question:
{query}

Instructions:
1. If the Follow-up Question refers to a topic in the Conversation History (using pronouns like he/she/it, or implicitly asking for more details about a discussed topic like "what is the fee?"), rewrite it into a complete, standalone search query containing the specific topic, name, department, or notice.
2. If the Follow-up Question is completely new and unrelated to the Conversation History (e.g., asking about a new department or person not mentioned before), DO NOT add previous context. Output the Follow-up Question EXACTLY AS IS.
3. Do not answer the question. Only output the standalone query.
4. Do not wrap the output in quotes.

Standalone Query:"""

    try:
        rewritten = str(generate_text(prompt, stream=False) or "").strip()
        # Clean any accidental quotes around the output
        if rewritten.startswith('"') and rewritten.endswith('"'):
            rewritten = rewritten[1:-1].strip()
        if rewritten.startswith("'") and rewritten.endswith("'"):
            rewritten = rewritten[1:-1].strip()
        return rewritten or query
    except Exception as exc:
        log.warning("Question condensation failed, using original query: %s", exc)
        return query


def rewrite_query(query: str, history: list[dict]) -> str:
    """Backward-compatible wrapper around ``condense_question``."""
    return condense_question(history, query)
