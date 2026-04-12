import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.settings import get_settings


def build_prompt(query: str, chunks: list[dict], history: list[dict]) -> str:
    settings = get_settings()

    context_parts = []
    total_chars = 0
    for index, chunk in enumerate(chunks, start=1):
        block = (
            f"[{index}]\n"
            f"Title: {chunk.get('title', 'Untitled')}\n"
            f"URL: {chunk.get('source_url') or 'N/A'}\n"
            f"Category: {chunk.get('category', 'general')}\n"
            f"Content:\n{chunk['text']}"
        )
        if total_chars + len(block) > settings.max_context_chars:
            break
        context_parts.append(block)
        total_chars += len(block)

    history_text = ""
    if history:
        turns = history[-3:]
        history_text = "\n".join(
            f"Student: {turn['user']}\nAssistant: {turn['bot']}" for turn in turns
        )
        history_text = f"\nRecent conversation:\n{history_text}\n"

    context = "\n\n".join(context_parts)
    return f"""You are the official AI assistant for the Central University of Kashmir.
Answer only from the supplied context.

Rules:
- If the answer is supported by the context, answer clearly and cite sources like [1] or [2].
- If the answer is only partially supported, say what is confirmed and what is missing.
- If the answer is not in the context, say exactly: "I don't have that information. Please contact the university office directly."
- Prefer exact facts, dates, eligibility rules, and links when they exist in the context.
- Do not invent contact details, deadlines, fees, or policies.
- For staff, teacher, faculty, or office-contact questions, prefer official role/designation and direct contact fields from the context.
- When the context contains table rows or row-like records, keep values matched to the correct row and do not mix cells from different rows.
- For count questions, only count what is explicitly listed or stated in the context, and say when the total is incomplete.
{history_text}
Context:
{context}

Question: {query}

Answer with citations:"""
