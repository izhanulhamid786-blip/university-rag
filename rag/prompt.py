import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.settings import get_settings


STYLE_GUIDANCE = {
    "concise": (
        "- Keep the answer compact: a direct answer first, then only the most important supporting points.\n"
        "- Use bullets only when they make the answer easier to scan.\n"
    ),
    "detailed": (
        "- Give a short summary first, then organize the answer into clear sections or bullets when helpful.\n"
        "- Include practical next steps, eligibility, dates, or documents only when the context supports them.\n"
    ),
    "balanced": (
        "- Start with a direct answer, then add short, well-grouped details that are easy to scan.\n"
        "- Use bullets for steps, requirements, dates, or lists when the context contains them.\n"
    ),
}


def build_prompt(
    query: str,
    chunks: list[dict],
    history: list[dict],
    *,
    answer_style: str = "balanced",
) -> str:
    settings = get_settings()
    style_key = (answer_style or "balanced").strip().lower()
    style_guidance = STYLE_GUIDANCE.get(style_key, STYLE_GUIDANCE["balanced"])

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
- Make the answer easy to scan. Use short sections, bullets, or numbered steps when they improve clarity.
- Cite grounded claims, but keep citations tidy.
- Prefer one citation block at the end of a sentence, bullet, or short paragraph instead of repeating the same citations after every line.
- Mention official URLs from the context when they directly help the student act on the answer.
{style_guidance}
{history_text}
Context:
{context}

Question: {query}

Answer with citations:"""
