import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.settings import get_settings
from rag.text_cleanup import clean_text_artifacts


STYLE_GUIDANCE = {
    "concise": (
        "- Keep the answer compact: 2-4 polished bullets after the direct answer.\n"
        "- Include only the most decision-useful facts and one next step when supported.\n"
    ),
    "detailed": (
        "- Give a short executive summary first, then organize the answer into clear sections.\n"
        "- Include practical next steps, eligibility, dates, documents, and official links only when the context supports them.\n"
    ),
    "balanced": (
        "- Start with a one-sentence answer, then add 3-6 well-grouped bullets.\n"
        "- Use bullets for steps, requirements, dates, documents, links, or office contacts when the context contains them.\n"
    ),
}
EXACT_LOOKUP_STARTS = ("who is", "who's", "what is", "what's")
PRESENTATION_RULES = (
    "- Write in a professional student-service tone suitable for an official presentation.\n"
    "- Do not output a single dense paragraph. Use Markdown with short sections and clean bullets.\n"
    "- When presenting repeated structured records, use a Markdown table instead of prose. This includes course structures, semester-wise papers, notices with dates, contact lists, fees, eligibility rows, result lists, timetables, and office/staff details.\n"
    "- For course-structure answers, use table columns such as Semester, S. No., Course Code, Course Title, Credits, CIA, and ESE when those fields are present. Keep one course per row.\n"
    "- For contact answers, use table columns such as Name, Designation, Department/Office, Email, Phone, and Source when those fields are present.\n"
    "- Never leave a heading on the same line as body text. Example: use `**Eligibility**` on its own line, then bullets below it.\n"
    "- Avoid filler phrases such as \"as follows\" unless a complete list follows.\n"
    "- If the retrieved context is a notice excerpt and does not contain the full process, give the confirmed facts and point to the official notice/source.\n"
    "- Do not add sections titled \"Missing Information\", \"Not Provided\", or similar. Avoid listing facts that are absent from the context unless the student asks what is unavailable.\n"
    "- Keep citations at the end of the relevant sentence or bullet, not after every phrase.\n"
    "- CRITICAL: If you cannot find the requested person's specific contact information, clearly state it is not available. DO NOT guess or substitute it with another person's contact information.\n"
)


def _exact_lookup_guidance(query: str) -> str:
    normalized = (query or "").strip().lower()
    if not normalized.startswith(EXACT_LOOKUP_STARTS):
        return ""

    return (
        "- For exact person lookup questions, answer only the person's confirmed identity, role, department, and direct contact details.\n"
        "- Do not include incidental event, course, workshop, or newsletter mentions unless the student specifically asks for background or achievements.\n"
        "- If sources disagree or only give partial details, say what is confirmed instead of blending unrelated snippets.\n"
        "- Do not list missing contact details, research interests, or expertise unless the question specifically asks for those fields.\n"
        "- CRITICAL: Never attribute an email address, phone number, or role to a person unless the context explicitly links them together.\n"
    )


def _truncate_answer(text: str, limit: int = 300) -> str:
    """Shorten a previous assistant answer to its essential gist."""
    compact = " ".join((text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


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
    lookup_guidance = _exact_lookup_guidance(query)

    context_parts = []
    total_chars = 0
    for index, chunk in enumerate(chunks, start=1):
        title = clean_text_artifacts(chunk.get("title", "Untitled"))
        content = clean_text_artifacts(chunk.get("text", ""))
        block = (
            f"[{index}]\n"
            f"Title: {title or 'Untitled'}\n"
            f"URL: {chunk.get('source_url') or 'N/A'}\n"
            f"Category: {chunk.get('category', 'general')}\n"
            f"Content:\n{content}"
        )
        if total_chars + len(block) > settings.max_context_chars:
            break
        context_parts.append(block)
        total_chars += len(block)

    history_text = ""
    if history:
        turns = history[-5:]
        lines = []
        for turn in turns:
            user_text = turn.get("user") or turn.get("question") or turn.get("human") or ""
            bot_text = turn.get("bot") or turn.get("assistant") or turn.get("answer") or ""
            if not user_text and not bot_text:
                if turn.get("role") == "user":
                    user_text = turn.get("content") or ""
                elif turn.get("role") == "assistant":
                    bot_text = turn.get("content") or ""
            if user_text:
                lines.append(f"Student: {user_text}")
            if bot_text:
                lines.append(f"Assistant: {_truncate_answer(bot_text)}")
        history_text = (
            "\nRecent conversation (use this to resolve pronouns, follow-ups, "
            "and implicit references in the student's question):\n"
            + "\n".join(lines)
            + "\n"
        )

    context = "\n\n".join(context_parts)
    return f"""You are the official AI assistant for the Central University of Kashmir.
Answer only from the supplied context.

Rules:
- If the answer is supported by the context, answer clearly and cite sources like [1] or [2].
- If the answer is only partially supported, answer with the confirmed facts only and suggest checking the cited official source for more detail.
- If the answer is not in the context, politely state that you don't have that specific information, and immediately offer to tell them about one or more related university topics that are present in the retrieved Context instead.
- Prefer exact facts, dates, eligibility rules, and links when they exist in the context.
- Do not invent contact details, deadlines, fees, or policies.
- For staff, teacher, faculty, or office-contact questions, prefer official role/designation and direct contact fields from the context.
- When the context contains table rows or row-like records, keep values matched to the correct row and do not mix cells from different rows.
- For count questions, only count what is explicitly listed or stated in the context, and say when the total is incomplete.
- Make the answer easy to scan. Use short sections, bullets, or numbered steps when they improve clarity.
- Cite grounded claims, but keep citations tidy.
- Prefer one citation block at the end of a sentence, bullet, or short paragraph instead of repeating the same citations after every line.
- Mention official URLs from the context when they directly help the student act on the answer.
- When the student's question refers to something from the recent conversation (e.g. "his email", "tell me more", "that department"), use the conversation history below to resolve the reference and answer about the correct subject.
{PRESENTATION_RULES}
{lookup_guidance}
{style_guidance}
Context:
{context}
{history_text}
Question: {query}

Answer in polished Markdown with citations:"""
