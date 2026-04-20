import html
import re
import sys
import threading
from collections import Counter
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from rag.pipeline import app_status, run_with_metadata, warmup_local_models


QUICK_PROMPTS = [
    "What is the admission process for 2026?",
    "Which courses are available at CUK?",
    "When is the CUET exam and where do I check notices?",
    "How do I apply for a PhD at CUK?",
    "Show me faculty or department contact details.",
    "What official documents do I need for admission?",
]

FOLLOW_UP_LIBRARY = {
    "admissions": [
        "What documents are required for admission?",
        "What is the eligibility criteria?",
        "Where can I find the official admission notice?",
    ],
    "departments": [
        "Which department should I contact for this?",
        "Show me the relevant faculty or office details.",
        "What programmes are offered in this department?",
    ],
    "contact": [
        "Do you have the official email or phone number?",
        "Which office handles this process?",
        "Can you point me to the official contact page?",
    ],
    "examinations": [
        "Where can I check the latest exam notice?",
        "What dates are confirmed in the official notice?",
        "Is there an official page for exam updates?",
    ],
    "general": [
        "Can you summarize the most important points?",
        "Show me the official sources for this answer.",
        "What should I ask next to confirm the details?",
    ],
}

STYLE_MAP = {
    "Balanced": "balanced",
    "Detailed": "detailed",
    "Concise": "concise",
}
NO_INFO_MESSAGE = "I don't have that information. Please contact the university office directly."
CITATION_BLOCK_RE = re.compile(r"\s*(\[(?:\d+(?:,\s*\d+)*)\])\.?\s*$")


def _start_background_warmup() -> bool:
    threading.Thread(target=warmup_local_models, daemon=True).start()
    return True


def _init_state() -> None:
    if "history" not in st.session_state:
        st.session_state.history = []
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "answer_style" not in st.session_state:
        st.session_state.answer_style = "Balanced"
    if "source_limit" not in st.session_state:
        st.session_state.source_limit = 4
    if "show_debug" not in st.session_state:
        st.session_state.show_debug = False


def _reset_chat() -> None:
    st.session_state.history = []
    st.session_state.messages = []
    st.session_state.pop("pending_question", None)


def _queue_question(question: str) -> None:
    st.session_state.pending_question = question
    st.rerun()


def _top_category(sources: list[dict]) -> str:
    categories = [str(source.get("category", "general")).lower() for source in sources if source.get("category")]
    if not categories:
        return "general"
    return Counter(categories).most_common(1)[0][0]


def _build_follow_ups(query: str, sources: list[dict]) -> list[str]:
    lower_query = (query or "").lower()
    category = _top_category(sources)

    if any(word in lower_query for word in ("admission", "apply", "eligibility", "cuet")):
        category = "admissions"
    elif any(word in lower_query for word in ("faculty", "teacher", "professor", "contact", "email", "phone")):
        category = "contact"
    elif any(word in lower_query for word in ("exam", "examination", "datesheet", "result")):
        category = "examinations"
    elif any(word in lower_query for word in ("department", "programme", "course", "school")):
        category = "departments"

    suggestions = FOLLOW_UP_LIBRARY.get(category, FOLLOW_UP_LIBRARY["general"]) + FOLLOW_UP_LIBRARY["general"]
    deduped = []
    seen = set()
    for item in suggestions:
        key = item.strip().lower()
        if not key or key == lower_query or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
        if len(deduped) == 3:
            break
    return deduped


def _confidence_label(sources: list[dict], details: dict) -> str:
    selected = int(details.get("selected_chunk_count") or 0)
    top_score = None
    if sources:
        top_score = sources[0].get("rerank_score")

    if selected >= 3 and isinstance(top_score, (int, float)) and top_score >= 1.5:
        return "Strong grounding"
    if selected >= 2 and sources:
        return "Good grounding"
    if sources:
        return "Limited grounding"
    return "Low grounding"


def _format_collection_state(status: dict) -> str:
    if status["knowledge_base_ready"]:
        return f"Collection `{status['collection_name']}` is ready."
    return status["message"] or "Knowledge base is not ready yet."


def _normalize_answer(text: str) -> str:
    return " ".join((text or "").split()).strip().lower()


def _is_no_info_answer(text: str) -> bool:
    normalized = _normalize_answer(text)
    target = _normalize_answer(NO_INFO_MESSAGE)
    return normalized == target or normalized.startswith(target + " ")


def _strip_trailing_citation(line: str) -> tuple[str, str | None]:
    match = CITATION_BLOCK_RE.search(line)
    if not match:
        return line, None
    cleaned = line[: match.start()].rstrip()
    if cleaned.endswith(" ."):
        cleaned = cleaned[:-2] + "."
    return cleaned, match.group(1)


def _tidy_answer_citations(text: str) -> str:
    lines = text.splitlines()
    tidied = []
    pending_group: list[tuple[str, str]] = []

    def flush_group() -> None:
        nonlocal pending_group
        if not pending_group:
            return

        citations = [citation for _, citation in pending_group]
        same_citation = len(set(citations)) == 1 and len(pending_group) >= 2
        if same_citation:
            for cleaned, _ in pending_group:
                tidied.append(cleaned)
            tidied.append(f"Sources: {citations[0]}")
        else:
            for cleaned, citation in pending_group:
                tidied.append(f"{cleaned} {citation}" if citation else cleaned)
        pending_group = []

    for line in lines:
        if re.match(r"^\s*[-*]\s+", line):
            cleaned, citation = _strip_trailing_citation(line)
            if citation:
                pending_group.append((cleaned, citation))
                continue
        flush_group()
        tidied.append(line)

    flush_group()
    return "\n".join(tidied)


def _inject_styles() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@500;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --bg: #f6efe7;
    --bg-soft: #fffaf4;
    --surface: rgba(255, 251, 245, 0.9);
    --surface-strong: #fffdf9;
    --border: rgba(113, 88, 63, 0.16);
    --border-strong: rgba(113, 88, 63, 0.28);
    --ink: #1f2d3d;
    --muted: #5e6f7f;
    --accent: #d46f4d;
    --accent-soft: #f6d4c7;
    --blue: #2f6ea5;
    --blue-soft: #d9e8f5;
    --gold: #c48a3a;
    --shadow: 0 18px 45px rgba(58, 43, 26, 0.08);
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background:
        radial-gradient(circle at top left, rgba(212, 111, 77, 0.13), transparent 30%),
        radial-gradient(circle at top right, rgba(47, 110, 165, 0.14), transparent 28%),
        linear-gradient(180deg, #fbf4ec 0%, #f6efe7 38%, #f9f3ea 100%);
    color: var(--ink);
    font-family: 'Space Grotesk', sans-serif;
}

[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(255,255,255,0.16) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.16) 1px, transparent 1px);
    background-size: 64px 64px;
    opacity: 0.35;
    pointer-events: none;
}

.block-container {
    max-width: 1180px !important;
    padding-top: 2rem !important;
    padding-bottom: 5rem !important;
}

[data-testid="stSidebar"] {
    background: rgba(255, 248, 240, 0.88);
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label {
    color: var(--ink) !important;
}

.hero-card {
    position: relative;
    overflow: hidden;
    padding: 2rem 2.1rem;
    border-radius: 28px;
    background:
        linear-gradient(135deg, rgba(255,255,255,0.86), rgba(255,246,235,0.92)),
        linear-gradient(135deg, rgba(47,110,165,0.08), rgba(212,111,77,0.06));
    border: 1px solid rgba(113, 88, 63, 0.18);
    box-shadow: var(--shadow);
    margin-bottom: 1.25rem;
}

.hero-card::after {
    content: "";
    position: absolute;
    width: 240px;
    height: 240px;
    top: -90px;
    right: -70px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(47, 110, 165, 0.16), transparent 68%);
}

.eyebrow {
    display: inline-flex;
    gap: 0.5rem;
    align-items: center;
    padding: 0.35rem 0.7rem;
    background: rgba(47, 110, 165, 0.09);
    color: var(--blue);
    border: 1px solid rgba(47, 110, 165, 0.15);
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

.hero-title {
    margin: 0.9rem 0 0.55rem;
    font-family: 'Fraunces', serif;
    font-size: clamp(2.2rem, 4vw, 3.55rem);
    line-height: 1.02;
    color: #172534;
}

.hero-copy {
    max-width: 760px;
    margin: 0;
    color: var(--muted);
    font-size: 1rem;
    line-height: 1.7;
}

.hero-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
    margin-top: 1.2rem;
}

.hero-pill {
    border-radius: 999px;
    border: 1px solid var(--border);
    background: rgba(255,255,255,0.75);
    color: var(--ink);
    padding: 0.45rem 0.8rem;
    font-size: 0.84rem;
}

.section-title {
    margin: 1.4rem 0 0.8rem;
    color: #25384b;
    font-size: 0.9rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.stat-card {
    padding: 1rem 1rem 0.9rem;
    border-radius: 22px;
    border: 1px solid rgba(113, 88, 63, 0.14);
    background: rgba(255,255,255,0.7);
    box-shadow: 0 12px 28px rgba(58, 43, 26, 0.05);
}

.stat-label {
    color: var(--muted);
    font-size: 0.76rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.stat-value {
    margin-top: 0.35rem;
    font-size: 1.3rem;
    font-weight: 700;
    color: #1a2938;
}

.stat-sub {
    margin-top: 0.25rem;
    color: var(--muted);
    font-size: 0.84rem;
}

.sidebar-card {
    padding: 1rem 1rem 0.7rem;
    border: 1px solid var(--border);
    border-radius: 22px;
    background: rgba(255, 253, 249, 0.86);
    box-shadow: 0 14px 34px rgba(58, 43, 26, 0.06);
    margin-bottom: 1rem;
}

.sidebar-kicker {
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.76rem;
    font-weight: 700;
}

.sidebar-title {
    margin: 0.3rem 0 0.4rem;
    color: var(--ink);
    font-size: 1.15rem;
    font-weight: 700;
}

.sidebar-copy {
    color: var(--muted);
    font-size: 0.92rem;
    line-height: 1.55;
}

.source-card {
    border: 1px solid var(--border);
    background: rgba(255,255,255,0.86);
    border-radius: 18px;
    padding: 0.95rem 1rem;
    margin-bottom: 0.85rem;
    box-shadow: 0 10px 22px rgba(58, 43, 26, 0.05);
}

.source-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    align-items: center;
    margin-bottom: 0.4rem;
}

.source-citation,
.source-category,
.source-match {
    display: inline-flex;
    align-items: center;
    border-radius: 999px;
    padding: 0.24rem 0.65rem;
    font-size: 0.75rem;
    font-weight: 600;
}

.source-citation {
    background: rgba(212, 111, 77, 0.12);
    color: var(--accent);
}

.source-category {
    background: rgba(47, 110, 165, 0.12);
    color: var(--blue);
}

.source-match {
    background: rgba(196, 138, 58, 0.12);
    color: #8d6221;
}

.source-title {
    color: #1a2938;
    font-weight: 700;
    margin-bottom: 0.35rem;
}

.source-preview {
    color: var(--muted);
    line-height: 1.6;
    font-size: 0.92rem;
}

.source-link {
    display: inline-flex;
    margin-top: 0.75rem;
    color: var(--blue);
    font-weight: 700;
    text-decoration: none;
}

.source-path {
    margin-top: 0.75rem;
    color: var(--muted);
    font-size: 0.8rem;
    word-break: break-all;
}

.followup-label {
    margin-top: 0.8rem;
    margin-bottom: 0.35rem;
    color: var(--muted);
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 700;
}

.stButton > button,
[data-testid="stBaseButton-secondary"] {
    border-radius: 16px !important;
    border: 1px solid rgba(113, 88, 63, 0.16) !important;
    background: rgba(255, 251, 245, 0.88) !important;
    color: var(--ink) !important;
    min-height: 3rem !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease !important;
    box-shadow: 0 10px 24px rgba(58, 43, 26, 0.05) !important;
}

.stButton > button:hover,
[data-testid="stBaseButton-secondary"]:hover {
    transform: translateY(-1px);
    border-color: rgba(47, 110, 165, 0.24) !important;
    box-shadow: 0 14px 28px rgba(58, 43, 26, 0.08) !important;
}

[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.3rem 0 !important;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) .stChatMessageContent {
    background: linear-gradient(135deg, rgba(47,110,165,0.12), rgba(47,110,165,0.06));
    border: 1px solid rgba(47, 110, 165, 0.16);
    border-radius: 22px 22px 8px 22px;
    padding: 1rem 1.1rem;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) .stChatMessageContent {
    background: rgba(255, 253, 250, 0.9);
    border: 1px solid rgba(113, 88, 63, 0.12);
    border-radius: 10px 22px 22px 22px;
    padding: 1rem 1.1rem;
    box-shadow: 0 12px 30px rgba(58, 43, 26, 0.05);
}

[data-testid="stChatInput"] {
    background: linear-gradient(180deg, rgba(246,239,231,0), rgba(246,239,231,0.92) 30%, rgba(246,239,231,1) 100%);
    padding-top: 1rem;
}

[data-testid="stChatInputTextArea"] {
    border-radius: 18px !important;
    border: 1px solid rgba(113, 88, 63, 0.18) !important;
    background: rgba(255, 251, 245, 0.9) !important;
    box-shadow: 0 14px 28px rgba(58, 43, 26, 0.05) !important;
}

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
    color: var(--ink);
    line-height: 1.7;
}

[data-testid="stMarkdownContainer"] a {
    color: var(--blue);
}

.stExpander {
    border-radius: 18px !important;
    border: 1px solid rgba(113, 88, 63, 0.12) !important;
    background: rgba(255, 251, 245, 0.74) !important;
}

footer, #MainMenu, header {
    display: none !important;
}
</style>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar(status: dict) -> None:
    with st.sidebar:
        st.markdown(
            """
<div class="sidebar-card">
    <div class="sidebar-kicker">Control Room</div>
    <div class="sidebar-title">Make the assistant feel sharper</div>
    <div class="sidebar-copy">
        Tune answer depth, evidence density, and diagnostics without changing the retrieval backbone.
    </div>
</div>
            """,
            unsafe_allow_html=True,
        )

        st.selectbox(
            "Answer style",
            options=["Balanced", "Detailed", "Concise"],
            key="answer_style",
            help="Balanced keeps answers clear, Detailed expands grounded context, and Concise stays short.",
        )
        st.slider(
            "Evidence cards",
            min_value=2,
            max_value=8,
            key="source_limit",
            help="Controls how many source cards are expanded under each answer.",
        )
        st.toggle(
            "Show retrieval diagnostics",
            key="show_debug",
            help="Shows rewritten query, retrieval counts, and pipeline timings for each answer.",
        )
        if st.button("Clear conversation", use_container_width=True):
            _reset_chat()
            st.rerun()


def _render_source_cards(sources: list[dict], limit: int) -> None:
    for source in sources[:limit]:
        citation = source.get("citation", "?")
        title = html.escape(str(source.get("label") or "Untitled"))
        category = html.escape(str(source.get("category") or "general").title())
        preview = html.escape(str(source.get("preview") or "No preview available."))
        matched_by = source.get("matched_by") or []
        matched_text = " + ".join(part.upper() for part in matched_by) if matched_by else "Grounded match"
        url = source.get("url")
        path = source.get("path")

        st.markdown(
            f"""
<div class="source-card">
    <div class="source-meta">
        <span class="source-citation">[{citation}]</span>
        <span class="source-category">{category}</span>
        <span class="source-match">{html.escape(matched_text)}</span>
    </div>
    <div class="source-title">{title}</div>
    <div class="source-preview">{preview}</div>
    {
        f'<a class="source-link" href="{html.escape(str(url), quote=True)}" target="_blank">Open official source</a>'
        if url
        else f'<div class="source-path">{html.escape(str(path or "Local source only"))}</div>'
    }
</div>
            """,
            unsafe_allow_html=True,
        )


def _render_answer_notes(details: dict, sources: list[dict]) -> None:
    answer_state = details.get("answer_state", "answered")
    confidence = _confidence_label(sources, details)
    rewritten_query = details.get("rewritten_query") or details.get("query") or ""
    original_query = details.get("query") or ""

    if answer_state == "abstained":
        st.markdown("**Support status:** No grounded answer found.")
        if details.get("suppressed_source_count"):
            st.caption(
                "Some related documents were retrieved, but they were too weak or off-target to present as evidence."
            )
    else:
        st.markdown(f"**Grounding strength:** {confidence}")

    if rewritten_query and rewritten_query != original_query:
        st.markdown(f"**Interpreted question:** `{rewritten_query}`")

    col1, col2, col3 = st.columns(3)
    col1.metric("Candidates", int(details.get("candidate_count") or 0))
    col2.metric("Reranked", int(details.get("rerank_input_count") or 0))
    col3.metric("Used in answer", int(details.get("selected_chunk_count") or 0))

    if st.session_state.show_debug:
        timings = details.get("timings_ms") or {}
        t1, t2, t3, t4 = st.columns(4)
        t1.metric("Rewrite", f"{timings.get('rewrite', 0):.1f} ms")
        t2.metric("Retrieve", f"{timings.get('retrieve', 0):.1f} ms")
        t3.metric("Rerank", f"{timings.get('rerank', 0):.1f} ms")
        t4.metric("Total", f"{timings.get('total', 0):.1f} ms")

        if sources:
            category_mix = ", ".join(
                f"{name}: {count}"
                for name, count in Counter(
                    str(source.get("category") or "general").title() for source in sources
                ).items()
            )
            st.caption(f"Source mix: {category_mix}")


def _render_assistant_panels(message: dict, message_index: int) -> None:
    sources = message.get("sources") or []
    details = message.get("details") or {}
    follow_ups = message.get("follow_ups") or []
    answer_state = details.get("answer_state", "answered")

    if sources:
        with st.expander(f"Evidence ({min(len(sources), st.session_state.source_limit)} shown)", expanded=False):
            _render_source_cards(sources, st.session_state.source_limit)

    show_notes = bool(sources) or st.session_state.show_debug
    if answer_state == "abstained" and not st.session_state.show_debug:
        show_notes = False

    if show_notes:
        with st.expander("Answer notes", expanded=False):
            _render_answer_notes(details, sources)

    if follow_ups and answer_state != "abstained":
        st.markdown('<div class="followup-label">Continue with</div>', unsafe_allow_html=True)
        columns = st.columns(len(follow_ups))
        for index, follow_up in enumerate(follow_ups):
            if columns[index].button(
                follow_up,
                key=f"follow_up_{message_index}_{index}",
                use_container_width=True,
            ):
                _queue_question(follow_up)


st.set_page_config(
    page_title="CUK AI Assistant",
    page_icon=":mortar_board:",
    layout="wide",
)

_init_state()
_inject_styles()

if not st.session_state.get("_warmup_started"):
    _start_background_warmup()
    st.session_state._warmup_started = True

status = app_status()
_render_sidebar(status)

st.markdown(
    """
<div class="hero-card">
    <div class="eyebrow">Interactive university support</div>
    <div class="hero-title">Central University of Kashmir Assistant</div>
    <p class="hero-copy">
        Ask about admissions, examinations, departments, faculty, notices, and official university documents.
        The interface is now lighter, source-aware, and built to make each answer easier to inspect and continue.
    </p>
    <div class="hero-pills">
        <span class="hero-pill">Grounded answers</span>
        <span class="hero-pill">Interactive evidence cards</span>
        <span class="hero-pill">Follow-up prompts</span>
        <span class="hero-pill">Adjustable answer depth</span>
    </div>
</div>
    """,
    unsafe_allow_html=True,
)

if not st.session_state.messages:
    st.markdown('<div class="section-title">Start with a quick question</div>', unsafe_allow_html=True)
    prompt_columns = st.columns(3)
    for index, question in enumerate(QUICK_PROMPTS):
        if prompt_columns[index % 3].button(question, key=f"quick_prompt_{index}", use_container_width=True):
            _queue_question(question)

for index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            _render_assistant_panels(message, index)

query = st.chat_input("Ask about admissions, faculty, notices, results, dates, or official documents")
if "pending_question" in st.session_state:
    query = st.session_state.pending_question
    del st.session_state.pending_question

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    answer_style = STYLE_MAP.get(st.session_state.answer_style, "balanced")
    assistant_message_index = len(st.session_state.messages)

    with st.chat_message("assistant"):
        with st.spinner("Searching official university material..."):
            stream, sources, details = run_with_metadata(
                query,
                st.session_state.history,
                answer_style=answer_style,
            )

        full_answer = ""
        placeholder = st.empty()
        for chunk in stream:
            text = getattr(chunk, "text", "")
            if text:
                full_answer += text
                placeholder.markdown(full_answer + "|")
        full_answer = _tidy_answer_citations(full_answer)
        placeholder.markdown(full_answer)

        abstained = _is_no_info_answer(full_answer)
        visible_sources = sources
        visible_follow_ups = _build_follow_ups(query, sources)
        if abstained:
            details = {
                **details,
                "answer_state": "abstained",
                "suppressed_source_count": len(sources),
            }
            visible_sources = []
            visible_follow_ups = []
        else:
            details = {
                **details,
                "answer_state": "answered",
            }

        message_payload = {
            "role": "assistant",
            "content": full_answer,
            "sources": visible_sources,
            "details": details,
            "follow_ups": visible_follow_ups,
        }
        _render_assistant_panels(message_payload, assistant_message_index)

    st.session_state.history.append({"user": query, "bot": full_answer})
    st.session_state.messages.append(message_payload)
