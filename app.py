import sys
import threading
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from rag.pipeline import run, warmup_local_models


def _start_background_warmup() -> bool:
    threading.Thread(target=warmup_local_models, daemon=True).start()
    return True

st.set_page_config(
    page_title="CUK AI Assistant",
    page_icon="🎓",
    layout="centered"
)

if not st.session_state.get("_warmup_started"):
    _start_background_warmup()
    st.session_state._warmup_started = True

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root & Reset ── */
:root {
    --midnight: #050d1a;
    --deep:     #091628;
    --navy:     #0d2240;
    --glass:    rgba(13, 34, 64, 0.55);
    --glass2:   rgba(255,255,255,0.04);
    --gold:     #c9a84c;
    --gold-lt:  #f0d080;
    --sky:      #4a9ede;
    --sky-lt:   #7ec8ff;
    --text:     #e8f0fa;
    --muted:    #7a9bc4;
    --border:   rgba(100,160,255,0.15);
    --border2:  rgba(201,168,76,0.3);
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background: var(--midnight) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--text);
}

/* ── Starfield background ── */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 80% 50% at 20% 0%,  rgba(74,158,222,0.12) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 100%, rgba(201,168,76,0.09) 0%, transparent 55%),
        radial-gradient(ellipse 40% 60% at 50% 50%,  rgba(9,22,40,0.8) 0%, transparent 100%);
    pointer-events: none;
    z-index: 0;
}

/* Subtle mountain silhouette */
[data-testid="stAppViewContainer"]::after {
    content: '';
    position: fixed;
    bottom: 0; left: 0; right: 0;
    height: 220px;
    background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1440 220'%3E%3Cpath fill='%230d1e35' fill-opacity='0.7' d='M0,180 L120,100 L240,150 L360,60 L480,120 L600,40 L720,110 L840,50 L960,130 L1080,70 L1200,140 L1320,80 L1440,160 L1440,220 L0,220 Z'/%3E%3Cpath fill='%230a1728' fill-opacity='0.9' d='M0,200 L180,140 L300,170 L420,100 L540,155 L660,80 L780,145 L900,90 L1020,160 L1140,110 L1260,170 L1440,120 L1440,220 L0,220 Z'/%3E%3C/svg%3E") bottom/cover no-repeat;
    pointer-events: none;
    z-index: 0;
}

/* Floating particles */
@keyframes float-up {
    0%   { transform: translateY(0) translateX(0) scale(1); opacity: 0; }
    10%  { opacity: 1; }
    90%  { opacity: 0.3; }
    100% { transform: translateY(-100vh) translateX(30px) scale(0.3); opacity: 0; }
}
.particle {
    position: fixed;
    border-radius: 50%;
    background: var(--gold);
    animation: float-up linear infinite;
    pointer-events: none;
    z-index: 1;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--deep); }
::-webkit-scrollbar-thumb { background: var(--gold); border-radius: 2px; }

/* ── Main block container ── */
.block-container {
    position: relative;
    z-index: 10;
    max-width: 820px !important;
    padding: 1.5rem 1.5rem 5rem !important;
}

/* ── Header card ── */
.hero-card {
    background: linear-gradient(135deg, rgba(13,34,64,0.9) 0%, rgba(5,13,26,0.95) 100%);
    border: 1px solid var(--border2);
    border-radius: 20px;
    padding: 2.2rem 2rem 1.8rem;
    text-align: center;
    margin-bottom: 1.6rem;
    position: relative;
    overflow: hidden;
    box-shadow:
        0 0 0 1px rgba(201,168,76,0.08),
        0 20px 60px rgba(0,0,0,0.5),
        0 4px 20px rgba(74,158,222,0.1),
        inset 0 1px 0 rgba(255,255,255,0.05);
    transform: perspective(1000px) rotateX(1deg);
    transform-origin: center top;
}
.hero-card::before {
    content: '';
    position: absolute;
    top: 0; left: 50%; transform: translateX(-50%);
    width: 60%; height: 2px;
    background: linear-gradient(90deg, transparent, var(--gold), transparent);
}
.hero-card::after {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(201,168,76,0.06) 0%, transparent 70%);
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--gold-lt) 0%, var(--gold) 50%, #a07830 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0.4rem 0 0.5rem;
    letter-spacing: 0.01em;
    line-height: 1.2;
}
.hero-sub {
    font-size: 0.82rem;
    color: var(--muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.hero-badge-row {
    display: flex;
    justify-content: center;
    gap: 8px;
    flex-wrap: wrap;
}
.hero-badge {
    background: rgba(74,158,222,0.1);
    border: 1px solid rgba(74,158,222,0.2);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.72rem;
    color: var(--sky-lt);
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

/* ── Suggested question buttons ── */
.stButton > button {
    background: var(--glass) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
    font-size: 0.82rem !important;
    font-family: 'DM Sans', sans-serif !important;
    padding: 0.55rem 0.9rem !important;
    transition: all 0.25s ease !important;
    text-align: left !important;
    backdrop-filter: blur(8px) !important;
    box-shadow: 0 4px 16px rgba(0,0,0,0.2), inset 0 1px 0 rgba(255,255,255,0.04) !important;
    transform: perspective(600px) translateZ(0) !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: rgba(74,158,222,0.12) !important;
    border-color: rgba(74,158,222,0.35) !important;
    color: var(--sky-lt) !important;
    transform: perspective(600px) translateZ(6px) translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(74,158,222,0.15), inset 0 1px 0 rgba(255,255,255,0.06) !important;
}
.stButton > button:active {
    transform: perspective(600px) translateZ(2px) translateY(0) !important;
}

/* ── Section label ── */
.section-label {
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.7rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.2rem 0 !important;
}

/* User bubble */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) .stChatMessageContent,
[data-testid="stChatMessage"][data-role="user"] .stMarkdown {
    background: linear-gradient(135deg, rgba(74,158,222,0.18), rgba(74,158,222,0.08)) !important;
    border: 1px solid rgba(74,158,222,0.25) !important;
    border-radius: 16px 16px 4px 16px !important;
    padding: 0.85rem 1.1rem !important;
    box-shadow: 0 4px 20px rgba(74,158,222,0.1), inset 0 1px 0 rgba(255,255,255,0.05) !important;
    transform: perspective(800px) rotateY(-0.5deg) !important;
}

/* Assistant bubble */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) .stChatMessageContent,
[data-testid="stChatMessage"][data-role="assistant"] .stMarkdown {
    background: linear-gradient(135deg, rgba(13,34,64,0.85), rgba(9,22,40,0.9)) !important;
    border: 1px solid var(--border) !important;
    border-left: 2px solid var(--gold) !important;
    border-radius: 4px 16px 16px 16px !important;
    padding: 0.9rem 1.1rem !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.03) !important;
    transform: perspective(800px) rotateY(0.5deg) !important;
}

/* Avatar orbs */
[data-testid="stChatMessageAvatarUser"] {
    background: linear-gradient(135deg, #1e6db5, #4a9ede) !important;
    border: 1px solid rgba(74,158,222,0.4) !important;
    box-shadow: 0 0 12px rgba(74,158,222,0.3) !important;
    border-radius: 50% !important;
}
[data-testid="stChatMessageAvatarAssistant"] {
    background: linear-gradient(135deg, #7a5a20, var(--gold)) !important;
    border: 1px solid rgba(201,168,76,0.4) !important;
    box-shadow: 0 0 12px rgba(201,168,76,0.3) !important;
    border-radius: 50% !important;
}

/* Source badges */
.src-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: rgba(201,168,76,0.08);
    border: 1px solid rgba(201,168,76,0.25);
    border-radius: 8px;
    padding: 3px 10px;
    font-size: 0.72rem;
    color: #d4b050;
    margin: 3px 3px 3px 0;
    font-family: 'DM Sans', sans-serif;
    letter-spacing: 0.03em;
    transition: all 0.2s;
}
.src-badge:hover {
    background: rgba(201,168,76,0.15);
    border-color: rgba(201,168,76,0.45);
    transform: translateY(-1px);
}
.src-label {
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    margin: 8px 0 4px;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    position: fixed !important;
    bottom: 0 !important;
    left: 50% !important;
    transform: translateX(-50%) !important;
    width: min(820px, 100%) !important;
    padding: 1rem 1.5rem !important;
    background: linear-gradient(to top, var(--midnight) 60%, transparent) !important;
    z-index: 100 !important;
    backdrop-filter: blur(12px) !important;
}
[data-testid="stChatInputTextArea"] {
    background: rgba(13,34,64,0.8) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    box-shadow: 0 0 0 0 transparent, 0 8px 32px rgba(0,0,0,0.3) !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
[data-testid="stChatInputTextArea"]:focus {
    border-color: rgba(201,168,76,0.5) !important;
    box-shadow: 0 0 0 3px rgba(201,168,76,0.08), 0 8px 32px rgba(0,0,0,0.3) !important;
}

/* ── Spinner ── */
.stSpinner > div {
    border-top-color: var(--gold) !important;
}
.stSpinner p { color: var(--muted) !important; font-size: 0.8rem !important; }

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 1rem 0 !important; }

/* ── Animations ── */
@keyframes fade-in-up {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
[data-testid="stChatMessage"] {
    animation: fade-in-up 0.35s ease both;
}

/* ── Streamlit default overrides ── */
[data-testid="stMarkdownContainer"] p { color: var(--text); line-height: 1.65; }
[data-testid="stMarkdownContainer"] strong { color: var(--gold-lt); }
[data-testid="stMarkdownContainer"] a { color: var(--sky-lt); }
footer, #MainMenu, header { display: none !important; }
</style>

<!-- Floating particles -->
<script>
(function(){
    const colors = ['#c9a84c','#4a9ede','#f0d080'];
    for(let i = 0; i < 18; i++){
        const p = document.createElement('div');
        p.className = 'particle';
        const s = Math.random()*3 + 1;
        p.style.cssText = `
            width:${s}px; height:${s}px;
            left:${Math.random()*100}%;
            bottom:${Math.random()*20}%;
            background:${colors[Math.floor(Math.random()*3)]};
            opacity:${Math.random()*0.5+0.1};
            animation-duration:${Math.random()*20+15}s;
            animation-delay:${Math.random()*10}s;
        `;
        document.body.appendChild(p);
    }
})();
</script>
""", unsafe_allow_html=True)

# ── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-card">
    <div style="font-size:2.4rem;margin-bottom:0.2rem">🎓</div>
    <div class="hero-title">Central University of Kashmir</div>
    <div class="hero-sub">AI Campus Intelligence</div>
    <div class="hero-badge-row">
        <span class="hero-badge">Admissions</span>
        <span class="hero-badge">Courses</span>
        <span class="hero-badge">Faculty</span>
        <span class="hero-badge">Events</span>
        <span class="hero-badge">Research</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Session state ────────────────────────────────────────────────────────────
if "history"  not in st.session_state: st.session_state.history  = []
if "messages" not in st.session_state: st.session_state.messages = []

# ── Suggested questions (first load only) ───────────────────────────────────
if not st.session_state.messages:
    st.markdown('<div class="section-label">Quick questions</div>', unsafe_allow_html=True)
    questions = [
        "📋  Admission process for 2026",
        "📚  Courses available at CUK",
        "📅  When is the CUET exam?",
        "🔬  How to apply for PhD?",
    ]
    cols = st.columns(2)
    for i, q in enumerate(questions):
        if cols[i % 2].button(q, use_container_width=True, key=f"sq_{i}"):
            st.session_state.pending_question = q.split("  ", 1)[-1]
            st.rerun()
    st.markdown("<br>", unsafe_allow_html=True)

# ── Render chat history ──────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            st.markdown('<div class="src-label">📚 Sources</div>', unsafe_allow_html=True)
            badges = "".join(
                f'<span class="src-badge">📄 {s}</span>' for s in msg["sources"]
            )
            st.markdown(f'<div>{badges}</div>', unsafe_allow_html=True)

# ── Input handling ───────────────────────────────────────────────────────────
query = st.chat_input("Ask anything about Central University of Kashmir…")
if hasattr(st.session_state, "pending_question"):
    query = st.session_state.pending_question
    del st.session_state.pending_question

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching university documents…"):
            stream, sources = run(query, st.session_state.history)

        full_answer = ""
        placeholder = st.empty()
        for chunk in stream:
            if hasattr(chunk, "text") and chunk.text:
                full_answer += chunk.text
                placeholder.markdown(full_answer + "▌")
        placeholder.markdown(full_answer)

        if sources:
            st.markdown('<div class="src-label">📚 Sources</div>', unsafe_allow_html=True)
            badges = "".join(
                f'<span class="src-badge">📄 {s}</span>' for s in sources
            )
            st.markdown(f'<div>{badges}</div>', unsafe_allow_html=True)

    st.session_state.history.append({"user": query, "bot": full_answer})
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_answer,
        "sources": sources
    })
