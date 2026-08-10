"""
RECALL — AI Meeting Intelligence

Premium dark AI workspace UI.
Backend / pipeline logic remains unchanged.
"""

import html
import re
import time
import textwrap
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from utils.audio_processor import process_input
from core.pipeline import build_meeting
from core.extractor import _format_action_items, _format_decisions, _format_questions
from core.rag_engine import build_rag_chains_from_meeting, ask_question_structured
from core.models.transcript import format_timestamp
from core.report_generator import generate_pdf_report, generate_txt_report


load_dotenv()

st.set_page_config(
    page_title="RECALL — Meeting Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def ui(markup: str) -> None:
    """Render HTML as a real HTML block, not a Markdown code block."""
    markup = textwrap.dedent(markup).strip()
    # CommonMark can terminate a <div> HTML block at blank lines.
    # Removing empty lines keeps nested UI markup inside one HTML block.
    markup = "\n".join(line for line in markup.splitlines() if line.strip())
    st.markdown(markup, unsafe_allow_html=True)


def summary_html(text: str) -> str:
    """Render generated meeting Markdown as safe, polished HTML."""
    text = esc(text).strip()
    if not text:
        return "<span style='color:var(--text-3);'>No summary available.</span>"

    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(
        r"(?m)^#{1,3}\s+(.+)$",
        r'<div class="summary-heading">\1</div>',
        text,
    )

    out = []
    list_type = None

    def close_list():
        nonlocal list_type
        if list_type:
            out.append(f"</{list_type}>")
            list_type = None

    for raw in text.splitlines():
        line = raw.strip()

        if not line:
            close_list()
            continue

        bullet = re.match(r"^[•\-\*]\s+(.+)$", line)
        number = re.match(r"^\d+[.)]\s+(.+)$", line)

        if bullet:
            if list_type != "ul":
                close_list()
                out.append("<ul class='summary-list'>")
                list_type = "ul"
            out.append(f"<li>{bullet.group(1)}</li>")

        elif number:
            if list_type != "ol":
                close_list()
                out.append("<ol class='summary-list'>")
                list_type = "ol"
            out.append(f"<li>{number.group(1)}</li>")

        elif line.startswith('<div class="summary-heading">'):
            close_list()
            out.append(line)

        else:
            close_list()
            out.append(f"<p>{line}</p>")

    close_list()
    return "".join(out)


def esc(value) -> str:
    return html.escape(str(value or ""))


def source_label(source: str) -> str:
    if not source:
        return "Unknown source"
    low = source.lower()
    if "youtube" in low or "youtu.be" in low:
        return "YouTube"
    return "Local recording"


def clear_meeting() -> None:
    st.session_state.result = None
    st.session_state.chat_history = []
    st.session_state.question_to_ask = None
    st.session_state.pipeline_done = False


# ─────────────────────────────────────────────────────────────────────────────
# PREMIUM VISUAL SYSTEM
# ─────────────────────────────────────────────────────────────────────────────

ui("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
    --bg: #080A0F;
    --bg-2: #0B0E15;
    --panel: #10141C;
    --panel-2: #141925;
    --panel-3: #181E2A;
    --line: rgba(255,255,255,.075);
    --line-strong: rgba(255,255,255,.13);

    --text: #F4F6FA;
    --text-2: #B6BECC;
    --text-3: #727C8E;

    --violet: #8C6BFF;
    --indigo: #6E7CFF;
    --cyan: #45D7D0;
    --blue: #4C9AFF;
    --green: #55D6A3;
    --amber: #F2B45F;
    --red: #FF6674;

    --shadow: 0 18px 60px rgba(0,0,0,.34);
    --radius: 14px;
}

* { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: "DM Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    color: var(--text);
}

.stApp {
    background:
        radial-gradient(900px 500px at 82% -8%, rgba(111,92,255,.13), transparent 62%),
        radial-gradient(700px 420px at 8% 105%, rgba(69,215,208,.065), transparent 65%),
        radial-gradient(520px 300px at 55% 45%, rgba(124,97,255,.035), transparent 70%),
        linear-gradient(180deg, #080A0F 0%, #090B10 100%);
}

.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    background-image:
        linear-gradient(rgba(255,255,255,.012) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,.012) 1px, transparent 1px);
    background-size: 56px 56px;
    mask-image: linear-gradient(to bottom, black, transparent 78%);
}

.main .block-container {
    max-width: 1440px;
    padding: 3.2rem 4.2rem 7rem;
    position: relative;
    z-index: 1;
}

h1, h2, h3, h4 {
    font-family: "Space Grotesk", "DM Sans", sans-serif !important;
    color: var(--text) !important;
    letter-spacing: -.035em !important;
}

h1 { font-size: 3.1rem !important; }
h2 { font-size: 2rem !important; }
h3 { font-size: 1.35rem !important; }

p, li { color: var(--text-2); }

.small-label {
    color: var(--text-3);
    font-size: .69rem;
    font-weight: 700;
    letter-spacing: .14em;
    text-transform: uppercase;
}

.mono {
    font-family: "SFMono-Regular", Menlo, Monaco, Consolas, monospace;
    letter-spacing: -.02em;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg, rgba(16,20,28,.98), rgba(10,13,18,.98));
    border-right: 1px solid var(--line);
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.7rem;
}

[data-testid="stSidebar"] .block-container {
    padding: 1.5rem 1.15rem 2rem;
}

.brand {
    display: flex;
    align-items: center;
    gap: .72rem;
}

.brand-mark {
    width: 31px;
    height: 31px;
    border-radius: 10px;
    background:
        radial-gradient(circle at 30% 28%, #C9C0FF 0 10%, transparent 11%),
        linear-gradient(135deg, var(--violet), var(--blue));
    box-shadow: 0 0 26px rgba(140,107,255,.25);
    position: relative;
}

.brand-mark::after {
    content: "";
    position: absolute;
    width: 9px;
    height: 9px;
    right: 6px;
    bottom: 6px;
    border-radius: 50%;
    background: var(--cyan);
    box-shadow: 0 0 12px rgba(69,215,208,.7);
}

.brand-name {
    font-family: "Space Grotesk", sans-serif;
    font-size: 1.15rem;
    font-weight: 700;
    letter-spacing: -.03em;
}

.brand-sub {
    color: var(--text-3);
    font-size: .68rem;
    letter-spacing: .12em;
    text-transform: uppercase;
    margin-top: 1px;
}

.sidebar-rule {
    height: 1px;
    background: var(--line);
    margin: 1.45rem 0;
}

[data-testid="stSidebar"] label {
    color: var(--text-3) !important;
    font-size: .72rem !important;
    font-weight: 700 !important;
    letter-spacing: .1em !important;
    text-transform: uppercase !important;
}

[data-testid="stSidebar"] .stRadio > div {
    gap: .3rem;
}

[data-testid="stSidebar"] .stRadio label {
    background: transparent !important;
    border: 1px solid transparent !important;
    border-radius: 9px !important;
    padding: .45rem .55rem !important;
    color: var(--text-2) !important;
    font-size: .88rem !important;
    letter-spacing: 0 !important;
    text-transform: none !important;
}

[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(255,255,255,.035) !important;
}

[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div {
    background: rgba(255,255,255,.025) !important;
    border: 1px solid var(--line-strong) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}

[data-testid="stSidebar"] .stTextInput input:focus {
    border-color: rgba(140,107,255,.75) !important;
    box-shadow: 0 0 0 3px rgba(140,107,255,.10) !important;
}

[data-testid="stFileUploader"] section {
    background: rgba(255,255,255,.018) !important;
    border: 1px dashed var(--line-strong) !important;
    border-radius: 11px !important;
}

[data-testid="stFileUploader"] section:hover {
    border-color: rgba(69,215,208,.5) !important;
}

[data-testid="stSidebar"] .stButton > button {
    min-height: 42px;
}

/* Buttons */
.stButton > button,
.stDownloadButton > button {
    border-radius: 10px !important;
    border: 1px solid var(--line-strong) !important;
    background: rgba(255,255,255,.028) !important;
    color: var(--text) !important;
    font-weight: 600 !important;
    transition: .18s ease !important;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    border-color: rgba(140,107,255,.55) !important;
    background: rgba(140,107,255,.08) !important;
    transform: translateY(-1px);
    box-shadow: 0 8px 25px rgba(0,0,0,.2);
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #7B61FF, #4C9AFF) !important;
    border: 0 !important;
    color: white !important;
    box-shadow: 0 8px 28px rgba(94,93,255,.22) !important;
}

.stButton > button[kind="primary"]:hover {
    filter: brightness(1.08);
    box-shadow: 0 10px 34px rgba(94,93,255,.34) !important;
}

/* Inputs */
.stTextInput input,
.stTextArea textarea,
[data-baseweb="select"] > div {
    background: rgba(255,255,255,.025) !important;
    color: var(--text) !important;
    border-color: var(--line-strong) !important;
    border-radius: 10px !important;
}

.stTextInput input:focus,
.stTextArea textarea:focus {
    border-color: rgba(140,107,255,.75) !important;
    box-shadow: 0 0 0 3px rgba(140,107,255,.10) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: .2rem;
    border-bottom: 1px solid var(--line);
}

.stTabs [data-baseweb="tab"] {
    color: var(--text-3) !important;
    font-weight: 600 !important;
    padding: .8rem 1rem !important;
    border: 0 !important;
}

.stTabs [data-baseweb="tab"]:hover {
    color: var(--text-2) !important;
}

.stTabs [aria-selected="true"] {
    color: var(--text) !important;
}

.stTabs [data-baseweb="tab-highlight"] {
    background: linear-gradient(90deg, var(--violet), var(--cyan)) !important;
    height: 2px !important;
}

/* Native chat input — use this instead of forms */
[data-testid="stChatInput"] {
    border: 1px solid rgba(140,107,255,.26) !important;
    background: rgba(15,18,26,.94) !important;
    box-shadow: 0 14px 45px rgba(0,0,0,.35), 0 0 0 5px rgba(140,107,255,.035) !important;
    border-radius: 14px !important;
}

[data-testid="stChatInput"] textarea {
    color: var(--text) !important;
}

/* Streamlit messages */
[data-testid="stAlert"] {
    border-radius: 11px !important;
}

/* Custom surfaces */
.hero {
    position: relative;
    overflow: hidden;
    border: 1px solid var(--line-strong);
    border-radius: 22px;
    padding: 3.6rem;
    min-height: 470px;
    background:
        radial-gradient(circle at 78% 28%, rgba(124,97,255,.20), transparent 26%),
        radial-gradient(circle at 15% 90%, rgba(69,215,208,.075), transparent 28%),
        linear-gradient(145deg, rgba(20,25,37,.94), rgba(8,11,17,.98));
    box-shadow: 0 28px 90px rgba(0,0,0,.42), inset 0 1px 0 rgba(255,255,255,.035);
}

.hero::before {
    content: "";
    position: absolute;
    width: 340px;
    height: 340px;
    right: -160px;
    top: -170px;
    border-radius: 50%;
    border: 1px solid rgba(140,107,255,.16);
    box-shadow: 0 0 80px rgba(140,107,255,.08);
}

.hero-content { position: relative; z-index: 2; max-width: 760px; }
.hero-orbit { position:absolute; border:1px solid rgba(140,107,255,.12); border-radius:50%; pointer-events:none; }
.orbit-a { width:520px; height:520px; right:-170px; top:-40px; transform:rotate(-18deg); }
.orbit-b { width:350px; height:350px; right:-90px; top:50px; border-color:rgba(69,215,208,.10); }
.live-dot { width:6px; height:6px; border-radius:50%; background:#55D6A3; box-shadow:0 0 12px #55D6A3; }

.hero-kicker {
    display: inline-flex;
    align-items: center;
    gap: .45rem;
    color: #BDB4FF;
    background: rgba(140,107,255,.09);
    border: 1px solid rgba(140,107,255,.18);
    border-radius: 999px;
    padding: .34rem .65rem;
    font-size: .7rem;
    font-weight: 700;
    letter-spacing: .1em;
    text-transform: uppercase;
}

.hero-title {
    font-family: "Space Grotesk", sans-serif;
    font-size: clamp(3rem, 6vw, 5.7rem);
    line-height: .98;
    letter-spacing: -.065em;
    max-width: 760px;
    margin: 1.5rem 0 1rem;
    color: #F8F9FC;
}

.hero-title span {
    background: linear-gradient(100deg, #F8F9FC 15%, #AFA0FF 58%, #67DCD6 100%);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}

.hero-copy {
    max-width: 650px;
    color: var(--text-2);
    font-size: 1.03rem;
}

.hero-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: .75rem;
    margin-top: 2.3rem;
}

.hero-stat {
    border: 1px solid var(--line);
    background: rgba(255,255,255,.025);
    border-radius: 12px;
    padding: 1rem;
}

.hero-stat strong {
    display: block;
    color: var(--text);
    font-size: .88rem;
}

.hero-stat span {
    color: var(--text-3);
    font-size: .75rem;
}

.meeting-head {
    padding: 1.1rem 0 2rem;
}

.meeting-kicker {
    color: var(--cyan);
    font-size: .68rem;
    font-weight: 800;
    letter-spacing: .16em;
    text-transform: uppercase;
}

.meeting-title {
    font-family: "Space Grotesk", sans-serif;
    font-size: clamp(2rem, 4vw, 3.4rem);
    font-weight: 700;
    letter-spacing: -.055em;
    line-height: 1.05;
    margin: .65rem 0 .85rem;
    max-width: 980px;
}

.meta-row {
    display: flex;
    flex-wrap: wrap;
    gap: .5rem;
    align-items: center;
    color: var(--text-3);
    font-size: .78rem;
}

.meta-chip {
    border: 1px solid var(--line);
    background: rgba(255,255,255,.022);
    border-radius: 999px;
    padding: .28rem .58rem;
}

.section-head {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    margin: .2rem 0 1rem;
}

.section-head h3 {
    margin: 0;
}

.section-head span {
    color: var(--text-3);
    font-size: .75rem;
}

.surface {
    background: linear-gradient(145deg, rgba(22,27,39,.88), rgba(14,17,24,.88));
    border: 1px solid var(--line);
    border-radius: var(--radius);
    box-shadow: 0 10px 35px rgba(0,0,0,.18);
}

.summary-surface {
    padding: 1.55rem 1.7rem;
    border-left: 2px solid var(--violet);
}

.summary-heading {
    margin: 1.25rem 0 .55rem;
    color: var(--text);
    font-family: "Space Grotesk", sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: -.01em;
}
.summary-text p { margin: 0 0 .85rem; }
.summary-list {
    margin: .35rem 0 1rem 1.2rem;
    padding: 0;
}
.summary-list li {
    margin: .55rem 0;
    padding-left: .25rem;
    color: var(--text-2);
}
.pipeline-status {
    display: flex;
    align-items: center;
    gap: .8rem;
    padding: .85rem 1rem;
    margin-top: .8rem;
    border: 1px solid rgba(255,255,255,.07);
    border-radius: 12px;
    background: linear-gradient(90deg, rgba(20,27,39,.96), rgba(12,16,24,.92));
    box-shadow: inset 0 1px 0 rgba(255,255,255,.025);
}
.pipeline-orb {
    width: 9px;
    height: 9px;
    flex: 0 0 9px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--violet), var(--cyan));
    box-shadow: 0 0 16px rgba(140,107,255,.55);
    animation: pipelinePulse 1.5s ease-in-out infinite;
}
.pipeline-copy { display: flex; flex-direction: column; gap: .12rem; }
.pipeline-label {
    color: var(--text);
    font-size: .86rem;
    font-weight: 600;
}
.pipeline-sub {
    color: var(--text-3);
    font-size: .72rem;
    letter-spacing: .02em;
}
@keyframes pipelinePulse {
    0%,100% { transform: scale(.82); opacity: .65; }
    50% { transform: scale(1); opacity: 1; }
}
.summary-text {
    color: #DDE2EA;
    font-size: .98rem;
    line-height: 1.72;
}

.metrics {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: .8rem;
}

.metric {
    position: relative;
    overflow: hidden;
    padding: 1.15rem 1.25rem;
    background: rgba(255,255,255,.025);
    border: 1px solid var(--line);
    border-radius: 13px;
}

.metric::after {
    content: "";
    position: absolute;
    width: 100px;
    height: 100px;
    right: -55px;
    bottom: -55px;
    border-radius: 50%;
    opacity: .12;
}

.metric.violet::after { background: var(--violet); }
.metric.green::after { background: var(--green); }
.metric.cyan::after { background: var(--cyan); }

.metric-value {
    font-family: "Space Grotesk", sans-serif;
    font-size: 1.9rem;
    font-weight: 700;
    color: var(--text);
}

.metric-label {
    color: var(--text-3);
    font-size: .7rem;
    font-weight: 700;
    letter-spacing: .12em;
    text-transform: uppercase;
}

.insight {
    position: relative;
    margin-bottom: .75rem;
    padding: 1.25rem 1.35rem 1.25rem 1.5rem;
    background: rgba(255,255,255,.024);
    border: 1px solid var(--line);
    border-radius: 13px;
    transition: .18s ease;
}

.insight:hover {
    background: rgba(255,255,255,.038);
    border-color: var(--line-strong);
    transform: translateY(-1px);
}

.insight::before {
    content: "";
    position: absolute;
    left: 0;
    top: 14px;
    bottom: 14px;
    width: 3px;
    border-radius: 0 4px 4px 0;
}

.insight.decision::before { background: var(--amber); }
.insight.action::before { background: var(--green); }
.insight.question::before { background: var(--cyan); }

.insight-label {
    color: var(--text-3);
    font-size: .66rem;
    font-weight: 800;
    letter-spacing: .12em;
    text-transform: uppercase;
}

.insight-text {
    color: #E6E9EF;
    margin-top: .45rem;
    line-height: 1.65;
}

.insight-meta {
    margin-top: .65rem;
    color: var(--text-3);
    font-size: .72rem;
}

.transcript-row {
    display: grid;
    grid-template-columns: 72px 110px 1fr;
    gap: 1rem;
    padding: .9rem 0;
    border-bottom: 1px solid var(--line);
}

.transcript-row:last-child { border-bottom: 0; }

.ts {
    color: var(--cyan);
    font-family: monospace;
    font-size: .72rem;
    padding-top: .15rem;
}

.speaker {
    color: var(--text-3);
    font-size: .76rem;
    font-weight: 700;
}

.transcript-text {
    color: #DCE1E9;
    line-height: 1.65;
    font-size: .9rem;
}

.chat-shell {
    margin-top: .5rem;
    padding: 1.2rem;
    border: 1px solid var(--line);
    border-radius: 16px;
    background:
        radial-gradient(circle at 90% 0%, rgba(140,107,255,.07), transparent 30%),
        rgba(15,18,25,.78);
}

.chat-row {
    display: flex;
    margin: 1rem 0;
}

.chat-row.user { justify-content: flex-end; }

.chat-bubble {
    max-width: 78%;
    padding: .8rem 1rem;
    border-radius: 13px;
    line-height: 1.62;
    font-size: .91rem;
}

.chat-bubble.user {
    color: white;
    background: linear-gradient(135deg, rgba(124,97,255,.96), rgba(70,143,255,.96));
    box-shadow: 0 8px 28px rgba(86,91,255,.16);
    border-bottom-right-radius: 4px;
}

.chat-bubble.assistant {
    color: #E4E8EF;
    background: rgba(255,255,255,.035);
    border: 1px solid var(--line);
    border-bottom-left-radius: 4px;
}

.evidence {
    margin-top: .9rem;
    padding: .85rem 1rem;
    background: rgba(69,215,208,.035);
    border: 1px solid rgba(69,215,208,.12);
    border-radius: 11px;
}

.evidence-title {
    color: var(--cyan);
    font-size: .68rem;
    font-weight: 800;
    letter-spacing: .12em;
    text-transform: uppercase;
}

.evidence-item {
    margin-top: .65rem;
    padding-left: .75rem;
    border-left: 2px solid rgba(69,215,208,.42);
}

.evidence-time {
    color: var(--cyan);
    font-family: monospace;
    font-size: .7rem;
}

.evidence-quote {
    color: var(--text-2);
    font-size: .8rem;
    font-style: italic;
    line-height: 1.55;
    margin-top: .15rem;
}

.sidebar-meeting {
    padding: .9rem;
    border: 1px solid var(--line);
    border-radius: 11px;
    background: rgba(255,255,255,.022);
}

.sidebar-title {
    color: var(--text);
    font-size: .86rem;
    font-weight: 600;
    line-height: 1.4;
}

.sidebar-meta {
    color: var(--text-3);
    font-size: .72rem;
    margin-top: .45rem;
    line-height: 1.55;
}

@media (max-width: 900px) {
    .main .block-container { padding: 1.5rem 1.2rem 6rem; }
    .hero { padding: 2rem; }
    .hero-grid, .metrics { grid-template-columns: 1fr; }
    .transcript-row { grid-template-columns: 60px 1fr; }
    .speaker { display: none; }
}
</style>
""")


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────

defaults = {
    "result": None,
    "chat_history": [],
    "pipeline_done": False,
    "question_to_ask": None,
    "input_mode": "YouTube",
    "language": "english",
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    ui("""
    <div class="brand">
        <div class="brand-mark"></div>
        <div>
            <div class="brand-name">RECALL</div>
            <div class="brand-sub">Meeting intelligence</div>
        </div>
    </div>
    <div class="sidebar-rule"></div>
    """)

    ui('<div class="small-label" style="margin-bottom:.7rem;">New meeting</div>')

    input_mode = st.radio(
        "Input source",
        ["YouTube", "Upload"],
        horizontal=True,
        label_visibility="collapsed",
        key="input_mode",
    )

    source = None

    if input_mode == "YouTube":
        source = st.text_input(
            "YouTube URL",
            placeholder="Paste a YouTube URL",
            label_visibility="collapsed",
            key="yt_url",
        )
    else:
        uploaded_file = st.file_uploader(
            "Meeting recording",
            type=("mp3", "m4a", "wav", "mp4", "webm", "ogg", "flac", "aac", "mkv", "mov", "avi"),
            label_visibility="collapsed",
            key="file_upload",
        )

        if uploaded_file is not None:
            import tempfile

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=Path(uploaded_file.name).suffix,
            ) as tmp:
                tmp.write(uploaded_file.getbuffer())
                source = tmp.name

            ui(f"""
            <div style="margin-top:.55rem; color:var(--text-3); font-size:.72rem;">
                <span style="color:var(--green);">●</span>
                {esc(uploaded_file.name)} · {uploaded_file.size / 1024 / 1024:.1f} MB
            </div>
            """)

    language = st.selectbox(
        "Language",
        ["english", "hinglish"],
        label_visibility="collapsed",
        key="language",
    )

    analyze_btn = st.button(
        "Analyze meeting",
        use_container_width=True,
        type="primary",
        key="analyze_btn",
    )

    if st.session_state.result:
        ui('<div class="sidebar-rule"></div><div class="small-label" style="margin-bottom:.7rem;">Current meeting</div>')

        meeting = st.session_state.result["meeting"]
        ui(f"""
        <div class="sidebar-meeting">
            <div class="sidebar-title">{esc(meeting.title or "Untitled meeting")}</div>
            <div class="sidebar-meta">
                {esc(meeting.language.title())} · {format_timestamp(meeting.duration or 0)}<br>
                {esc(source_label(meeting.source))}
            </div>
        </div>
        """)

        if st.button("＋  New meeting", use_container_width=True, key="new_meeting"):
            clear_meeting()
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# LANDING
# ─────────────────────────────────────────────────────────────────────────────

if not st.session_state.result:
    st.markdown("")

    ui("""
    <div class="hero">
        <div class="hero-orbit orbit-a"></div>
        <div class="hero-orbit orbit-b"></div>
        <div class="hero-content">
            <div class="hero-kicker"><span class="live-dot"></span> AI MEETING INTELLIGENCE</div>
            <div class="hero-title">Meetings in.<br><span>Intelligence out.</span></div>
            <div class="hero-copy">RECALL turns raw recordings into a decision-ready workspace — with searchable transcripts, actions, decisions, and evidence-grounded answers.</div>
            <div class="hero-grid">
                <div class="hero-stat"><strong>01&nbsp; / &nbsp;TRANSCRIBE</strong><span>Capture every spoken detail</span></div>
                <div class="hero-stat"><strong>02&nbsp; / &nbsp;UNDERSTAND</strong><span>Extract what actually matters</span></div>
                <div class="hero-stat"><strong>03&nbsp; / &nbsp;ASK</strong><span>Find answers with evidence</span></div>
            </div>
        </div>
    </div>
    """)

    st.markdown("<div style='height:1.1rem'></div>", unsafe_allow_html=True)

    c1, c2 = st.columns([1.35, .65], gap="medium")

    with c1:
        ui("""
        <div class="surface" style="padding:1.35rem 1.45rem;">
            <div class="small-label">Get started</div>
            <div style="font-family:'Space Grotesk';font-size:1.15rem;font-weight:600;margin-top:.4rem;">
                Add a recording from the sidebar
            </div>
            <div style="color:var(--text-3);font-size:.82rem;margin-top:.3rem;">
                MP4 · MP3 · WAV · M4A · WEBM · YouTube
            </div>
        </div>
        """)

    with c2:
        ui("""
        <div class="surface" style="padding:1.35rem 1.45rem;">
            <div class="small-label">Built for</div>
            <div style="color:var(--text-2);font-size:.86rem;margin-top:.4rem;line-height:1.6;">
                Product reviews · research calls · interviews · team syncs
            </div>
        </div>
        """)


# ─────────────────────────────────────────────────────────────────────────────
# MEETING DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

else:
    r = st.session_state.result
    meeting = r["meeting"]

    ui(f"""
    <div class="meeting-head">
        <div class="meeting-kicker"><span style="color:var(--green);">●</span> ANALYSIS COMPLETE</div>
        <div class="meeting-title">{esc(meeting.title or "Untitled meeting")}</div>
        <div class="meta-row">
            <span class="meta-chip mono">{format_timestamp(0)} → {format_timestamp(meeting.duration or 0)}</span>
            <span class="meta-chip">{esc(meeting.language.title())}</span>
            <span class="meta-chip">{esc(source_label(meeting.source))}</span>
        </div>
    </div>
    """)

    tab_overview, tab_decisions, tab_actions, tab_questions, tab_transcript = st.tabs(
        ["Overview", "Decisions", "Actions", "Questions", "Transcript"]
    )

    with tab_overview:
        ui("""
        <div class="section-head">
            <h3>Meeting overview</h3>
            <span>Executive intelligence</span>
        </div>
        """)

        ui(f"""
        <div class="surface summary-surface">
            <div class="small-label" style="margin-bottom:.8rem;">Executive brief</div>
            <div class="summary-text">{summary_html(r["summary"])}</div>
        </div>
        """)

        st.markdown("<div style='height:.9rem'></div>", unsafe_allow_html=True)

        ui(f"""
        <div class="metrics">
            <div class="metric violet">
                <div class="metric-value">{len(meeting.decisions)}</div>
                <div class="metric-label">Decisions</div>
            </div>
            <div class="metric green">
                <div class="metric-value">{len(meeting.action_items)}</div>
                <div class="metric-label">Action items</div>
            </div>
            <div class="metric cyan">
                <div class="metric-value">{len(meeting.open_questions)}</div>
                <div class="metric-label">Open questions</div>
            </div>
        </div>
        """)

    with tab_decisions:
        ui("""
        <div class="section-head">
            <h3>Key decisions</h3>
            <span>What was agreed</span>
        </div>
        """)

        if meeting.decisions:
            for i, decision in enumerate(meeting.decisions, 1):
                ts = (
                    format_timestamp(decision.timestamp_ref)
                    if decision.timestamp_ref is not None
                    else ""
                )
                ui(f"""
                <div class="insight decision">
                    <div class="insight-label">Decision {i:02d}</div>
                    <div class="insight-text">{esc(decision.decision)}</div>
                    {f'<div class="insight-meta mono">{esc(ts)}</div>' if ts else ''}
                </div>
                """)
        else:
            st.info("No decisions identified in this meeting.")

    with tab_actions:
        ui("""
        <div class="section-head">
            <h3>Action items</h3>
            <span>What happens next</span>
        </div>
        """)

        if meeting.action_items:
            for i, item in enumerate(meeting.action_items, 1):
                owner = item.owner or "Unassigned"
                due = item.deadline or "No deadline specified"
                ui(f"""
                <div class="insight action">
                    <div class="insight-label">Action {i:02d}</div>
                    <div class="insight-text">{esc(item.task)}</div>
                    <div class="insight-meta">
                        <span style="color:var(--green);">Owner</span> · {esc(owner)}
                        &nbsp;&nbsp; <span style="color:var(--green);">Due</span> · {esc(due)}
                    </div>
                </div>
                """)
        else:
            st.info("No action items identified in this meeting.")

    with tab_questions:
        ui("""
        <div class="section-head">
            <h3>Open questions</h3>
            <span>Still unresolved</span>
        </div>
        """)

        if meeting.open_questions:
            for i, question in enumerate(meeting.open_questions, 1):
                ui(f"""
                <div class="insight question">
                    <div class="insight-label">Question {i:02d}</div>
                    <div class="insight-text">{esc(question.question)}</div>
                </div>
                """)
        else:
            st.info("No open questions identified in this meeting.")

    with tab_transcript:
        ui("""
        <div class="section-head">
            <h3>Transcript</h3>
            <span>Timestamped source material</span>
        </div>
        """)

        if meeting.segments:
            rows = []
            for seg in meeting.segments:
                speaker = esc(seg.speaker or "Speaker")
                rows.append(
                    f"""
                    <div class="transcript-row">
                        <div class="ts">{esc(format_timestamp(seg.start))}</div>
                        <div class="speaker">{speaker}</div>
                        <div class="transcript-text">{esc(seg.text)}</div>
                    </div>
                    """
                )

            ui(f'<div class="surface" style="padding:0 1.25rem;">{"".join(rows)}</div>')
        else:
            st.info("No transcript available.")

    # ─────────────────────────────────────────────────────────────────────────
    # EXPORT
    # ─────────────────────────────────────────────────────────────────────────

    st.markdown("<div style='height:2.2rem'></div>", unsafe_allow_html=True)

    ui("""
    <div class="section-head">
        <h3>Export</h3>
        <span>Take the meeting with you</span>
    </div>
    """)

    export1, export2 = st.columns(2, gap="medium")

    with export1:
        pdf_bytes = generate_pdf_report(meeting, r["summary"])
        st.download_button(
            "↓  Download PDF report",
            data=pdf_bytes,
            file_name=f"RECALL-{meeting.title or 'Meeting'}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    with export2:
        txt_content = generate_txt_report(meeting, r["summary"])
        st.download_button(
            "↓  Download TXT report",
            data=txt_content,
            file_name=f"RECALL-{meeting.title or 'Meeting'}.txt",
            mime="text/plain",
            use_container_width=True,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # ASK RECALL
    # ─────────────────────────────────────────────────────────────────────────

    st.markdown("<div style='height:2.2rem'></div>", unsafe_allow_html=True)

    ui("""
    <div class="chat-shell">
        <div class="small-label">Ask RECALL</div>
        <div style="font-family:'Space Grotesk';font-size:1.5rem;font-weight:600;margin-top:.35rem;">
            Ask anything about this meeting.
        </div>
        <div style="color:var(--text-3);font-size:.82rem;margin-top:.25rem;">
            Answers are grounded in the transcript and show their source evidence.
        </div>
    </div>
    """)

    suggestions = [
        ("Decisions", "What are the main decisions?"),
        ("Actions", "What are the action items?"),
        ("Ideas", "Explain the main ideas"),
        ("Ownership", "Who owns which tasks?"),
    ]

    s_cols = st.columns(4, gap="small")
    for col, (label, question) in zip(s_cols, suggestions):
        with col:
            if st.button(label, use_container_width=True, key=f"suggest_{label}"):
                st.session_state.question_to_ask = question
                st.rerun()

    # Existing chat history
    if st.session_state.chat_history:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                ui(f"""
                <div class="chat-row user">
                    <div class="chat-bubble user">{esc(msg["content"])}</div>
                </div>
                """)
            else:
                ui(f"""
                <div class="chat-row">
                    <div class="chat-bubble assistant">
                        {esc(msg["answer"]).replace(chr(10), "<br>")}
                    </div>
                </div>
                """)

                if msg.get("evidence"):
                    evidence_html = []
                    for ev in msg["evidence"]:
                        evidence_html.append(
                            f"""
                            <div class="evidence-item">
                                <div class="evidence-time">
                                    {esc(format_timestamp(ev.start))} → {esc(format_timestamp(ev.end))}
                                </div>
                                <div class="evidence-quote">“{esc(ev.text)}”</div>
                            </div>
                            """
                        )

                    ui(f"""
                    <div class="evidence">
                        <div class="evidence-title">Transcript evidence</div>
                        {''.join(evidence_html)}
                    </div>
                    """)

        if st.button("Clear conversation", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()

    # IMPORTANT:
    # st.chat_input is intentionally NOT inside st.form.
    # Streamlit's native chat input handles Enter submission correctly.
    typed_question = st.chat_input(
        "Ask about the meeting…",
        key="recall_chat_input",
    )

    question = typed_question.strip() if typed_question else None

    if not question and st.session_state.question_to_ask:
        question = st.session_state.question_to_ask
        st.session_state.question_to_ask = None

    if question:
        st.session_state.chat_history.append({
            "role": "user",
            "content": question,
        })

        with st.spinner("RECALL is searching the meeting…"):
            rag_response = ask_question_structured(
                r["rag_chain"],
                r["retriever_chain"],
                question,
            )

        st.session_state.chat_history.append({
            "role": "assistant",
            "answer": rag_response.answer,
            "evidence": rag_response.evidence or [],
        })

        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

if analyze_btn:
    if not source or not source.strip():
        st.error("Please enter a YouTube URL or upload a meeting file.")
    else:
        st.session_state.result = None
        st.session_state.chat_history = []
        st.session_state.pipeline_done = False
        st.session_state.question_to_ask = None

        progress = st.empty()

        try:
            ui("""
            <div class="surface" style="padding:1.25rem 1.4rem;margin-top:1rem;">
                <div class="small-label">Processing</div>
                <div style="font-family:'Space Grotesk';font-size:1.15rem;font-weight:600;margin-top:.35rem;">
                    Turning the recording into searchable intelligence
                </div>
            </div>
            """)

            def show_pipeline(label: str, sub: str) -> None:
                progress.markdown(
                    f"""
                    <div class="pipeline-status">
                        <span class="pipeline-orb"></span>
                        <div class="pipeline-copy">
                            <span class="pipeline-label">{esc(label)}</span>
                            <span class="pipeline-sub">{esc(sub)}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            show_pipeline(
                "Preparing the recording",
                "Loading audio and creating transcript chunks",
            )
            chunks, cleanup = process_input(source)

            try:
                stages = {
                    "transcription": (
                        "Transcribing your meeting",
                        "Turning speech into searchable text",
                    ),
                    "title": (
                        "Finding the meeting title",
                        "Understanding the central topic",
                    ),
                    "summary": (
                        "Building the executive brief",
                        "Condensing the conversation into key ideas",
                    ),
                    "extraction": (
                        "Extracting meeting intelligence",
                        "Finding decisions, actions and open questions",
                    ),
                }

                def on_progress(stage: str) -> None:
                    if stage in stages:
                        show_pipeline(*stages[stage])

                meeting, summary = build_meeting(
                    chunks=chunks,
                    source=source,
                    language=language,
                    on_progress=on_progress,
                )
            finally:
                cleanup()

            show_pipeline(
                "Indexing the meeting",
                "Making the transcript searchable for Ask RECALL",
            )
            rag_chain, retriever_chain = build_rag_chains_from_meeting(meeting)

            st.session_state.result = {
                "title": meeting.title,
                "transcript": meeting.plain_transcript(),
                "summary": summary,
                "action_items": _format_action_items(meeting.action_items),
                "key_decisions": _format_decisions(meeting.decisions),
                "open_questions": _format_questions(meeting.open_questions),
                "rag_chain": rag_chain,
                "retriever_chain": retriever_chain,
                "meeting": meeting,
            }
            st.session_state.pipeline_done = True

            progress.markdown(
                """
                <div class="pipeline-status">
                    <span class="pipeline-orb" style="background:var(--green);box-shadow:0 0 16px rgba(85,214,163,.45);animation:none;"></span>
                    <div class="pipeline-copy">
                        <span class="pipeline-label">Analysis complete</span>
                        <span class="pipeline-sub">Your meeting is ready to explore</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            time.sleep(.45)
            progress.empty()
            st.rerun()

        except Exception as exc:
            progress.markdown(
                f"""
                <div class="pipeline-status" style="border-color:rgba(255,102,116,.18);">
                    <span class="pipeline-orb" style="background:var(--red);box-shadow:0 0 16px rgba(255,102,116,.35);animation:none;"></span>
                    <div class="pipeline-copy">
                        <span class="pipeline-label">Analysis couldn't be completed</span>
                        <span class="pipeline-sub">{esc(str(exc)[:180])}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.session_state.pipeline_done = False
