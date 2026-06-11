"""
BriefMap — Streamlit UI

Koray-framework semantic topical maps, query networks, and content briefs.

Run with:
    streamlit run app.py

Requirements:
    pip install streamlit pydantic anthropic google-genai requests jinja2
"""

import streamlit as st

st.set_page_config(
    page_title="BriefMap",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:       #0a0a0f;
    --surface:  #13131a;
    --border:   #1e1e2e;
    --accent:   #6c63ff;
    --accent2:  #ff6b6b;
    --accent3:  #43e97b;
    --text:     #e8e8f0;
    --muted:    #6b6b8a;
    --card:     #16161f;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: var(--bg);
    color: var(--text);
}

.stApp { background: var(--bg); }

h1, h2, h3 {
    font-family: 'DM Serif Display', serif;
    color: var(--text);
}

.stButton > button {
    background: var(--accent);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    padding: 0.6rem 1.4rem;
    transition: all 0.2s;
}

.stButton > button:hover {
    background: #7c74ff;
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(108, 99, 255, 0.4);
}

.stTextInput > div > div > input,
.stSelectbox > div > div,
.stMultiselect > div > div {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
}

.stProgress > div > div {
    background: var(--accent);
}

.metric-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
}

.metric-value {
    font-family: 'DM Mono', monospace;
    font-size: 2rem;
    font-weight: 500;
    color: var(--accent);
}

.metric-label {
    font-size: 0.8rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.3rem;
}

.pillar-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.5rem;
    cursor: pointer;
    transition: border-color 0.2s;
}

.pillar-card:hover {
    border-color: var(--accent);
}

.tag {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    font-size: 0.72rem;
    font-weight: 500;
    font-family: 'DM Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.tag-commercial  { background: rgba(108, 99, 255, 0.2); color: #9c95ff; }
.tag-info        { background: rgba(67, 233, 123, 0.2); color: #43e97b; }
.tag-bofu        { background: rgba(255, 107, 107, 0.2); color: #ff6b6b; }
.tag-tofu        { background: rgba(67, 233, 123, 0.15); color: #43e97b; }
.tag-mofu        { background: rgba(255, 193, 7, 0.2); color: #ffc107; }
.tag-p1          { background: rgba(255, 107, 107, 0.25); color: #ff6b6b; }
.tag-p2          { background: rgba(255, 193, 7, 0.2); color: #ffc107; }
.tag-p3          { background: rgba(108, 99, 255, 0.2); color: #9c95ff; }

.log-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.82rem;
    color: var(--accent3);
    max-height: 300px;
    overflow-y: auto;
}

.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 3.5rem;
    line-height: 1.1;
    margin-bottom: 1rem;
}

.hero-subtitle {
    font-size: 1.15rem;
    color: var(--muted);
    max-width: 600px;
    line-height: 1.6;
}

.step-indicator {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 2rem;
}

.step-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--border);
}

.step-dot.active { background: var(--accent); }
.step-dot.done   { background: var(--accent3); }

.divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.5rem 0;
}

.stExpander {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# ── Router ────────────────────────────────────────────────────────────────────
from ui.home import render_home
from ui.intake import render_intake
from ui.pipeline import render_pipeline
from ui.results import render_results
from ui.briefs import render_briefs
from ui.sidebar import render_sidebar
from ui.router import restore_from_url, sync_to_url

# URL is the source of truth — restore page + run_id from query params
# before doing anything else. This lets the app survive WebSocket reconnects.
restore_from_url()

# Render sidebar on every page
render_sidebar()

page = st.session_state.page

if page == "home":
    render_home()
elif page == "intake":
    render_intake()
elif page == "pipeline":
    render_pipeline()
elif page == "results":
    render_results()
elif page == "briefs":
    render_briefs()

# Push current session_state back into URL so the next reconnect can restore
sync_to_url()
