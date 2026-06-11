"""
URL-driven router — keeps st.session_state.page and st.query_params in sync.

Why this exists:
    Streamlit's session_state lives in server memory and is tied to the
    WebSocket session. If the browser disconnects for even 1-2 seconds,
    Streamlit Cloud often issues a new session ID on reconnect and the
    old state is gone — the user lands back on the home page.

    By keeping `page` (and the active `run_id`) in the URL query string,
    the URL itself becomes the source of truth. A reconnect that preserves
    the URL restores the page and lets us look up the in-progress run on disk.

Usage:
    from ui.router import set_page, restore_from_url, sync_to_url

    # At the top of app.py, BEFORE routing:
    restore_from_url()

    # Anywhere you previously did `st.session_state.page = "results"`:
    set_page("results")
    st.rerun()
"""

from __future__ import annotations

import re

import streamlit as st

VALID_PAGES = {"home", "intake", "pipeline", "results", "briefs"}

# IDs restored from the URL are untrusted input — they flow into filesystem
# paths (runs/<run_id>) — so they must match the minted-id shape exactly.
_RUN_ID_RE       = re.compile(r"^\d{8}_\d{6}_[a-z0-9_]{1,40}$")
_BRIEF_PILLAR_RE = re.compile(r"^[a-z0-9_\-]{1,80}$")


# ── URL → session_state ──────────────────────────────────────────────────────

def restore_from_url() -> None:
    """
    Read query params on every script run and reconcile with session_state.

    URL is treated as source of truth for `page` and `run_id`. If the URL
    says page=pipeline&run=20260524_120000_seo but session_state has lost
    those, this populates them so the rest of the app keeps working.
    """
    params = st.query_params

    # Page
    url_page = params.get("page")
    if url_page in VALID_PAGES:
        if st.session_state.get("page") != url_page:
            st.session_state.page = url_page
    elif "page" not in st.session_state:
        st.session_state.page = "home"

    # Run id (in-progress or completed pipeline run).
    # Reject anything that doesn't look like a minted run id — this is the
    # security boundary against path traversal via ?run=../../...
    url_run = params.get("run")
    if url_run and _RUN_ID_RE.fullmatch(url_run):
        if st.session_state.get("active_run_id") != url_run:
            st.session_state.active_run_id = url_run

    # Brief target pillar id (so brief generation survives reconnect)
    url_brief_pillar = params.get("brief_pillar")
    if url_brief_pillar and _BRIEF_PILLAR_RE.fullmatch(url_brief_pillar):
        if st.session_state.get("brief_target_pillar_id") != url_brief_pillar:
            st.session_state.brief_target_pillar_id = url_brief_pillar


# ── session_state → URL ──────────────────────────────────────────────────────

def sync_to_url() -> None:
    """Mirror critical session_state values back into the URL."""
    page = st.session_state.get("page", "home")
    if st.query_params.get("page") != page:
        st.query_params["page"] = page

    run_id = st.session_state.get("active_run_id")
    if run_id:
        if st.query_params.get("run") != run_id:
            st.query_params["run"] = run_id
    else:
        if "run" in st.query_params:
            del st.query_params["run"]

    brief_pillar = st.session_state.get("brief_target_pillar_id")
    if brief_pillar:
        if st.query_params.get("brief_pillar") != brief_pillar:
            st.query_params["brief_pillar"] = brief_pillar
    else:
        if "brief_pillar" in st.query_params:
            del st.query_params["brief_pillar"]


# ── Navigation helper ────────────────────────────────────────────────────────

def set_page(page: str, run_id: str | None = None, brief_pillar: str | None = None) -> None:
    """
    Set the active page (and optionally run_id / brief_pillar) and mirror
    everything to the URL. Caller is responsible for calling st.rerun().
    """
    if page not in VALID_PAGES:
        raise ValueError(f"Unknown page: {page}")

    st.session_state.page = page
    st.query_params["page"] = page

    if run_id is not None:
        st.session_state.active_run_id = run_id
        st.query_params["run"] = run_id

    if brief_pillar is not None:
        st.session_state.brief_target_pillar_id = brief_pillar
        st.query_params["brief_pillar"] = brief_pillar


def clear_run() -> None:
    """Remove the active run from both session_state and URL."""
    st.session_state.pop("active_run_id", None)
    if "run" in st.query_params:
        del st.query_params["run"]
