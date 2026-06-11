"""
Sidebar — Session History panel.

Shows all saved sessions with quick-load buttons.
Rendered on every page via app.py.
"""

import html

import streamlit as st
from pathlib import Path

from ui.router import set_page, clear_run


def render_sidebar():
    """Render the session history sidebar."""
    with st.sidebar:
        st.markdown("## 🗂️ Session History")
        st.markdown("---")

        try:
            from ui.session_manager import list_sessions, load_session, delete_session, get_session_output_dir
            sessions = list_sessions()
        except Exception:
            sessions = []

        if not sessions:
            st.markdown(
                "<p style='color:#6b6b8a; font-size:0.85rem;'>"
                "No saved sessions yet.<br>Run a topical map to save it here."
                "</p>",
                unsafe_allow_html=True,
            )
            return

        st.markdown(f"<p style='color:#6b6b8a; font-size:0.8rem;'>{len(sessions)} saved maps</p>", unsafe_allow_html=True)

        for s in sessions:
            session_id  = s.get("session_id", "")
            seed        = s.get("seed", "Unknown")
            central     = s.get("central", "")
            created     = s.get("created_at", "")[:10]
            stats       = s.get("stats", {})
            pillars     = stats.get("pillars", 0)
            clusters    = stats.get("clusters", 0)

            # Card for each session
            with st.container():
                st.markdown(
                    f"<div style='background:#13131a; border:1px solid #1e1e2e; "
                    f"border-radius:8px; padding:0.7rem; margin-bottom:0.5rem;'>"
                    f"<div style='font-size:0.85rem; font-weight:500; color:#e8e8f0; "
                    f"white-space:nowrap; overflow:hidden; text-overflow:ellipsis;'>"
                    f"{html.escape(seed[:35])}</div>"
                    f"<div style='font-size:0.72rem; color:#6b6b8a; margin-top:0.2rem;'>"
                    f"{html.escape(str(created))} · {pillars}P · {clusters}C</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                col1, col2 = st.columns([3, 1])
                with col1:
                    if st.button(
                        "📂 Load",
                        key=f"load_{session_id}",
                        use_container_width=True,
                    ):
                        output, meta = load_session(session_id)
                        if output:
                            st.session_state.output     = output
                            st.session_state.output_dir = get_session_output_dir(session_id)
                            # Loading a saved session — drop any active in-progress run
                            clear_run()
                            set_page("results")
                            st.success(f"Loaded: {seed[:30]}")
                            st.rerun()
                        else:
                            st.error("Failed to load session.")

                with col2:
                    if st.button(
                        "🗑️",
                        key=f"del_{session_id}",
                        use_container_width=True,
                        help="Delete this session",
                    ):
                        delete_session(session_id)
                        st.rerun()

        st.markdown("---")
        st.markdown(
            "<p style='color:#6b6b8a; font-size:0.75rem;'>"
            "Sessions saved locally.<br>Last 50 runs kept."
            "</p>",
            unsafe_allow_html=True,
        )

        # Upload saved JSON to reload
        st.markdown("---")
        st.markdown("**📤 Load from file**")
        uploaded = st.file_uploader(
            "Upload topical_map.json",
            type=["json"],
            key="session_upload",
            label_visibility="collapsed",
        )
        if uploaded:
            try:
                import json
                from models import EngineOutput
                data = json.loads(uploaded.read())
                output = EngineOutput.model_validate(data)
                seed = output.input.seed_keyword
                st.session_state.output     = output
                st.session_state.output_dir = "uploaded_session"
                clear_run()
                set_page("results")
                st.success(f"Loaded: {seed[:30]}")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to load: {e}")
