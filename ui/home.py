"""Home page — hero + feature overview."""

import streamlit as st

from ui.router import set_page


def render_home():
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
<div class="hero-title">
    Brief<em>Map</em>
</div>
<div class="hero-subtitle">
    Generate Koray-framework semantic topical maps,
    query networks, and content briefs — powered by
    Anthropic, Serper, and Gemini.
</div>
""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_a, col_b = st.columns([1, 2])
        with col_a:
            if st.button("🗺️  New Project", use_container_width=True):
                # Fresh intake — drop any prior in-progress run id
                from ui.router import clear_run
                clear_run()
                st.session_state.pop("seed_input", None)
                st.session_state.pop("intake_step", None)
                st.session_state.pop("intake_data", None)
                set_page("intake")
                st.rerun()

        if st.session_state.get("output"):
            with col_b:
                if st.button("📊  View Last Results", use_container_width=True):
                    set_page("results")
                    st.rerun()

    with col2:
        st.markdown("""
<div style="display:flex; flex-direction:column; gap:0.8rem; padding-top:1rem;">

<div class="metric-card">
    <div class="metric-value">89%</div>
    <div class="metric-label">Koray Framework Coverage</div>
</div>

<div class="metric-card">
    <div class="metric-value">9</div>
    <div class="metric-label">Pipeline Stages</div>
</div>

<div class="metric-card">
    <div class="metric-value">~$1</div>
    <div class="metric-label">Cost per Topical Map</div>
</div>

<div class="metric-card">
    <div class="metric-value">8</div>
    <div class="metric-label">Export Formats</div>
</div>

</div>
""", unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Feature grid
    st.markdown("### What it builds")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
**🏛️ Topical Map**
Pillar / Cluster / Supplementary hierarchy with intent tagging, funnel stages, and Core/Outer section distinction.
""")
    with c2:
        st.markdown("""
**🔗 Query Network**
Representative + represented queries enriched with real PAA data from Serper SERP pulls.
""")
    with c3:
        st.markdown("""
**📝 Content Briefs**
Per-page briefs with information gain angles, entity attribute maps, heading structures, and semantic bridges.
""")

    c4, c5, c6 = st.columns(3)
    with c4:
        st.markdown("""
**🌐 Internal Linking**
Directed link graph with entity bridge weights. Pillar↔cluster links generated deterministically.
""")
    with c5:
        st.markdown("""
**✅ Web Validation**
Topics validated against real SERP data — strong / medium / weak signal per pillar.
""")
    with c6:
        st.markdown("""
**📊 Export — JSON · MD · CSV**
Full JSON + Markdown report. Koray-format CSV (Excel ready) with Contextual Vector / Hierarchy / Connection columns.
""")

    c7, c8, _ = st.columns(3)
    with c7:
        st.markdown("""
**🗂️ Session History**
Every run auto-saved. Reload any previous map from the sidebar — or upload a saved JSON to restore.
""")
    with c8:
        st.markdown("""
**💰 Cost Tracker**
Per-run cost report: LLM calls, token usage, Serper calls. Monthly projection included.
""")
