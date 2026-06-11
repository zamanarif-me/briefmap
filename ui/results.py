"""Results page — displays the topical map output."""

import json
import streamlit as st
from pathlib import Path

from ui.router import set_page


def render_results():
    output = st.session_state.get("output")
    output_dir = st.session_state.get("output_dir", "streamlit_output")

    if not output:
        st.warning("No results yet. Generate a topical map first.")
        if st.button("← Back to home"):
            set_page("home")
            st.rerun()
        return

    tm = output.topical_map
    lp = output.linking_plan

    # ── Header ────────────────────────────────────────────────────────────────
    col_nav, col_title = st.columns([1, 5])
    with col_nav:
        if st.button("← Home"):
            set_page("home")
            st.rerun()
    with col_title:
        st.markdown(f"## {tm.central_entity.primary}")
        st.markdown(f"*{tm.central_entity.source_context}*")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Stats row ─────────────────────────────────────────────────────────────
    total_clusters = sum(len(p.clusters) for p in tm.pillars)
    total_supp = sum(sum(len(c.supplementary_nodes) for c in p.clusters) for p in tm.pillars)
    total_queries = sum(
        len(p.representative_queries) + sum(len(c.represented_queries) for c in p.clusters)
        for p in tm.pillars
    )
    bridges = [l for l in lp.links if l.relationship.value == "entity_bridge"]

    metrics = [
        (len(tm.pillars), "Pillars"),
        (total_clusters, "Clusters"),
        (total_supp, "Supplementary Nodes"),
        (total_queries, "Queries"),
        (len(lp.links), "Internal Links"),
        (len(bridges), "Entity Bridges"),
    ]

    cols = st.columns(6)
    for col, (val, label) in zip(cols, metrics):
        with col:
            st.markdown(f"""
<div class="metric-card">
    <div class="metric-value">{val}</div>
    <div class="metric-label">{label}</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["🏛️ Pillars", "🔗 Entity Bridges", "🌍 Geo Pages", "📥 Export"])

    with tab1:
        _render_pillars(tm)

    with tab2:
        _render_bridges(bridges)

    with tab3:
        _render_geo(tm)

    with tab4:
        _render_export(output, output_dir)


def _render_pillars(tm):
    for p in tm.pillars:
        # Pillar card
        intent_cls = "commercial" if p.intent.value == "commercial" else "info"
        funnel_cls = p.funnel_stage.value.lower()
        priority_cls = f"p{p.priority}"
        val_icon = {"strong": "✅", "medium": "⚠️", "weak": "❌"}.get(p.validation_signal or "", "")

        with st.expander(
            f"P{p.priority} — {p.title}  {val_icon}",
            expanded=False,
        ):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**Commercial value:** {p.commercial_value}")
                if p.related_entities:
                    st.markdown(f"**Entities:** {', '.join(p.related_entities[:6])}")
                if p.representative_queries:
                    st.markdown(f"**Representative queries:**")
                    for q in p.representative_queries[:3]:
                        st.markdown(f"  • `{q.text}`")

            with col2:
                st.markdown(f"""
<span class="tag tag-{intent_cls}">{p.intent.value}</span><br><br>
<span class="tag tag-{funnel_cls}">{p.funnel_stage.value}</span><br><br>
<span class="tag tag-{priority_cls}">P{p.priority}</span>
""", unsafe_allow_html=True)
                if p.validation_signal:
                    st.markdown(f"**Signal:** {p.validation_signal}")

            st.markdown(f"**Clusters ({len(p.clusters)}):**")
            for c in p.clusters:
                supp_count = len(c.supplementary_nodes)
                query_count = len(c.represented_queries)
                angles = list(set(s.angle for s in c.supplementary_nodes if s.angle))
                angle_str = " · ".join(angles) if angles else ""

                st.markdown(
                    f"&nbsp;&nbsp;• **{c.title}** "
                    f"<span class='tag tag-{c.intent.value}'>{c.intent.value}</span> "
                    f"— {supp_count} supp · {query_count} queries"
                    + (f" · *{angle_str}*" if angle_str else ""),
                    unsafe_allow_html=True,
                )

            # Brief generator button
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button(f"📝 Generate Briefs for this Pillar", key=f"brief_{p.id}"):
                set_page("briefs", brief_pillar=p.id)
                st.rerun()


def _render_bridges(bridges):
    if not bridges:
        st.info("No entity bridges found.")
        return

    st.markdown(f"**{len(bridges)} cross-pillar entity bridges**")
    st.markdown("These are strategic links that build topical authority across pillars by reinforcing shared entities.")
    st.markdown("<br>", unsafe_allow_html=True)

    for b in bridges:
        strength = b.relationship_strength or 0.0
        strength_color = "#43e97b" if strength >= 0.85 else "#ffc107" if strength >= 0.65 else "#ff6b6b"
        st.markdown(
            f"<span style='color:{strength_color}; font-family: DM Mono, monospace;'>"
            f"[{strength:.2f}]</span> "
            f"`{b.from_page_id}` → `{b.to_page_id}`<br>"
            f"<span style='color:#6b6b8a; font-size:0.85rem;'>"
            f"Anchor: \"{b.anchor_text}\" · {b.reasoning}</span>",
            unsafe_allow_html=True,
        )
        st.markdown("")


def _render_geo(tm):
    if not tm.geo_pages:
        st.info("No geographic service pages generated.")
        return

    st.markdown(f"**{len(tm.geo_pages)} geographic service pages**")
    for g in tm.geo_pages:
        st.markdown(f"• **{g.title}** — {g.geography} ({g.parent_pillar_id})")


def _render_export(output, output_dir):
    st.markdown("### Download Files")

    out_path = Path(output_dir)

    # JSON
    json_path = out_path / "topical_map.json"
    if json_path.exists():
        st.download_button(
            label="📥 topical_map.json",
            data=json_path.read_text(encoding="utf-8"),
            file_name="topical_map.json",
            mime="application/json",
        )

    # Markdown report
    md_path = out_path / "topical_map_report.md"
    if md_path.exists():
        st.download_button(
            label="📄 topical_map_report.md",
            data=md_path.read_text(encoding="utf-8"),
            file_name="topical_map_report.md",
            mime="text/markdown",
        )

    # CSV (Koray format)
    csv_path = out_path / "topical_map.csv"
    if csv_path.exists():
        st.download_button(
            label="📊 topical_map.csv (Koray format — Excel ready)",
            data=csv_path.read_bytes(),
            file_name="topical_map.csv",
            mime="text/csv",
        )
    else:
        # Generate on-the-fly if not saved yet
        try:
            from stages.render import render_koray_csv
            csv_data = render_koray_csv(output)
            st.download_button(
                label="📊 topical_map.csv (Koray format — Excel ready)",
                data=csv_data.encode("utf-8-sig"),
                file_name="topical_map.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.warning(f"CSV generation failed: {e}")

    # Cost report
    cost_path = out_path / "cost_report.json"
    if cost_path.exists():
        cost_data = json.loads(cost_path.read_text(encoding="utf-8"))
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Cost Report")
        summary = cost_data.get("summary", {})
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Cost", f"${summary.get('total_cost_usd', 0):.4f}")
        with col2:
            st.metric("LLM Calls", summary.get("total_llm_calls", 0))
        with col3:
            st.metric("Serper Calls", summary.get("total_serper_calls", 0))

        st.download_button(
            label="📊 cost_report.json",
            data=cost_path.read_text(encoding="utf-8"),
            file_name="cost_report.json",
            mime="application/json",
        )
