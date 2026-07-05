"""
Briefs page — async brief generation with progress tracking.

Uses threading to avoid Streamlit Cloud timeout.
Progress is shown via polling — user sees real-time updates.

Logging uses an on_log callback into the shared state dict — the previous
implementation monkey-patched builtins.print process-wide, which raced
between concurrent sessions and could leave print broken after overlap.

Generated briefs are checkpointed to disk per page (see brief_batch), so a
reconnect mid-batch only costs the brief that was in flight: clicking
Generate again skips every brief that already finished.
"""

import html
import time
import threading
import streamlit as st
from pathlib import Path

from ui.router import set_page


# ── Background worker ─────────────────────────────────────────────────────────

def _load_serp_context(output_dir: str, pillar_id: str) -> str:
    """Load persisted real SERP data for this pillar, if the pipeline saved it."""
    try:
        from stages.serp import load_serp_data, serp_data_to_summary
        serp_data = load_serp_data(Path(output_dir) / "serp_data.json")
        if serp_data and pillar_id in serp_data:
            return serp_data_to_summary(serp_data[pillar_id])
    except Exception:
        pass
    return ""


def _run_brief_generation(
    pillar_id: str,
    max_clusters: int,
    output,
    output_dir: str,
    state: dict,
) -> None:
    """Runs in a background thread. Updates state dict for UI polling."""
    try:
        from stages.brief_batch import run_batch_for_pillar

        # Find target pillar
        pillar = next(
            (p for p in output.topical_map.pillars if p.id == pillar_id),
            None,
        )
        if not pillar:
            state["error"] = f"Pillar {pillar_id} not found"
            state["done"]  = True
            return

        state["status"] = f"Generating pillar brief: {pillar.title[:50]}..."

        def on_log(msg: str) -> None:
            state["logs"].append(msg)
            state["status"] = msg

        serp_context = _load_serp_context(output_dir, pillar_id)
        if serp_context:
            on_log("Real SERP data loaded — briefs will use live PAA/competitor data")

        briefs_dir = Path(output_dir) / "briefs"
        package = run_batch_for_pillar(
            pillar=pillar,
            topical_map=output.topical_map,
            output_dir=briefs_dir,
            include_clusters=max_clusters > 0,
            max_clusters=max_clusters,
            delay_between_calls=0.5,
            auto_correct_ids=True,
            serp_context=serp_context,
            on_log=on_log,
        )
        state["package"]  = package
        state["done"]     = True
        state["status"]   = f"Done! {package.total_generated} briefs generated."

    except Exception as e:
        import traceback
        state["error"] = str(e)
        state["trace"] = traceback.format_exc()
        state["done"]  = True


# ── Main render ───────────────────────────────────────────────────────────────

def render_briefs():
    output     = st.session_state.get("output")
    output_dir = st.session_state.get("output_dir", "streamlit_output")

    if not output:
        # Reconnect fallback — reload from disk via the run id in the URL.
        try:
            from ui.results import _restore_output_from_disk
            output, restored_dir = _restore_output_from_disk()
            if output:
                st.session_state.output = output
                if restored_dir:
                    st.session_state.output_dir = restored_dir
                    output_dir = restored_dir
        except Exception:
            pass

    if not output:
        st.warning("No topical map found. Generate one first.")
        if st.button("← Back to home"):
            set_page("home")
            st.rerun()
        return

    tm = output.topical_map

    col_nav, col_title = st.columns([1, 5])
    with col_nav:
        if st.button("← Results"):
            set_page("results")
            st.rerun()
    with col_title:
        st.markdown("## 📝 Content Brief Generator")
        st.markdown("Generate full content briefs for any pillar and its clusters.")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Pillar selector ───────────────────────────────────────────────────────
    pillar_map = {f"P{p.priority} — {p.title}": p for p in tm.pillars}
    preselected = st.session_state.get("brief_target_pillar_id")

    default_idx = 0
    if preselected:
        for i, p in enumerate(tm.pillars):
            if p.id == preselected:
                default_idx = i
                break

    selected_label = st.selectbox(
        "Select pillar",
        list(pillar_map.keys()),
        index=default_idx,
    )
    pillar = pillar_map[selected_label]

    col1, col2 = st.columns(2)
    with col1:
        max_clusters = st.slider(
            "Number of cluster briefs",
            min_value=0,
            max_value=min(len(pillar.clusters), 5),
            value=min(2, len(pillar.clusters)),
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        total_briefs = 1 + max_clusters
        est_cost     = 0.12 + max_clusters * 0.10
        est_time     = total_briefs * 25
        st.markdown(
            f"**{total_briefs} briefs** · ~${est_cost:.2f} · ~{est_time}s"
        )

    if max_clusters > 0:
        # Preview must match the batch's actual processing order
        # (BOFU-first, strongest signal first) — not raw map order.
        from stages.brief_batch import ordered_clusters
        st.markdown("**Clusters that will get briefs:**")
        for c in ordered_clusters(pillar)[:max_clusters]:
            st.markdown(f"  • {c.title}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── State management ──────────────────────────────────────────────────────
    # Keyed by pillar only: including max_clusters meant moving the slider
    # mid-run orphaned the running job's state and re-enabled Generate,
    # allowing a duplicate concurrent batch on the same checkpoint dir.
    brief_key    = f"brief_state_{pillar.id}"
    package_key  = f"brief_package_{pillar.id}"

    # ── Generate button ───────────────────────────────────────────────────────
    col_btn, col_cancel = st.columns([3, 1])
    with col_btn:
        generate_clicked = st.button(
            "🚀 Generate Briefs",
            use_container_width=True,
            disabled=bool(st.session_state.get(brief_key + "_running")),
        )
    with col_cancel:
        if st.session_state.get(brief_key + "_running"):
            if st.button("⏹ Stop", use_container_width=True):
                st.session_state[brief_key + "_running"] = False
                st.rerun()

    if generate_clicked:
        # Initialize state
        state = {
            "done":    False,
            "error":   None,
            "trace":   None,
            "status":  "Starting...",
            "logs":    [],
            "package": None,
        }
        st.session_state[brief_key]           = state
        st.session_state[brief_key + "_running"] = True

        # Launch background thread
        thread = threading.Thread(
            target=_run_brief_generation,
            args=(pillar.id, max_clusters, output, output_dir, state),
            daemon=True,
        )
        thread.start()
        st.rerun()

    # ── Progress display ──────────────────────────────────────────────────────
    state = st.session_state.get(brief_key)

    if state and st.session_state.get(brief_key + "_running"):
        if not state.get("done"):
            # Spinning animation + progress
            status = state.get("status", "Working...")
            logs   = state.get("logs", [])
            # Batch log lines for a finished brief are "  Done — ..." or
            # "  Resumed from checkpoint ..." — count those.
            done_count = sum(
                1 for l in logs
                if l.strip().lower().startswith("done —")
                or "resumed from checkpoint" in l.lower()
            )
            total_count = 1 + max_clusters

            st.markdown(f"""
<div style="display:flex; align-items:center; gap:1rem; padding:1rem;
            background:#13131a; border:1px solid #1e1e2e; border-radius:12px; margin-bottom:1rem;">
    <div style="width:48px; height:48px; flex-shrink:0;">
        <svg viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg">
            <style>
                .spin {{ animation: rotate 1.2s linear infinite; transform-origin: 24px 24px; }}
                @keyframes rotate {{ from {{ transform: rotate(0deg); }} to {{ transform: rotate(360deg); }} }}
            </style>
            <circle cx="24" cy="24" r="20" fill="none" stroke="#1e1e2e" stroke-width="4"/>
            <path class="spin" d="M24 4 A20 20 0 0 1 44 24" fill="none" stroke="#6c63ff" stroke-width="4" stroke-linecap="round"/>
        </svg>
    </div>
    <div style="flex:1;">
        <div style="font-size:0.9rem; font-weight:500; color:#e8e8f0; margin-bottom:0.3rem;">
            Generating briefs... {done_count}/{total_count}
        </div>
        <div style="background:#1e1e2e; border-radius:4px; height:6px; margin-bottom:0.4rem;">
            <div style="background:#6c63ff; height:6px; border-radius:4px;
                        width:{min(100, int(done_count/total_count*100))}%; transition:width 0.5s;"></div>
        </div>
        <div style="font-size:0.78rem; color:#6b6b8a; font-family:DM Mono,monospace;
                    white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
            {html.escape(status[:80])}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

            if logs:
                log_html = (
                    "<div class='log-box' style='max-height:180px;'>"
                    + "<br>".join(
                        f"<span style='color:{'#43e97b' if 'done' in l.lower() or 'saved' in l.lower() else '#6b6b8a'}'>{html.escape(l)}</span>"
                        for l in logs[-12:]
                    )
                    + "</div>"
                )
                st.markdown(log_html, unsafe_allow_html=True)

            # Auto-refresh every 2 seconds
            time.sleep(2)
            st.rerun()

        else:
            # Done
            st.session_state[brief_key + "_running"] = False

            if state.get("error"):
                st.error(f"❌ Failed: {state['error']}")
                if state.get("trace"):
                    with st.expander("Error details"):
                        st.code(state["trace"])
            else:
                package = state.get("package")
                if package:
                    st.session_state[package_key] = package
                    st.success(f"✅ {package.total_generated} briefs generated!")

    # ── Show results ──────────────────────────────────────────────────────────
    package = st.session_state.get(package_key)
    if package and package.total_generated > 0:
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown(f"### Download Briefs")

        # Validation summary
        if package.validation:
            total_issues = sum(len(r.issues) for r in package.validation.values())
            if total_issues == 0:
                st.success(f"✅ All {package.total_generated} briefs validated — no broken page IDs.")
            else:
                st.warning(f"⚠️ {total_issues} broken IDs auto-corrected.")

        # Download buttons
        for path in package.get_markdown_paths():
            if path.exists() and not path.name.startswith("_"):
                st.download_button(
                    label=f"📄 {path.stem.replace('brief_', '')}",
                    data=path.read_text(encoding="utf-8"),
                    file_name=path.name,
                    mime="text/markdown",
                    key=f"dl_{path.stem}",
                )

        # JSON bundle
        briefs_dir = Path(output_dir) / "briefs"
        json_path  = briefs_dir / "all_briefs.json"
        if json_path.exists():
            st.download_button(
                label="📦 all_briefs.json",
                data=json_path.read_text(encoding="utf-8"),
                file_name="all_briefs.json",
                mime="application/json",
                key="dl_all_briefs",
            )

        # CSV + DOCX bundles
        try:
            from stages.brief_export import briefs_to_csv, briefs_to_docx

            pillar_slug = (package.pillar_id or "briefs").replace("/", "_")

            csv_bytes = briefs_to_csv(package.briefs)
            st.download_button(
                label="📊 all_briefs.csv",
                data=csv_bytes,
                file_name=f"{pillar_slug}_briefs.csv",
                mime="text/csv",
                key="dl_all_briefs_csv",
            )

            try:
                docx_bytes = briefs_to_docx(
                    package.briefs,
                    title=f"Content Briefs — {package.pillar_title}",
                )
                st.download_button(
                    label="📄 all_briefs.docx",
                    data=docx_bytes,
                    file_name=f"{pillar_slug}_briefs.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    key="dl_all_briefs_docx",
                )
            except RuntimeError as e:
                st.warning(f"DOCX export unavailable: {e}")
        except Exception as e:
            st.warning(f"CSV/DOCX export failed: {e}")

        # Preview
        st.markdown("<br>", unsafe_allow_html=True)
        for page_id, brief in package.briefs.items():
            with st.expander(f"📄 {brief.page_title}"):
                st.markdown(f"**Information Gain:** {brief.information_gain_angle}")
                st.markdown(f"**Journey Stage:** `{brief.queries.journey_stage}`")
                st.markdown(f"**Word Count:** {brief.content_specs.recommended_word_count:,}")
                st.markdown(f"**Primary Query:** `{brief.queries.primary_query}`")
                st.markdown("")
                st.markdown("**Heading Structure:**")
                for h in brief.headings[:8]:
                    depth = int(h.level[1]) - 1 if len(h.level) > 1 and h.level[1].isdigit() else 0
                    indent = "&nbsp;" * depth * 4
                    st.markdown(f"{indent}**{h.level}:** {h.text}", unsafe_allow_html=True)
                if brief.semantic_bridges:
                    st.markdown("")
                    st.markdown(f"**Semantic Bridges ({len(brief.semantic_bridges)}):**")
                    for b in brief.semantic_bridges[:3]:
                        strength = float(b.relationship_strength) if b.relationship_strength else 0.0
                        st.markdown(f"  • [{strength:.2f}] → `{b.link_destination}`")
