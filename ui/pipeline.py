"""
Pipeline page — runs the engine in a background thread, polls progress
from disk, and resumes interrupted runs.

Architecture
------------
The pipeline is a long-running blocking call (~minutes). Streamlit's main
script thread cannot block on it without freezing the UI, and a network
blip will sever the WebSocket. To survive both:

  1. The actual run executes in a `threading.Thread` (daemon) inside the
     Streamlit server process. It writes progress + checkpoints to disk
     under `runs/<run_id>/`.

  2. The Streamlit UI polls `runs/<run_id>/progress.json` every 2 seconds
     and re-renders. Polling is cheap and survives WS blips because the
     URL preserves `run_id` (see ui/router.py).

  3. If the worker thread dies (page reload, Streamlit Cloud session
     reset) we detect on next entry that progress.status == "running"
     but no thread is alive in this process — and we re-spawn it. The
     refactored `pipeline.run_pipeline()` checks for per-stage
     checkpoints and skips work that finished before the crash, so the
     repeat is cheap.

Resume policy
-------------
  * If a run is `completed`, load the EngineOutput and jump to results.
  * If a run is `running_active` (heartbeat within 2 min): keep polling
    or re-spawn the worker if missing.
  * If a run is `running_stale` (>2 min since last heartbeat): ask the
    user — Resume (which will re-spawn the worker and continue from the
    last checkpoint) or Start Fresh.
  * If a run is `failed`: show the error and offer Restart.
"""

from __future__ import annotations

import html
import os
import threading
import time
import traceback
from pathlib import Path

import streamlit as st

from ui import run_state
from ui.router import set_page, clear_run

# Load .env BEFORE any API-key checks — previously the key check ran first,
# so keys that lived only in a .env file were incorrectly reported missing.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ── Background worker ─────────────────────────────────────────────────────────

# Track currently-running threads in this Python process so we don't spawn
# duplicates on every poll-rerun. Keyed by run_id.
_WORKERS: dict[str, threading.Thread] = {}
_WORKER_LOCK = threading.Lock()


def _worker_alive(run_id: str) -> bool:
    with _WORKER_LOCK:
        t = _WORKERS.get(run_id)
        return bool(t and t.is_alive())


def _safe_print(*args, sep: str = " ", end: str = "\n", file=None, flush: bool = False):
    """
    A drop-in `print` replacement that writes only to the original stdout
    captured by Python at interpreter start (sys.__stdout__). Safe to call
    from any thread because it never touches Streamlit UI.
    """
    import sys as _sys
    target = file if file is not None else _sys.__stdout__
    try:
        target.write(sep.join(str(a) for a in args) + end)
        if flush:
            target.flush()
    except Exception:
        pass


def _reset_builtin_print() -> None:
    """
    Forcibly restore a safe `builtins.print`. Older versions of this UI
    monkey-patched `builtins.print` to call Streamlit UI helpers so that
    pipeline log output appeared inside the page. That patch leaks across
    code reloads on Streamlit Cloud (same Python process), and from a
    background thread the Streamlit calls raise NoSessionContext. We
    install a thread-safe replacement once per worker spawn.
    """
    import builtins
    builtins.print = _safe_print


def _spawn_worker(run_id: str, seed, settings: dict) -> None:
    """Start the pipeline in a background thread. Idempotent per run_id."""
    with _WORKER_LOCK:
        existing = _WORKERS.get(run_id)
        if existing and existing.is_alive():
            return

        # Clean up any stale print monkey-patch before launching the worker.
        # Do this in both the parent and child threads to be safe.
        _reset_builtin_print()

        # A fresh worker means fresh intent to run — drop any stale cancel flag.
        run_state.clear_cancel(run_id)

        def _target():
            # Background thread — make absolutely sure print is harmless here.
            _reset_builtin_print()
            try:
                from pipeline import run_pipeline, PipelineCancelled
                run_pipeline(
                    seed=seed,
                    output_dir=run_state.run_dir(run_id),
                    skip_serp=settings.get("skip_serp", False),
                    skip_validation=settings.get("skip_validation", False),
                    serp_geo=settings.get("serp_geo", "us"),
                    serp_lang=settings.get("serp_lang", "en"),
                    run_id=run_id,
                )
            except PipelineCancelled:
                run_state.mark_run_failed(run_id, "Cancelled by user")
            except Exception as e:
                run_state.mark_run_failed(run_id, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

        thread = threading.Thread(target=_target, daemon=True, name=f"pipeline-{run_id}")
        thread.start()
        _WORKERS[run_id] = thread


# ── Helpers ───────────────────────────────────────────────────────────────────

STAGE_LABELS = {
    "stage2":   "Central entity",
    "stage3":   "Pillars & clusters",
    "stage3_5": "SERP intelligence",
    "stage4":   "Topic validation",
    "stage5":   "Query generation",
    "stage6":   "Supplementary nodes",
    "stage7":   "Internal linking",
    "stage8":   "Render outputs",
}


def _check_api_keys(skip_serp: bool = False) -> list[str]:
    # GEMINI_API_KEY is required too: Stage 6 (supplementary) runs on Gemini
    # Flash, and Stage 4 falls back to Gemini when SERP is skipped.
    # SERPER_API_KEY is only needed when the SERP pull is enabled.
    required = ["ANTHROPIC_API_KEY", "GEMINI_API_KEY"]
    if not skip_serp:
        required.append("SERPER_API_KEY")
    return [key for key in required if not os.environ.get(key)]


def _load_completed_output(run_id: str):
    """Load the final EngineOutput for a completed run. Returns (output, output_dir)."""
    try:
        from ui.session_manager import load_session, get_session_output_dir
        output, _meta = load_session(run_id)
        if output:
            return output, get_session_output_dir(run_id)
    except Exception:
        pass
    # Fallback: try loading directly from the run dir
    try:
        import json as _json
        from models import EngineOutput
        path = run_state.run_dir(run_id) / "topical_map.json"
        if path.exists():
            data = _json.loads(path.read_text(encoding="utf-8"))
            return EngineOutput.model_validate(data), str(run_state.run_dir(run_id))
    except Exception:
        pass
    return None, None


def _render_progress_panel(progress: dict, run_id: str) -> None:
    """Visual progress: completed stages, current stage, last message."""
    completed = set(progress.get("completed_stages", []))
    total     = len(STAGE_LABELS)
    done      = len(completed)
    pct       = int(100 * done / total) if total else 0

    st.markdown(f"""
<div style="background:#13131a; border:1px solid #1e1e2e; border-radius:12px;
            padding:1.2rem; margin-bottom:1rem;">
    <div style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom:0.6rem;">
        <div style="font-size:1rem; font-weight:500; color:#e8e8f0;">
            Pipeline progress
        </div>
        <div style="font-family:DM Mono,monospace; color:#6c63ff; font-size:0.95rem;">
            {done} / {total} stages
        </div>
    </div>
    <div style="background:#1e1e2e; border-radius:6px; height:8px; margin-bottom:0.8rem;">
        <div style="background:#6c63ff; height:8px; border-radius:6px;
                    width:{pct}%; transition:width 0.5s;"></div>
    </div>
    <div style="font-size:0.78rem; color:#6b6b8a; font-family:DM Mono,monospace;
                white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
        {html.escape(progress.get("message", "")[:120])}
    </div>
</div>
""", unsafe_allow_html=True)

    # Stage checklist
    cols = st.columns(4)
    for i, (sid, label) in enumerate(STAGE_LABELS.items()):
        with cols[i % 4]:
            if sid in completed:
                icon, color = "✅", "#43e97b"
            elif sid == progress.get("last_stage"):
                icon, color = "⏳", "#6c63ff"
            else:
                icon, color = "○", "#6b6b8a"
            st.markdown(
                f"<div style='font-size:0.85rem; color:{color}; margin-bottom:0.3rem;'>"
                f"{icon} {label}</div>",
                unsafe_allow_html=True,
            )


# ── Main render ───────────────────────────────────────────────────────────────

def render_pipeline():
    st.markdown("## 🔄 Generating Topical Map")

    # API key check — respect the skip_serp setting (from the intake form or,
    # on resume, from the run's saved settings).
    _run_id_for_settings = st.session_state.get("active_run_id")
    _settings = st.session_state.get("pipeline_settings")
    if _settings is None and _run_id_for_settings:
        _settings = run_state.load_settings(_run_id_for_settings)
    _settings = _settings or {}
    missing_keys = _check_api_keys(skip_serp=_settings.get("skip_serp", False))
    if missing_keys:
        st.error(f"Missing API keys: {', '.join(missing_keys)}")
        st.info("Set them in your environment or a `.env` file before running.")
        if st.button("← Back to home"):
            set_page("home")
            st.rerun()
        return

    run_id = st.session_state.get("active_run_id")

    # ── No active run: must come from intake with seed_input in session ──────
    if not run_id:
        seed = st.session_state.get("seed_input")
        settings = st.session_state.get("pipeline_settings", {})
        if not seed:
            st.error("No seed input found. Please start from the intake form.")
            if st.button("← Back to form"):
                set_page("intake")
                st.rerun()
            return

        # Mint a fresh run_id, persist seed+settings, kick off worker
        run_id = run_state.new_run_id(seed.seed_keyword)
        run_state.save_seed(run_id, seed)
        run_state.save_settings(run_id, settings)
        run_state.init_progress(run_id)
        st.session_state.active_run_id = run_id
        st.query_params["run"] = run_id
        _spawn_worker(run_id, seed, settings)
        st.rerun()
        return

    # ── Active run exists: figure out its state ──────────────────────────────
    status = run_state.run_status(run_id)

    # Show header
    progress = run_state.read_progress(run_id) or {}
    seed_obj = run_state.load_seed(run_id)
    if seed_obj:
        st.markdown(f"**Seed:** `{seed_obj.seed_keyword}`")
    st.markdown(f"**Run ID:** `{run_id}` &nbsp;•&nbsp; **Status:** `{status}`",
                unsafe_allow_html=True)
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── State: missing ───────────────────────────────────────────────────────
    if status == "missing":
        st.error(f"Run `{run_id}` not found on disk. It may have been deleted.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Start a new run"):
                clear_run()
                set_page("intake")
                st.rerun()
        with col2:
            if st.button("Home"):
                clear_run()
                set_page("home")
                st.rerun()
        return

    # ── State: completed ─────────────────────────────────────────────────────
    if status == "completed":
        output, output_dir = _load_completed_output(run_id)
        if output:
            st.session_state.output = output
            st.session_state.output_dir = output_dir
            st.success("✅ Pipeline complete!")
            _render_progress_panel(progress, run_id)
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📊  View Results", use_container_width=True, type="primary"):
                    set_page("results", run_id=run_id)
                    st.rerun()
            with col2:
                if st.button("← Run Again (new map)", use_container_width=True):
                    clear_run()
                    st.session_state.pop("output", None)
                    set_page("intake")
                    st.rerun()
        else:
            st.warning("Run marked complete but output files are missing. Try resuming.")
            if st.button("🔄 Re-run from last checkpoint"):
                seed = run_state.load_seed(run_id)
                settings = run_state.load_settings(run_id)
                if seed:
                    _spawn_worker(run_id, seed, settings)
                st.rerun()
        return

    # ── State: failed ────────────────────────────────────────────────────────
    if status == "failed":
        st.error("Pipeline failed.")
        err = progress.get("error", "")
        if err:
            with st.expander("Error details", expanded=True):
                st.code(err)
        _render_progress_panel(progress, run_id)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Retry from last checkpoint", use_container_width=True):
                seed = run_state.load_seed(run_id)
                settings = run_state.load_settings(run_id)
                if seed:
                    # Clear failed status — set back to running before spawning
                    run_state.heartbeat(run_id, "Retrying from last checkpoint...")
                    progress["status"] = "running"
                    progress["error"] = None
                    run_state._write_progress(run_id, progress)
                    _spawn_worker(run_id, seed, settings)
                st.rerun()
        with col2:
            if st.button("← Start a fresh run", use_container_width=True):
                clear_run()
                set_page("intake")
                st.rerun()
        return

    # ── State: running ───────────────────────────────────────────────────────
    # No time limit on resume. If the worker thread is gone (e.g. process
    # restart, network drop that killed the script, or a silent crash),
    # we re-spawn it automatically from the last completed checkpoint —
    # unless a cancel is pending (respawning would clear the cancel flag).
    cancel_pending = run_state.cancel_requested(run_id)
    worker_was_dead = not _worker_alive(run_id)
    if worker_was_dead and not cancel_pending:
        seed = run_state.load_seed(run_id)
        settings = run_state.load_settings(run_id)
        if seed:
            run_state.heartbeat(run_id, "Auto-resuming from last checkpoint...")
            _spawn_worker(run_id, seed, settings)
            progress = run_state.read_progress(run_id) or progress

    if cancel_pending:
        if worker_was_dead:
            # Worker already gone — finalize the cancellation ourselves.
            run_state.clear_cancel(run_id)
            run_state.mark_run_failed(run_id, "Cancelled by user")
            st.rerun()
        st.info("⏹ Cancel requested — the worker will stop after the current stage finishes.")

    # Surface how long since the worker last reported progress. Purely
    # informational — does NOT gate resume.
    last_ts = progress.get("last_updated_at")
    if last_ts:
        age = int(run_state._seconds_since(last_ts))
        if worker_was_dead:
            st.info(
                f"🔄 The previous worker had stopped ({age}s since last update). "
                f"Restarted automatically — picking up from the last completed checkpoint."
            )
        elif age > 30:
            st.caption(f"Last activity {age}s ago. Stage in progress may take a few minutes for LLM calls.")

    _render_progress_panel(progress, run_id)

    # Live log (last completed stage messages)
    completed = progress.get("completed_stages", [])
    if completed:
        log_html = "<div class='log-box'>" + "<br>".join(
            f"<span style='color:#43e97b'>✓ {STAGE_LABELS.get(s, s)}</span>"
            for s in completed
        ) + "</div>"
        st.markdown(log_html, unsafe_allow_html=True)

    # Manual controls — escape hatches if the worker really is wedged.
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄  Force restart worker", use_container_width=True,
                     help="Re-spawn the worker thread once the current one has stopped"):
            # Threads can't be killed. Spawning a second worker while the old
            # one is alive would run two pipelines on the same run dir, so:
            # if the old worker is alive, request cancel and wait; only spawn
            # once it's actually gone.
            if _worker_alive(run_id):
                run_state.request_cancel(run_id)
                run_state.heartbeat(run_id, "Stopping current worker before restart...")
                st.info("Current worker is still running — it will stop at the next "
                        "stage boundary. Click Force restart again once it has stopped.")
            else:
                seed = run_state.load_seed(run_id)
                settings = run_state.load_settings(run_id)
                if seed:
                    with _WORKER_LOCK:
                        _WORKERS.pop(run_id, None)
                    run_state.heartbeat(run_id, "Force-restarting worker...")
                    _spawn_worker(run_id, seed, settings)
                st.rerun()
    with col2:
        if st.button("⏹  Stop run", use_container_width=True):
            if _worker_alive(run_id):
                # Cooperative cancel: the pipeline checks the flag between
                # stages, marks the run failed, and stops spending API money.
                run_state.request_cancel(run_id)
                run_state.heartbeat(run_id, "Cancel requested...")
            else:
                run_state.mark_run_failed(run_id, "Cancelled by user")
            st.rerun()

    # Poll: rerun after 2s to refresh progress
    time.sleep(2)
    st.rerun()
