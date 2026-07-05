"""
Run State Manager — persists in-progress pipeline runs to disk so they can
survive WebSocket disconnects, browser reconnects, and Streamlit Cloud
session resets.

Layout:
    runs/<run_id>/
        seed.json          — full SeedInput, written before stage 2 starts
        settings.json      — pipeline settings (skip_serp, serp_geo, ...)
        progress.json      — {status, completed_stages[], started_at,
                              last_updated_at, error, last_stage, message}
        _checkpoint_<stage>.json   — written by pipeline after each stage
        topical_map.json / .csv / .md   — final outputs (written by render)
        cost_report.json   — final cost summary

The 2-minute grace window means: after a network blip, if last_updated_at
is within 120 seconds, we auto-resume. Past that, the user is asked.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from models import SeedInput


RUNS_DIR = Path("runs")
GRACE_WINDOW_SECONDS = 120          # 2 minutes — auto-resume window
STALE_AFTER_SECONDS = 24 * 3600     # 24 hours — runs older than this are pruned

# run_ids are minted by new_run_id() and must match this shape. The id also
# arrives from the URL query string (?run=...), so this pattern is a security
# boundary: it prevents path traversal (e.g. ?run=../../etc) from reaching
# the filesystem helpers below (including shutil.rmtree in delete_run).
RUN_ID_RE = re.compile(r"^\d{8}_\d{6}_[a-z0-9_]{1,40}$")


def is_valid_run_id(run_id: str) -> bool:
    return bool(run_id) and bool(RUN_ID_RE.fullmatch(run_id))

# Canonical stage order — pipeline must follow this sequence
STAGE_ORDER = [
    "stage2",      # Central entity
    "stage3",      # Pillars + clusters
    "stage3_5",    # SERP intelligence (optional)
    "stage4",      # Validation (optional)
    "stage5",      # Queries
    "stage6",      # Supplementary nodes
    "stage7",      # Linking plan
    "stage8",      # Render outputs
]


# ── ID + path helpers ────────────────────────────────────────────────────────

def _slugify(seed: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", seed.lower().strip())
    return slug[:40].strip("_") or "run"


def new_run_id(seed_keyword: str) -> str:
    """Generate a unique run id tied to a seed keyword."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{_slugify(seed_keyword)}"


def run_dir(run_id: str) -> Path:
    # Defense in depth: never build a path from an id that could traverse
    # outside RUNS_DIR (ids reach this function from URL query params).
    if not is_valid_run_id(run_id):
        raise ValueError(f"Invalid run_id: {run_id!r}")
    return RUNS_DIR / run_id


def run_exists(run_id: str) -> bool:
    if not is_valid_run_id(run_id):
        return False
    return run_dir(run_id).exists()


# ── Atomic write helper ───────────────────────────────────────────────────────

def _atomic_write_text(path: Path, text: str) -> None:
    """
    Write via temp-file + os.replace so a crash mid-write can never leave a
    half-written (corrupt) JSON file behind. Corrupt checkpoints previously
    made resume crash with a TypeError.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


# ── Seed + settings persistence ──────────────────────────────────────────────

def save_seed(run_id: str, seed: SeedInput) -> None:
    _atomic_write_text(
        run_dir(run_id) / "seed.json",
        json.dumps(seed.model_dump(mode="json"), indent=2),
    )


def load_seed(run_id: str) -> Optional[SeedInput]:
    path = run_dir(run_id) / "seed.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return SeedInput.model_validate(data)
    except Exception:
        return None


def save_settings(run_id: str, settings: dict) -> None:
    _atomic_write_text(run_dir(run_id) / "settings.json", json.dumps(settings, indent=2))


def load_settings(run_id: str) -> dict:
    path = run_dir(run_id) / "settings.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


# ── Progress tracking ────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_progress(run_id: str) -> dict:
    """Create a fresh progress.json for a new run."""
    progress = {
        "run_id":            run_id,
        "status":            "running",
        "completed_stages":  [],
        "last_stage":        None,
        "message":           "Starting...",
        "started_at":        _now_iso(),
        "last_updated_at":   _now_iso(),
        "error":             None,
    }
    _write_progress(run_id, progress)
    return progress


def read_progress(run_id: str) -> Optional[dict]:
    path = run_dir(run_id) / "progress.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_progress(run_id: str, progress: dict) -> None:
    _atomic_write_text(run_dir(run_id) / "progress.json", json.dumps(progress, indent=2))


def mark_stage_complete(run_id: str, stage: str, message: str = "") -> None:
    progress = read_progress(run_id) or init_progress(run_id)
    if stage not in progress["completed_stages"]:
        progress["completed_stages"].append(stage)
    progress["last_stage"]      = stage
    progress["message"]         = message or f"Completed {stage}"
    progress["last_updated_at"] = _now_iso()
    progress["status"]          = "running"
    _write_progress(run_id, progress)


def heartbeat(run_id: str, message: str = "") -> None:
    """Touch last_updated_at without changing stage. Call from long stages."""
    progress = read_progress(run_id)
    if not progress:
        return
    progress["last_updated_at"] = _now_iso()
    if message:
        progress["message"] = message
    _write_progress(run_id, progress)


def mark_run_completed(run_id: str, message: str = "Pipeline complete") -> None:
    progress = read_progress(run_id) or init_progress(run_id)
    # A run the user cancelled (status=failed) must never be flipped back to
    # completed by a worker that was still finishing its last stage.
    if progress.get("status") == "failed":
        return
    progress["status"]          = "completed"
    progress["message"]         = message
    progress["last_updated_at"] = _now_iso()
    _write_progress(run_id, progress)


def mark_run_failed(run_id: str, error: str) -> None:
    progress = read_progress(run_id) or init_progress(run_id)
    progress["status"]          = "failed"
    progress["error"]           = error
    progress["last_updated_at"] = _now_iso()
    _write_progress(run_id, progress)


# ── Resume detection ─────────────────────────────────────────────────────────

def _seconds_since(iso_ts: str) -> float:
    try:
        dt = datetime.fromisoformat(iso_ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt).total_seconds()
    except Exception:
        return float("inf")


def is_within_grace(run_id: str, grace_seconds: int = GRACE_WINDOW_SECONDS) -> bool:
    """
    True if the run was updated within the grace window. Kept for callers
    that want to *display* how fresh a run is, but `run_status` no longer
    uses it — resume is always allowed regardless of age.
    """
    progress = read_progress(run_id)
    if not progress:
        return False
    if progress.get("status") not in ("running",):
        return False
    return _seconds_since(progress.get("last_updated_at", "")) <= grace_seconds


def run_status(run_id: str) -> str:
    """
    Returns one of: 'missing', 'running', 'completed', 'failed'.

    There is no time limit on resume. A run that hasn't heartbeated in
    hours or days still reports 'running' — the caller is expected to
    check whether a worker thread is alive in *this* process and
    re-spawn it if not.
    """
    progress = read_progress(run_id)
    if not progress:
        return "missing"
    status = progress.get("status", "running")
    if status == "completed":
        return "completed"
    if status == "failed":
        return "failed"
    return "running"


def next_stage_to_run(run_id: str) -> Optional[str]:
    """Return the next stage that has NOT been completed yet, or None if all done."""
    progress = read_progress(run_id)
    if not progress:
        return STAGE_ORDER[0]
    completed = set(progress.get("completed_stages", []))
    for stage in STAGE_ORDER:
        if stage not in completed:
            return stage
    return None


# ── Cancellation ─────────────────────────────────────────────────────────────
# Python threads can't be killed, so cancellation is cooperative: the UI
# writes a _cancel flag file and the pipeline checks it between stages.

def _cancel_path(run_id: str) -> Path:
    return run_dir(run_id) / "_cancel"


def request_cancel(run_id: str) -> None:
    """Ask the worker to stop at the next stage boundary."""
    _atomic_write_text(_cancel_path(run_id), _now_iso())


def cancel_requested(run_id: str) -> bool:
    try:
        return _cancel_path(run_id).exists()
    except Exception:
        return False


def clear_cancel(run_id: str) -> None:
    """Remove a stale cancel flag (called before spawning a fresh worker)."""
    try:
        _cancel_path(run_id).unlink(missing_ok=True)
    except Exception:
        pass


# ── Checkpoint helpers ───────────────────────────────────────────────────────

def checkpoint_path(run_id: str, stage: str) -> Path:
    return run_dir(run_id) / f"_checkpoint_{stage}.json"


def has_checkpoint(run_id: str, stage: str) -> bool:
    return checkpoint_path(run_id, stage).exists()


def read_checkpoint(run_id: str, stage: str) -> Optional[dict]:
    path = checkpoint_path(run_id, stage)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_checkpoint(run_id: str, stage: str, data: dict) -> None:
    _atomic_write_text(
        checkpoint_path(run_id, stage),
        json.dumps(data, indent=2, default=str),
    )


# ── Listing + cleanup ────────────────────────────────────────────────────────

def list_active_runs() -> list[dict]:
    """All runs currently in 'running' state (active or stale), newest first."""
    if not RUNS_DIR.exists():
        return []
    runs = []
    for d in RUNS_DIR.iterdir():
        if not d.is_dir() or not is_valid_run_id(d.name):
            continue
        progress = read_progress(d.name)
        if not progress:
            continue
        if progress.get("status") == "running":
            progress["age_seconds"] = _seconds_since(progress.get("last_updated_at", ""))
            runs.append(progress)
    runs.sort(key=lambda p: p.get("started_at", ""), reverse=True)
    return runs


def find_resumable_run_for_seed(seed_keyword: str) -> Optional[str]:
    """Find the most recent running/stale run for this seed keyword."""
    slug = _slugify(seed_keyword)
    for progress in list_active_runs():
        rid = progress.get("run_id", "")
        if rid.endswith(slug):
            return rid
    return None


def delete_run(run_id: str) -> bool:
    d = run_dir(run_id)
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
        return True
    return False


def prune_stale_runs(max_age_seconds: int = STALE_AFTER_SECONDS) -> int:
    """Delete runs older than max_age. Returns count removed."""
    if not RUNS_DIR.exists():
        return 0
    removed = 0
    for d in RUNS_DIR.iterdir():
        if not d.is_dir() or not is_valid_run_id(d.name):
            continue
        progress = read_progress(d.name)
        if not progress:
            continue
        if _seconds_since(progress.get("last_updated_at", "")) > max_age_seconds:
            if delete_run(d.name):
                removed += 1
    return removed
