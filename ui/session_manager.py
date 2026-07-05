"""
Session Manager — saves and loads topical map runs.

Each run is saved to:
  sessions/<timestamp>_<slug>/
    topical_map.json
    topical_map.csv
    topical_map_report.md
    session_meta.json  ← title, seed, date, stats

sessions/index.json  ← list of all sessions (for sidebar)
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

SESSIONS_DIR = Path("sessions")
INDEX_FILE   = SESSIONS_DIR / "index.json"

# Same shape as run_state.RUN_ID_RE — session ids are run ids. This guard
# keeps path traversal out of session_dir()/delete_session() (rmtree!) even
# if a caller ever passes an id from an untrusted source.
_SESSION_ID_RE = re.compile(r"^\d{8}_\d{6}_[a-z0-9_]{1,40}$")


# ── Meta ──────────────────────────────────────────────────────────────────────

def _make_slug(seed: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", seed.lower().strip())
    return slug[:40].strip("_")


def _session_id(seed: str) -> str:
    ts   = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    slug = _make_slug(seed)
    return f"{ts}_{slug}"


def session_dir(session_id: str) -> Path:
    if not session_id or not _SESSION_ID_RE.fullmatch(session_id):
        raise ValueError(f"Invalid session_id: {session_id!r}")
    return SESSIONS_DIR / session_id


# ── Save ──────────────────────────────────────────────────────────────────────

def save_session(output, session_id: str) -> Path:
    """
    Save a completed EngineOutput as a named session.
    Returns the session directory path.
    """
    import shutil

    from stages.render import save_outputs, render_koray_csv

    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = session_dir(session_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save main output files
    paths = save_outputs(output, out_dir)

    # Copy persisted SERP data from the run dir (if present) so brief
    # generation from a reloaded session can still use real PAA/competitor
    # data instead of hallucinating it.
    try:
        run_serp = Path("runs") / session_id / "serp_data.json"
        if run_serp.exists():
            shutil.copyfile(run_serp, out_dir / "serp_data.json")
    except Exception:
        pass

    # Build session metadata
    tm = output.topical_map
    lp = output.linking_plan
    meta = {
        "session_id":   session_id,
        "seed":         output.input.seed_keyword,
        "central":      tm.central_entity.primary,
        "source_context": tm.central_entity.source_context,
        "created_at":   datetime.now(timezone.utc).isoformat(),
        "stats": {
            "pillars":     len(tm.pillars),
            "clusters":    sum(len(p.clusters) for p in tm.pillars),
            "supp_nodes":  sum(sum(len(c.supplementary_nodes) for c in p.clusters) for p in tm.pillars),
            "queries":     sum(len(p.representative_queries) + sum(len(c.represented_queries) for c in p.clusters) for p in tm.pillars),
            "links":       len(lp.links),
            "geo_pages":   len(tm.geo_pages),
        },
    }

    meta_path = out_dir / "session_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Update sessions index
    _update_index(meta)

    return out_dir


def _update_index(meta: dict) -> None:
    """Add or update a session in the index file."""
    sessions = _load_index()
    # Remove old entry if same session_id
    sessions = [s for s in sessions if s.get("session_id") != meta["session_id"]]
    # Add new entry at top
    sessions.insert(0, meta)
    # Keep last 50 sessions
    sessions = sessions[:50]
    INDEX_FILE.write_text(json.dumps(sessions, indent=2), encoding="utf-8")


# ── Load ──────────────────────────────────────────────────────────────────────

def _load_index() -> list[dict]:
    """Load the sessions index. Returns empty list if not found."""
    if not INDEX_FILE.exists():
        return []
    try:
        return json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def list_sessions() -> list[dict]:
    """Return all saved sessions, newest first."""
    return _load_index()


def load_session(session_id: str):
    """
    Load a saved EngineOutput from disk.
    Returns (EngineOutput, session_meta) or (None, None) if not found.
    """
    from models import EngineOutput

    try:
        out_dir = session_dir(session_id)
    except ValueError:
        return None, None
    json_path = out_dir / "topical_map.json"
    meta_path = out_dir / "session_meta.json"

    if not json_path.exists():
        return None, None

    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        output = EngineOutput.model_validate(data)
        meta   = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        return output, meta
    except Exception as e:
        print(f"[session] Failed to load {session_id}: {e}")
        return None, None


def delete_session(session_id: str) -> bool:
    """Delete a session and remove from index."""
    import shutil
    try:
        out_dir = session_dir(session_id)
    except ValueError:
        return False
    if out_dir.exists():
        shutil.rmtree(out_dir)
    sessions = [s for s in _load_index() if s.get("session_id") != session_id]
    INDEX_FILE.write_text(json.dumps(sessions, indent=2), encoding="utf-8")
    return True


def get_session_output_dir(session_id: str) -> str:
    """Return the output directory path for a session."""
    return str(session_dir(session_id))
