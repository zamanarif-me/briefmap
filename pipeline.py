"""
BriefMap — top-level pipeline (Resumable: Anthropic + Gemini + Serper)

Stage assignments:
  Anthropic Sonnet → Stages 2, 3, 5, 7  (quality-critical reasoning)
  Gemini Flash     → Stages 4, 6        (validation fallback, supplementary — cost saving)
  Serper.dev       → Stage 3.5          (SERP + PAA + competitor data)

Resumability:
  Every stage writes a checkpoint to runs/<run_id>/_checkpoint_<stage>.json
  on completion. If the pipeline is interrupted (network drop, page reload,
  Streamlit Cloud session reset) and re-invoked with the same run_id, each
  stage checks for its checkpoint and skips work that is already done.

  Resume is automatic when a run_id is passed. To force a fresh run, use
  a new run_id.
"""

from __future__ import annotations
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional


def _safe_stdout_write(msg: str) -> None:
    """
    Write to the real terminal stdout, bypassing any monkey-patched
    `print` or redirected `sys.stdout`. This is essential because the
    pipeline runs in a background thread — if some other code has
    patched `builtins.print` to call Streamlit UI methods, those calls
    will fail with NoSessionContext when invoked from a non-main thread.

    `sys.__stdout__` is the ORIGINAL stdout that Python captured at
    interpreter start; it is never reassigned by user code.
    """
    try:
        sys.__stdout__.write(msg)
        sys.__stdout__.flush()
    except Exception:
        pass

from models import (
    SeedInput, TopicalMap, EngineOutput,
    CentralEntity, Pillar, LinkingPlan,
)
from stages.intake import load_intake_from_json
from stages.central_entity import extract_central_entity
from stages.expansion import expand_topics
from stages.serp import pull_serp_for_pillars, save_serp_data, SerpData, OrganicResult
from stages.validation import validate_topics
from stages.queries import generate_queries_for_all
from stages.tiering import generate_supplementary_for_all
from stages.linking import build_linking_plan
from stages.geo import derive_geo_pages
from stages.render import save_outputs
from stages.cost_tracker import CostTracker, use_tracker


def _log(msg: str):
    # Use sys.__stdout__ directly so we can never accidentally end up
    # calling a monkey-patched `print` that touches Streamlit UI from
    # the background worker thread (that path raises NoSessionContext).
    _safe_stdout_write(f"[pipeline] {msg}\n")


# ── Serialization helpers for SerpData (dataclass, not pydantic) ──────────────

def _serpdata_to_dict(s: SerpData) -> dict:
    return {
        "pillar_id":        s.pillar_id,
        "query":            s.query,
        "organic":          [asdict(o) for o in s.organic],
        "paa":              list(s.paa),
        "related_searches": list(s.related_searches),
        "featured_snippet": s.featured_snippet,
    }


def _dict_to_serpdata(d: dict) -> SerpData:
    sd = SerpData(pillar_id=d.get("pillar_id", ""), query=d.get("query", ""))
    sd.organic = [
        OrganicResult(
            position=o.get("position", 0),
            title=o.get("title", ""),
            url=o.get("url", ""),
            snippet=o.get("snippet", ""),
        )
        for o in d.get("organic", [])
    ]
    sd.paa              = list(d.get("paa", []))
    sd.related_searches = list(d.get("related_searches", []))
    sd.featured_snippet = d.get("featured_snippet", "")
    return sd


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    seed: SeedInput,
    output_dir: str | Path | None = None,
    skip_serp: bool = False,
    skip_validation: bool = False,
    serp_geo: str = "us",
    serp_lang: str = "en",
    run_id: Optional[str] = None,
) -> EngineOutput:
    """
    Run all stages end-to-end. Resumable when run_id is provided.

    skip_serp:       Skip Stage 3.5 (saves Serper calls during prompt iteration)
    skip_validation: Skip Stage 4 entirely
    serp_geo:        Serper country code — "us", "gb", "au" etc.
    serp_lang:       Language code — "en" etc.
    run_id:          Unique run identifier. If supplied and prior checkpoints
                     exist under runs/<run_id>/, those stages are skipped.
                     If omitted, a fresh run_id is generated.
    """
    # Lazy import so this module is usable from CLI without Streamlit installed
    from ui import run_state

    # ── Resolve run_id and working directory ─────────────────────────────────
    if run_id is None:
        run_id = run_state.new_run_id(seed.seed_keyword)
        _log(f"New run_id: {run_id}")
    else:
        _log(f"Resuming run_id: {run_id}")

    work_dir = run_state.run_dir(run_id)
    work_dir.mkdir(parents=True, exist_ok=True)

    # output_dir for final files defaults to the run directory itself
    output_dir = Path(output_dir) if output_dir else work_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Persist seed + settings so a fully-fresh process can resume
    run_state.save_seed(run_id, seed)
    run_state.save_settings(run_id, {
        "skip_serp":       skip_serp,
        "skip_validation": skip_validation,
        "serp_geo":        serp_geo,
        "serp_lang":       serp_lang,
        "output_dir":      str(output_dir),
    })

    # Initialize progress tracking
    if run_state.read_progress(run_id) is None:
        run_state.init_progress(run_id)

    # Cost tracker — one tracker per run, installed for this thread/context.
    # Concurrent runs (multiple users) no longer intermix or reset each
    # other's cost data.
    run_tracker = CostTracker()
    use_tracker(run_tracker)

    # ── Stage 2: Central Entity (Anthropic) ──────────────────────────────────
    # NOTE on all stages below: read_checkpoint() returns None for a missing
    # OR corrupt checkpoint file, so a half-written checkpoint (crash mid-
    # write) re-runs the stage instead of crashing the resume.
    cp = run_state.read_checkpoint(run_id, "stage2")
    if cp and "central_entity" in cp:
        central = CentralEntity.model_validate(cp["central_entity"])
        _log(f"Stage 2: SKIPPED (checkpoint found) — {central.primary}")
    else:
        _log("Stage 2: extracting central entity [Anthropic]...")
        run_state.heartbeat(run_id, "Stage 2: extracting central entity...")
        central = extract_central_entity(seed)
        _log(f"  Primary: {central.primary}")
        run_state.write_checkpoint(run_id, "stage2", {
            "central_entity": central.model_dump(mode="json"),
        })
        run_state.mark_stage_complete(run_id, "stage2", f"Central entity: {central.primary}")

    # ── Stage 3: Topic Expansion (Anthropic) ─────────────────────────────────
    cp = run_state.read_checkpoint(run_id, "stage3")
    if cp and "pillars" in cp:
        pillars = [Pillar.model_validate(p) for p in cp["pillars"]]
        _log(f"Stage 3: SKIPPED (checkpoint found) — {len(pillars)} pillars")
    else:
        _log("Stage 3: expanding pillars and clusters [Anthropic]...")
        run_state.heartbeat(run_id, "Stage 3: expanding pillars and clusters...")
        pillars = expand_topics(seed, central)
        _log(f"  {len(pillars)} pillars, {sum(len(p.clusters) for p in pillars)} clusters")
        run_state.write_checkpoint(run_id, "stage3", {
            "central_entity": central.model_dump(mode="json"),
            "pillars":        [p.model_dump(mode="json") for p in pillars],
        })
        run_state.mark_stage_complete(run_id, "stage3", f"{len(pillars)} pillars expanded")

    # ── Stage 3.5: SERP Intelligence (Serper.dev) ────────────────────────────
    serp_data: Optional[dict[str, SerpData]] = None
    cp = None if skip_serp else run_state.read_checkpoint(run_id, "stage3_5")
    if skip_serp:
        _log("Stage 3.5: SKIPPED (skip_serp=True)")
        run_state.mark_stage_complete(run_id, "stage3_5", "SERP skipped by setting")
    elif cp:
        serp_data = {pid: _dict_to_serpdata(d) for pid, d in cp.items()}
        _log(f"Stage 3.5: SKIPPED (checkpoint found) — {len(serp_data)} pillars cached")
    else:
        _log("Stage 3.5: pulling SERP data [Serper.dev]...")
        run_state.heartbeat(run_id, "Stage 3.5: pulling SERP data...")
        serp_data = pull_serp_for_pillars(pillars, geo=serp_geo, lang=serp_lang)
        run_tracker.log_serper_call("Stage 3.5 — Serper.dev", len(pillars))
        total_paa     = sum(len(s.paa) for s in serp_data.values())
        total_related = sum(len(s.related_searches) for s in serp_data.values())
        _log(f"  PAA questions: {total_paa} | Related searches: {total_related}")
        run_state.write_checkpoint(run_id, "stage3_5", {
            pid: _serpdata_to_dict(s) for pid, s in serp_data.items()
        })
        run_state.mark_stage_complete(run_id, "stage3_5", f"{total_paa} PAA, {total_related} related")

    # Persist SERP data alongside the outputs so Stage 9 (content briefs)
    # can ground itself in real PAA / competitor data later.
    if serp_data:
        try:
            save_serp_data(serp_data, output_dir / "serp_data.json")
        except Exception as e:
            _log(f"  WARNING: could not persist serp_data.json: {e}")

    # ── Stage 4: Validation ──────────────────────────────────────────────────
    cp = None if skip_validation else run_state.read_checkpoint(run_id, "stage4")
    if skip_validation:
        _log("Stage 4: SKIPPED (skip_validation=True)")
        run_state.mark_stage_complete(run_id, "stage4", "Validation skipped by setting")
    elif cp and "pillars" in cp:
        pillars = [Pillar.model_validate(p) for p in cp["pillars"]]
        _log("Stage 4: SKIPPED (checkpoint found)")
    else:
        provider_label = "data-driven" if serp_data else "Gemini Flash"
        _log(f"Stage 4: validating topics [{provider_label}]...")
        run_state.heartbeat(run_id, f"Stage 4: validating topics [{provider_label}]...")
        validations = validate_topics(pillars, serp_data)
        strong = sum(1 for v in validations.values() if v.signal == "strong")
        medium = sum(1 for v in validations.values() if v.signal == "medium")
        weak   = sum(1 for v in validations.values() if v.signal == "weak")
        _log(f"  strong: {strong} | medium: {medium} | weak: {weak}")
        for pillar in pillars:
            if pillar.id in validations:
                v = validations[pillar.id]
                pillar.validation_signal    = v.signal
                pillar.validation_reasoning = v.reasoning
            for cluster in pillar.clusters:
                if cluster.id in validations:
                    v = validations[cluster.id]
                    cluster.validation_signal    = v.signal
                    cluster.validation_reasoning = v.reasoning
        run_state.write_checkpoint(run_id, "stage4", {
            "pillars": [p.model_dump(mode="json") for p in pillars],
        })
        run_state.mark_stage_complete(run_id, "stage4", f"strong:{strong} medium:{medium} weak:{weak}")

    # ── Stage 5: Query Generation (Anthropic + Serper PAA) ───────────────────
    cp = run_state.read_checkpoint(run_id, "stage5")
    if cp and "pillars" in cp:
        pillars = [Pillar.model_validate(p) for p in cp["pillars"]]
        total_queries = sum(
            len(p.representative_queries) + sum(len(c.represented_queries) for c in p.clusters)
            for p in pillars
        )
        _log(f"Stage 5: SKIPPED (checkpoint found) — {total_queries} queries")
    else:
        _log("Stage 5: generating queries [Anthropic + Serper PAA]...")
        run_state.heartbeat(run_id, "Stage 5: generating queries...")
        pillars = generate_queries_for_all(pillars, serp_data)
        total_queries = sum(
            len(p.representative_queries) + sum(len(c.represented_queries) for c in p.clusters)
            for p in pillars
        )
        _log(f"  {total_queries} queries total")
        run_state.write_checkpoint(run_id, "stage5", {
            "pillars": [p.model_dump(mode="json") for p in pillars],
        })
        run_state.mark_stage_complete(run_id, "stage5", f"{total_queries} queries generated")

    # ── Stage 6: Supplementary Nodes (Gemini Flash + Serper related) ─────────
    cp = run_state.read_checkpoint(run_id, "stage6")
    if cp and "pillars" in cp:
        pillars = [Pillar.model_validate(p) for p in cp["pillars"]]
        total_supp = sum(
            sum(len(c.supplementary_nodes) for c in p.clusters)
            for p in pillars
        )
        _log(f"Stage 6: SKIPPED (checkpoint found) — {total_supp} supplementary nodes")
    else:
        _log("Stage 6: generating supplementary nodes [Gemini Flash + Serper related]...")
        run_state.heartbeat(run_id, "Stage 6: generating supplementary nodes...")
        pillars = generate_supplementary_for_all(pillars, serp_data)
        total_supp = sum(
            sum(len(c.supplementary_nodes) for c in p.clusters)
            for p in pillars
        )
        _log(f"  {total_supp} supplementary nodes")
        run_state.write_checkpoint(run_id, "stage6", {
            "pillars": [p.model_dump(mode="json") for p in pillars],
        })
        run_state.mark_stage_complete(run_id, "stage6", f"{total_supp} supplementary nodes")

    # ── Geo pages (deterministic, fast — no checkpoint needed) ───────────────
    _log("Deriving geographic service pages...")
    geo_pages = derive_geo_pages(pillars, seed.intake.geo)
    _log(f"  {len(geo_pages)} geo pages")

    topical_map = TopicalMap(
        central_entity=central,
        pillars=pillars,
        geo_pages=geo_pages,
    )

    # ── Stage 7: Internal Linking (Anthropic) ────────────────────────────────
    cp = run_state.read_checkpoint(run_id, "stage7")
    if cp and "linking_plan" in cp:
        linking_plan = LinkingPlan.model_validate(cp["linking_plan"])
        _log(f"Stage 7: SKIPPED (checkpoint found) — {len(linking_plan.links)} links")
    else:
        _log("Stage 7: building internal linking plan [Anthropic]...")
        run_state.heartbeat(run_id, "Stage 7: building internal linking plan...")
        linking_plan = build_linking_plan(pillars)
        _log(f"  {len(linking_plan.links)} links | {len(linking_plan.homepage_links)} homepage links")
        run_state.write_checkpoint(run_id, "stage7", {
            "linking_plan": linking_plan.model_dump(mode="json"),
        })
        run_state.mark_stage_complete(run_id, "stage7", f"{len(linking_plan.links)} links")

    final_output = EngineOutput(
        input=seed,
        topical_map=topical_map,
        linking_plan=linking_plan,
    )

    # ── Stage 8: Render ──────────────────────────────────────────────────────
    if run_state.read_checkpoint(run_id, "stage8") is not None:
        _log("Stage 8: SKIPPED (checkpoint found) — outputs already rendered")
    else:
        _log("Stage 8: rendering JSON + Markdown report...")
        run_state.heartbeat(run_id, "Stage 8: rendering outputs...")
        paths = save_outputs(final_output, output_dir)
        _log(f"  JSON:     {paths['json']}")
        _log(f"  Markdown: {paths['markdown']}")
        run_state.write_checkpoint(run_id, "stage8", {
            "paths": {k: str(v) for k, v in paths.items()},
        })
        run_state.mark_stage_complete(run_id, "stage8", "Outputs rendered")

    _log("Pipeline complete.")
    run_tracker.print_report()
    run_tracker.save_report(output_dir / "cost_report.json")

    # ── Auto-save session (Session History sidebar) ──────────────────────────
    try:
        from ui.session_manager import save_session
        save_session(final_output, run_id)
        _log(f"Session saved: {run_id}")
    except Exception as e:
        _log(f"Session save skipped: {e}")

    # Attach run metadata BEFORE marking the run completed, so a failure
    # here cannot flip an already-completed run to "failed".
    try:
        final_output._session_id = run_id
        final_output._run_id = run_id
    except Exception:
        pass

    # Mark the run as fully completed in run_state — last, nothing after this
    # can raise.
    run_state.mark_run_completed(run_id, "Pipeline complete")

    return final_output


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        print("Usage: python -m pipeline <input.json> <output_dir> [run_id]")
        sys.exit(1)
    seed = load_intake_from_json(sys.argv[1])
    rid = sys.argv[3] if len(sys.argv) == 4 else None
    run_pipeline(seed, sys.argv[2], run_id=rid)
