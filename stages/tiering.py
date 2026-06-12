"""
Stage 6: Supplementary Nodes — Per-Pillar (reliable JSON parsing)

Runs on Gemini Flash (cheap) — supplementary node generation is templated
work and does not need the main reasoning model.

Per-pillar calls = smaller JSON output = no parsing errors.
Each pillar gets 3-4 supplementary nodes per cluster.

Hardened:
  - Fuzzy match for hallucinated parent_cluster_id values
  - Deterministic fallback so every cluster always ends up with >= 2 nodes
    (1 contradiction + 1 information_gain) even if the LLM fails entirely.
"""

import json
import re
from difflib import SequenceMatcher
from typing import Optional
from pydantic import BaseModel

from models import Pillar, Cluster, SupplementaryNode, Intent, FunnelStage
from stages._client import call_gemini_flash_structured, load_prompt
from stages.serp import SerpData


class _RawNode(BaseModel):
    id: str
    title: str
    parent_cluster_id: str
    intent: Intent
    funnel_stage: FunnelStage
    angle: Optional[str] = None


class SupplementaryResponse(BaseModel):
    supplementary_nodes: list[_RawNode]


# Full Koray supplementary prompt lives in prompts/supplementary.txt
# (four angles, anti-patterns, funnel rules). The addendum below pins the
# per-pillar hard constraints this stage relies on.
_SUPP_ADDENDUM = """

# THIS RUN — HARD CONSTRAINTS

- At least 1 contradiction node per cluster (MANDATORY)
- At least 1 information_gain node per cluster (MANDATORY)
- parent_cluster_id MUST match one of the cluster IDs given to you exactly — do NOT invent new IDs
- IDs must be unique — format: supp_<short_slug>
- Output ONLY valid JSON matching the schema shown above — no trailing commas, no comments, no fences."""


def _supp_system_prompt() -> str:
    return load_prompt("supplementary") + _SUPP_ADDENDUM


# ── Angle normalization ───────────────────────────────────────────────────────
# LLMs label the same four canonical angles with synonyms ("perspective_diversity",
# "lifecycle_state_transition", "myth_busting"...). Without normalization the
# minimum-enforcement logic sees "perspective" as missing next to
# "perspective_diversity" and appends a near-duplicate fallback node — the
# Tree Removal map shipped 23 junk duplicates exactly this way.

_ANGLE_SYNONYMS = {
    "contradiction":              "contradiction",
    "myth_busting":               "contradiction",
    "myth-busting":               "contradiction",
    "information_gain":           "information_gain",
    "information-gain":           "information_gain",
    "rare_attribute":             "information_gain",
    "perspective":                "perspective",
    "perspective_diversity":      "perspective",
    "stakeholder_perspective":    "perspective",
    "lifecycle":                  "lifecycle",
    "lifecycle_state_transition": "lifecycle",
    "state_transition":           "lifecycle",
}


def _normalize_angle(angle: Optional[str]) -> Optional[str]:
    if not angle:
        return angle
    key = angle.strip().lower().replace(" ", "_")
    return _ANGLE_SYNONYMS.get(key, key)


# ── ID matcher ────────────────────────────────────────────────────────────────

def _resolve_cluster_id(
    candidate: str,
    cluster_lookup: dict[str, Cluster],
) -> Optional[Cluster]:
    """Exact → case-insensitive → fuzzy match against real cluster IDs."""
    if not candidate:
        return None
    if candidate in cluster_lookup:
        return cluster_lookup[candidate]

    # case-insensitive
    lc_lookup = {cid.lower(): c for cid, c in cluster_lookup.items()}
    if candidate.lower() in lc_lookup:
        return lc_lookup[candidate.lower()]

    # title keyword overlap
    cand_words = set(re.sub(r"[_\-]", " ", candidate).lower().split())
    cand_words -= {"cluster", "pillar", "supp"}
    best: tuple[Optional[Cluster], float] = (None, 0.0)
    for cid, cluster in cluster_lookup.items():
        title_words = set(cluster.title.lower().split())
        overlap = len(cand_words & title_words) / max(len(cand_words), 1)
        slug_sim = SequenceMatcher(None, candidate, cid).ratio()
        score = max(overlap, slug_sim)
        if score > best[1]:
            best = (cluster, score)
    return best[0] if best[1] >= 0.45 else None


# ── Deterministic fallback nodes ──────────────────────────────────────────────

def _slugify(text: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug[:max_len]


FALLBACK_RATIONALE = "Deterministic fallback (LLM did not return a node for this cluster) — REGENERATE before publishing."


def is_fallback_node(node: SupplementaryNode) -> bool:
    """True when a supplementary node came from the deterministic fallback."""
    return bool(node.rationale and "fallback" in node.rationale.lower())


def _fallback_base(cluster_title: str) -> str:
    """
    Title-safe base for fallback templates: drop a ': subtitle' tail and
    trailing punctuation so wrapped titles stay grammatical (the old
    templates produced things like 'How How to Prevent ... Actually
    Affects Site Performance').
    """
    return cluster_title.split(":")[0].strip().rstrip(".")


def _fallback_nodes_for_cluster(cluster: Cluster, pillar: Pillar) -> list[SupplementaryNode]:
    """
    Generate deterministic supplementary nodes when LLM output is missing
    for a cluster. Templates are DOMAIN-NEUTRAL — the previous set was
    written for WordPress ('Affects Site Performance', 'Site Owner') and
    leaked that vocabulary into every non-WordPress map.
    """
    base = _fallback_base(cluster.title)
    pillar_slug = _slugify(pillar.title, 20)
    cluster_slug = _slugify(cluster.title, 20)

    templates = [
        (
            f"Why Common Advice About {base} Is Often Wrong",
            "contradiction",
            FunnelStage.MOFU,
            Intent.INFORMATIONAL,
            "myth",
        ),
        (
            f"Lesser-Known Factors That Affect {base}",
            "information_gain",
            FunnelStage.MOFU,
            Intent.INFORMATIONAL,
            "mechanism",
        ),
        (
            f"{base}: What First-Time Customers Need to Know",
            "perspective",
            FunnelStage.TOFU,
            Intent.INFORMATIONAL,
            "perspective",
        ),
        (
            f"{base}: Long-Term Maintenance and Follow-Up",
            "lifecycle",
            FunnelStage.MOFU,
            Intent.INFORMATIONAL,
            "followup",
        ),
    ]

    nodes: list[SupplementaryNode] = []
    for title, angle, fs, intent, tag in templates:
        nodes.append(SupplementaryNode(
            id=f"supp_{pillar_slug}_{cluster_slug}_{tag}",
            title=title,
            parent_cluster_id=cluster.id,
            intent=intent,
            funnel_stage=fs,
            angle=angle,
            rationale=FALLBACK_RATIONALE,
        ))
    return nodes


def _cluster_satisfied(cluster: Cluster) -> bool:
    """A cluster is satisfied with >= 1 contradiction + >= 1 information_gain + >= 2 total."""
    angles = {_normalize_angle(n.angle) for n in cluster.supplementary_nodes}
    return (
        {"contradiction", "information_gain"}.issubset(angles)
        and len(cluster.supplementary_nodes) >= 2
    )


def _ensure_minimums(pillar: Pillar) -> None:
    """
    Each cluster must have >= 1 contradiction + >= 1 information_gain +
    >= 2 nodes total. Satisfaction is checked BEFORE appending anything —
    the old version appended first and checked after, so a cluster that
    already met every minimum still received one junk fallback node.
    """
    for cluster in pillar.clusters:
        if _cluster_satisfied(cluster):
            continue

        existing_angles = {_normalize_angle(n.angle) for n in cluster.supplementary_nodes}
        existing_ids = {n.id for n in cluster.supplementary_nodes}

        for node in _fallback_nodes_for_cluster(cluster, pillar):
            if _cluster_satisfied(cluster):
                break
            if _normalize_angle(node.angle) in existing_angles:
                continue
            if node.id in existing_ids:
                continue
            cluster.supplementary_nodes.append(node)
            existing_angles.add(_normalize_angle(node.angle))
            existing_ids.add(node.id)


# ── Main per-pillar generator ─────────────────────────────────────────────────

def generate_supplementary_for_pillar(
    pillar: Pillar,
    serp_data: dict[str, SerpData] | None = None,
) -> Pillar:
    """Generate supplementary nodes for ONE pillar — small JSON, reliable."""

    cluster_list = [{"cluster_id": c.id, "title": c.title} for c in pillar.clusters]
    valid_ids = [c.id for c in pillar.clusters]

    related_context = ""
    if serp_data and pillar.id in serp_data:
        related = serp_data[pillar.id].related_searches[:6]
        if related:
            related_context = "\nRelated searches (use as topic seeds):\n" + "\n".join(f"- {r}" for r in related)

    user_msg = (
        f"Pillar: {pillar.title}\n"
        f"Clusters (use these EXACT cluster_ids in parent_cluster_id):\n"
        f"{json.dumps(cluster_list, indent=2)}\n"
        f"Valid cluster IDs only: {valid_ids}\n"
        f"{related_context}\n\n"
        "Generate 3-4 supplementary nodes per cluster.\n"
        "MANDATORY: at least 1 contradiction + 1 information_gain per cluster.\n"
        "parent_cluster_id MUST be one of the listed cluster_ids — do not invent.\n"
        "Output ONLY valid JSON."
    )

    cluster_lookup = {c.id: c for c in pillar.clusters}
    added = 0
    fuzzy_corrected = 0

    try:
        resp = call_gemini_flash_structured(
            system_prompt=_supp_system_prompt(),
            user_message=user_msg,
            response_model=SupplementaryResponse,
            stage="Stage 6 — Supplementary",
        )
        seen_ids: set[str] = {n.id for c in pillar.clusters for n in c.supplementary_nodes}
        for node in resp.supplementary_nodes:
            cluster = _resolve_cluster_id(node.parent_cluster_id, cluster_lookup)
            if cluster is None:
                continue
            if cluster.id != node.parent_cluster_id:
                fuzzy_corrected += 1
            if node.id in seen_ids:
                continue
            seen_ids.add(node.id)
            cluster.supplementary_nodes.append(SupplementaryNode(
                id=node.id,
                title=node.title,
                parent_cluster_id=cluster.id,
                intent=node.intent,
                funnel_stage=node.funnel_stage,
                angle=_normalize_angle(node.angle),
            ))
            added += 1
        msg = f"    {added} nodes added"
        if fuzzy_corrected:
            msg += f" ({fuzzy_corrected} parent_cluster_id auto-corrected)"
        print(msg)
    except Exception as e:
        print(f"    [tiering] LLM call failed: {e}. Using deterministic fallback only.")

    # Always enforce minimums — guarantees no empty clusters
    before = sum(len(c.supplementary_nodes) for c in pillar.clusters)
    _ensure_minimums(pillar)
    after = sum(len(c.supplementary_nodes) for c in pillar.clusters)
    if after > before:
        print(f"    +{after - before} fallback nodes added to satisfy minimums")

    return pillar


def generate_supplementary_for_all(
    pillars: list[Pillar],
    serp_data: dict[str, SerpData] | None = None,
    batch_size: int = 5,  # ignored — per-pillar
) -> list[Pillar]:
    """Generate supplementary nodes per pillar — reliable, no batch JSON issues."""
    for i, pillar in enumerate(pillars):
        print(f"  [{i+1}/{len(pillars)}] Supp: {pillar.title[:50]}")
        generate_supplementary_for_pillar(pillar, serp_data)
    return pillars
