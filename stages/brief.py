"""
Stage 9: Content Brief Generator — v2 (robust validation, SERP-grounded)

All fields have defaults. field_validators handle model quirks:
  - featured_snippet: accepts bool or string ("true"/"yes")
  - relationship_strength: accepts float or text ("strong"/"moderate")
  - HeadingNode.level: normalizes "h2" → "H2"

When real SERP data is available (serp_context), the brief is grounded in
it: real PAA questions are copied into serp_target.people_also_ask, and
competitor titles inform the information-gain angle and word count —
Koray briefs are SERP-informed by definition.
"""

import json
from pathlib import Path
from typing import Optional, Any
from pydantic import BaseModel, Field, field_validator

from models import Pillar, Cluster, TopicalMap
from stages._client import call_anthropic_structured, load_prompt


# ── Output Models ─────────────────────────────────────────────────────────────

class QuerySet(BaseModel):
    primary_query: str = ""
    secondary_queries: list[str] = []
    question_queries: list[str] = []
    journey_stage: str = "awareness"


class EntityAttributeMap(BaseModel):
    core_attributes: list[str] = []
    performance_states: list[str] = []
    failure_modes: list[str] = []
    dependencies: list[str] = []
    optimization_levers: list[str] = []


class HeadingNode(BaseModel):
    level: str = "H2"
    text: str = ""
    semantic_purpose: str = ""

    @field_validator("level", mode="before")
    @classmethod
    def normalize_level(cls, v):
        if isinstance(v, str):
            v = v.upper().strip()
            if not v.startswith("H"):
                v = "H" + v
        return v


class NLPTerms(BaseModel):
    must_include: list[str] = []
    should_include: list[str] = []
    semantic_variants: list[str] = []


class ContentSpecs(BaseModel):
    recommended_word_count: int = 2000
    content_format: str = "article"
    reading_level: str = "intermediate"
    pov: str = "second_person"
    e_e_a_t_signals: list[str] = []


class SERPTarget(BaseModel):
    featured_snippet: Any = False
    featured_snippet_section: Optional[str] = None
    featured_snippet_format: Optional[str] = None
    people_also_ask: list[str] = []
    schema_markup: str = "Article"

    @field_validator("featured_snippet", mode="before")
    @classmethod
    def parse_bool(cls, v):
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.lower() in ("true", "yes", "1")
        return bool(v)


class SemanticBridge(BaseModel):
    bridge_point: str = ""
    link_destination: str = ""
    anchor_suggestion: str = ""
    shared_entity: str = ""
    relationship_strength: Any = 0.7

    @field_validator("relationship_strength", mode="before")
    @classmethod
    def parse_strength(cls, v):
        if isinstance(v, (int, float)):
            return float(v)
        mapping = {
            "strong": 0.85, "very strong": 0.92, "high": 0.85,
            "moderate": 0.65, "medium": 0.65,
            "weak": 0.40, "low": 0.40,
        }
        if isinstance(v, str):
            return mapping.get(v.lower().strip(), 0.70)
        return 0.70


class NextDestination(BaseModel):
    next_page_id: str = ""
    next_page_title: str = ""
    transition_reason: str = ""
    transition_anchor: str = ""


class EEATRequirements(BaseModel):
    author_expertise: str = ""
    experience_signals: list[str] = []
    trust_signals: list[str] = []
    ymyl_considerations: Optional[str] = None


class ContentBrief(BaseModel):
    page_id: str = ""
    page_title: str = ""
    page_type: str = "cluster"
    parent_pillar: Optional[str] = None
    central_entity: str = ""
    section_type: str = "CORE"
    information_gain_angle: str = ""
    queries: QuerySet = Field(default_factory=QuerySet)
    entity_attribute_map: EntityAttributeMap = Field(default_factory=EntityAttributeMap)
    headings: list[HeadingNode] = []
    nlp_terms: NLPTerms = Field(default_factory=NLPTerms)
    content_specs: ContentSpecs = Field(default_factory=ContentSpecs)
    serp_target: SERPTarget = Field(default_factory=SERPTarget)
    semantic_bridges: list[SemanticBridge] = []
    next_destination: NextDestination = Field(default_factory=NextDestination)
    eeat_requirements: EEATRequirements = Field(default_factory=EEATRequirements)
    quality_checklist: dict = {}


# ── Context builders ───────────────────────────────────────────────────────────

def _pillar_ctx(pillar: Pillar, topical_map: TopicalMap) -> dict:
    other = [
        {"id": p.id, "title": p.title, "entities": p.related_entities[:4]}
        for p in topical_map.pillars if p.id != pillar.id
    ]
    return {
        "page_id": pillar.id,
        "page_title": pillar.title,
        "page_type": "pillar",
        "central_entity": pillar.related_entities[0] if pillar.related_entities else pillar.title,
        "intent": pillar.intent.value,
        "commercial_value": pillar.commercial_value,
        "related_entities": pillar.related_entities,
        "clusters": [{"id": c.id, "title": c.title} for c in pillar.clusters[:8]],
        "representative_queries": [q.text for q in pillar.representative_queries[:3]],
        "other_pillars_for_bridges": other[:6],
    }


def _cluster_ctx(cluster: Cluster, parent: Pillar, topical_map: TopicalMap) -> dict:
    other = [
        {"id": p.id, "title": p.title, "entities": p.related_entities[:4]}
        for p in topical_map.pillars if p.id != parent.id
    ]
    return {
        "page_id": cluster.id,
        "page_title": cluster.title,
        "page_type": "cluster",
        "parent_pillar": parent.title,
        "central_entity": cluster.related_entities[0] if cluster.related_entities else cluster.title,
        "intent": cluster.intent.value,
        "related_entities": cluster.related_entities,
        "represented_queries": [q.text for q in cluster.represented_queries[:5]],
        "supplementary_nodes": [{"title": s.title, "angle": s.angle} for s in cluster.supplementary_nodes[:6]],
        "sibling_clusters": [{"id": c.id, "title": c.title} for c in parent.clusters if c.id != cluster.id][:5],
        "other_pillars_for_bridges": other[:6],
    }


_INSTRUCTIONS = (
    "CRITICAL — every top-level field is REQUIRED, none may be omitted or empty:\n"
    "  page_id, page_title, page_type, central_entity, section_type,\n"
    "  information_gain_angle, queries, entity_attribute_map, headings, nlp_terms,\n"
    "  content_specs, serp_target, semantic_bridges, next_destination,\n"
    "  eeat_requirements, quality_checklist.\n"
    "CRITICAL: headings is an array of OBJECTS like "
    '[{"level":"H1","text":"...","semantic_purpose":"..."}]. '
    "Include at least 8 headings, with one CONTRADICTION H2 and one TROUBLESHOOTING H2.\n"
    "CRITICAL: semantic_bridges must contain AT LEAST 2 entries, each pointing to a "
    "DIFFERENT pillar from `other_pillars_for_bridges`. Use the real page_ids.\n"
    "CRITICAL: relationship_strength must be a decimal number like 0.85, NOT text.\n"
    "CRITICAL: featured_snippet must be true or false (boolean), NOT text.\n"
    "CRITICAL: next_destination.next_page_id must be a real page_id from the context.\n"
    "CRITICAL: entity_attribute_map lists must each contain at least 2 items.\n"
    "CRITICAL: nlp_terms.must_include must contain at least 3 entities.\n"
    "Output ONLY valid JSON — no fences, no preamble, no commentary."
)


# ── Generators ────────────────────────────────────────────────────────────────

def _serp_block(serp_context: str) -> str:
    """Format real SERP data for the brief prompt — empty string if none."""
    if not serp_context.strip():
        return ""
    return (
        "\n# REAL SERP DATA (from live Google results — use it)\n"
        "Ground the brief in this data:\n"
        "- Copy real People-Also-Ask questions into serp_target.people_also_ask "
        "(do NOT invent PAA questions when real ones are listed).\n"
        "- Make the information_gain_angle something the ranking competitors "
        "below do NOT already cover.\n"
        "- Calibrate recommended_word_count against the competition shown.\n\n"
        + serp_context + "\n"
    )


def generate_brief_for_pillar(
    pillar: Pillar,
    topical_map: TopicalMap,
    serp_context: str = "",
) -> ContentBrief:
    system_prompt = load_prompt("content_brief")
    ctx = _pillar_ctx(pillar, topical_map)
    user_message = (
        "Generate a complete content brief for this PILLAR page.\n"
        + _INSTRUCTIONS + "\n"
        + _serp_block(serp_context)
        + "# Page Context\n"
        + json.dumps(ctx, indent=2)
    )
    return call_anthropic_structured(
        system_prompt=system_prompt,
        user_message=user_message,
        response_model=ContentBrief,
        max_tokens=8000,
        temperature=0.5,   # briefs benefit from some creativity
        stage="Stage 9 — Brief (pillar)",
    )


def generate_brief_for_cluster(
    cluster: Cluster,
    parent_pillar: Pillar,
    topical_map: TopicalMap,
    serp_context: str = "",
) -> ContentBrief:
    system_prompt = load_prompt("content_brief")
    ctx = _cluster_ctx(cluster, parent_pillar, topical_map)
    user_message = (
        "Generate a complete content brief for this CLUSTER page.\n"
        + _INSTRUCTIONS + "\n"
        "Include at least one contradiction/myth-busting H2.\n"
        + _serp_block(serp_context)
        + "# Page Context\n"
        + json.dumps(ctx, indent=2)
    )
    return call_anthropic_structured(
        system_prompt=system_prompt,
        user_message=user_message,
        response_model=ContentBrief,
        max_tokens=8000,
        temperature=0.5,
        stage="Stage 9 — Brief (cluster)",
    )


def generate_briefs_for_pillar_and_clusters(
    pillar: Pillar,
    topical_map: TopicalMap,
    include_clusters: bool = True,
    max_clusters: int = 2,
    serp_context: str = "",
) -> dict[str, ContentBrief]:
    briefs: dict[str, ContentBrief] = {}
    print(f"  Pillar: {pillar.title}")
    try:
        briefs[pillar.id] = generate_brief_for_pillar(pillar, topical_map, serp_context)
        print("    Pillar brief done")
    except Exception as e:
        print(f"    Pillar failed: {e}")

    if include_clusters:
        for cluster in pillar.clusters[:max_clusters]:
            print(f"  Cluster: {cluster.title[:55]}")
            try:
                briefs[cluster.id] = generate_brief_for_cluster(cluster, pillar, topical_map, serp_context)
                print("    Cluster brief done")
            except Exception as e:
                print(f"    Cluster failed: {e}")

    return briefs


# ── Save ──────────────────────────────────────────────────────────────────────

def save_briefs(briefs: dict[str, ContentBrief], output_dir) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    for page_id, brief in briefs.items():
        path = output_dir / f"brief_{page_id}.md"
        lines = [
            f"# Content Brief: {brief.page_title}",
            "",
            f"**Information Gain:** {brief.information_gain_angle}",
            f"**Journey Stage:** `{brief.queries.journey_stage}`",
            f"**Word Count:** {brief.content_specs.recommended_word_count:,}",
            f"**Section:** {brief.section_type}",
            "",
            "## Target Queries",
            f"- Primary: `{brief.queries.primary_query}`",
        ]
        for q in brief.queries.secondary_queries:
            lines.append(f"- `{q}`")
        lines += ["", "## Questions to Answer"]
        for q in brief.queries.question_queries:
            lines.append(f"- {q}")
        lines += [
            "",
            "## Entity Attributes",
            f"**Core:** {', '.join(brief.entity_attribute_map.core_attributes)}",
            f"**Failure modes:** {', '.join(brief.entity_attribute_map.failure_modes)}",
            f"**Optimization:** {', '.join(brief.entity_attribute_map.optimization_levers)}",
            "",
            "## Heading Structure",
        ]
        for h in brief.headings:
            lvl = h.level
            depth = int(lvl[1]) - 1 if len(lvl) > 1 and lvl[1].isdigit() else 0
            indent = "  " * depth
            lines.append(f"{indent}**{lvl}:** {h.text}")
            if h.semantic_purpose:
                lines.append(f"{indent}*{h.semantic_purpose}*")
        lines += [
            "",
            "## NLP Terms",
            f"**Must include (3-5x):** {', '.join(brief.nlp_terms.must_include)}",
            f"**Should include (1-2x):** {', '.join(brief.nlp_terms.should_include)}",
            "",
            "## Semantic Bridges",
        ]
        for b in brief.semantic_bridges:
            strength = float(b.relationship_strength) if b.relationship_strength else 0.0
            lines.append(f"- [{strength:.2f}] to `{b.link_destination}` via {b.shared_entity}")
            lines.append(f"  Anchor: {b.anchor_suggestion}")
        lines += [
            "",
            "## Next Destination",
            f"Next: **{brief.next_destination.next_page_title}**",
            f"Why: {brief.next_destination.transition_reason}",
            f"CTA: *{brief.next_destination.transition_anchor}*",
            "",
            "## SERP Target",
            f"- Featured snippet: {'Yes' if brief.serp_target.featured_snippet else 'No'}",
            f"- Schema: {brief.serp_target.schema_markup}",
            "",
            "## PAA Questions",
        ]
        for q in brief.serp_target.people_also_ask:
            lines.append(f"- {q}")
        lines += [
            "",
            "## E-E-A-T",
            f"**Expertise:** {brief.eeat_requirements.author_expertise}",
        ]
        for s in brief.eeat_requirements.experience_signals:
            lines.append(f"- {s}")
        lines += ["", "## Quality Checklist"]
        for k, v in brief.quality_checklist.items():
            lines.append(f"- [{'x' if v else ' '}] {k}")

        path.write_text("\n".join(lines), encoding="utf-8")
        paths.append(path)
        print(f"  Saved: {path.name}")

    json_path = output_dir / "all_briefs.json"
    json_path.write_text(json.dumps(
        {pid: b.model_dump(mode="json") for pid, b in briefs.items()},
        indent=2,
    ), encoding="utf-8")
    paths.append(json_path)

    # CSV + DOCX bundles
    try:
        from stages.brief_export import briefs_to_csv, briefs_to_docx
        csv_path = output_dir / "all_briefs.csv"
        csv_path.write_bytes(briefs_to_csv(briefs))
        paths.append(csv_path)
        try:
            docx_path = output_dir / "all_briefs.docx"
            docx_path.write_bytes(briefs_to_docx(briefs))
            paths.append(docx_path)
        except RuntimeError as e:
            print(f"  [export] DOCX skipped: {e}")
    except Exception as e:
        print(f"  [export] CSV/DOCX skipped: {e}")

    return paths
