"""
Stage 8: Render.

Pure Python — no API calls. Takes the assembled EngineOutput and produces:
1. A JSON file with the complete structured data
2. A Markdown report suitable for sharing with a client

The rendering is intentionally template-driven (Jinja2) so non-engineers
can adjust the report format without touching code.
"""

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from models import EngineOutput, LinkRelationship


def _build_template_context(output: EngineOutput) -> dict:
    """Pre-compute derived data the template needs."""
    # Pillar title lookup for friendly homepage link display
    pillar_titles = {p.id: p.title for p in output.topical_map.pillars}

    # Count links by relationship type
    link_counts: dict[str, int] = defaultdict(int)
    for link in output.linking_plan.links:
        link_counts[link.relationship.value] += 1

    # Entity bridges (the strategic cross-pillar links)
    entity_bridges = [
        link for link in output.linking_plan.links
        if link.relationship == LinkRelationship.ENTITY_BRIDGE
    ]

    # Group links by pillar (links where either end is in the pillar's subtree)
    pillar_subtree_ids: dict[str, set[str]] = {}
    for pillar in output.topical_map.pillars:
        ids = {pillar.id}
        for c in pillar.clusters:
            ids.add(c.id)
            for s in c.supplementary_nodes:
                ids.add(s.id)
        pillar_subtree_ids[pillar.id] = ids

    links_by_pillar: dict[str, list] = defaultdict(list)
    for link in output.linking_plan.links:
        for pillar_id, subtree in pillar_subtree_ids.items():
            if link.from_page_id in subtree or link.to_page_id in subtree:
                links_by_pillar[pillar_id].append(link)

    return {
        "output": output,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "pillar_titles": pillar_titles,
        "link_counts": dict(link_counts),
        "entity_bridges": entity_bridges,
        "links_by_pillar": links_by_pillar,
    }


def render_markdown(output: EngineOutput, templates_dir: str | Path | None = None) -> str:
    """Render the output to a Markdown report."""
    if templates_dir is None:
        templates_dir = Path(__file__).parent.parent / "templates"

    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=select_autoescape(disabled_extensions=("md", "j2")),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template("report.md.j2")
    context = _build_template_context(output)
    return template.render(**context)


def save_outputs(output: EngineOutput, output_dir: str | Path) -> dict[str, Path]:
    """Save both JSON and Markdown to output_dir. Returns the paths."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "topical_map.json"
    md_path = out_dir / "topical_map_report.md"

    # JSON
    json_path.write_text(
        json.dumps(output.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )

    # Markdown
    md_path.write_text(render_markdown(output), encoding="utf-8")

    # CSV (Koray format)
    csv_path = out_dir / "topical_map.csv"
    csv_path.write_text(render_koray_csv(output), encoding="utf-8-sig")  # utf-8-sig for Excel

    return {"json": json_path, "markdown": md_path, "csv": csv_path}


def render_koray_csv(output: EngineOutput) -> str:
    """
    Export topical map as Koray-format CSV.

    Columns:
      Contextual Vector | Contextual Hierarchy | Contextual Connection |
      Query Terms | Volume | Source Context | Entity Bridges

    Structure:
      site_entity      → top-level source context anchor
      h1               → Pillar page
      h2               → Cluster page
      h3               → Supplementary node
      entity_bridge    → cross-pillar relationship row (appended after each pillar)

    `Source Context` is propagated on EVERY row so the source identity is
    semantically present at every level of the hierarchy — addresses the
    "source-centric not page-centric" gap.

    `Entity Bridges` carries cross-pillar destinations on every cluster/pillar
    row, and a dedicated trailing `entity_bridge` row block per pillar lists
    each bridge as a graph edge.
    """
    import csv, io
    out = io.StringIO()
    _raw_writer = csv.writer(out)

    def _csv_safe(value):
        # Excel executes cells starting with = + - @ (or tab/CR) as formulas.
        # All cell content here is LLM/SERP-derived, so neutralize with a
        # leading apostrophe (CSV formula-injection defense).
        if isinstance(value, str) and value and value[0] in ("=", "+", "-", "@", "\t", "\r"):
            return "'" + value
        return value

    class _SafeWriter:
        def writerow(self, row):
            _raw_writer.writerow([_csv_safe(cell) for cell in row])

    writer = _SafeWriter()

    HEADER = [
        "Contextual Vector",
        "Contextual Hierarchy",
        "Contextual Connection",
        "Query Terms",
        "Volume",
        "Source Context",
        "Entity Bridges",
    ]
    writer.writerow(HEADER)

    tm = output.topical_map
    source_context = (tm.central_entity.source_context or "").strip()

    # ── Build entity-bridge index: page_id → list of bridge descriptors ───────
    pillar_title = {p.id: p.title for p in tm.pillars}
    bridges_from: dict[str, list[str]] = defaultdict(list)
    bridges_for_pillar: dict[str, list] = defaultdict(list)

    cluster_to_pillar = {c.id: p.id for p in tm.pillars for c in p.clusters}

    for link in output.linking_plan.links:
        if link.relationship != LinkRelationship.ENTITY_BRIDGE:
            continue
        anchor_target = pillar_title.get(link.to_page_id, link.to_page_id)
        bridges_from[link.from_page_id].append(
            f"→ {anchor_target} ({link.anchor_text})"
        )
        owner = cluster_to_pillar.get(link.from_page_id)
        if owner:
            bridges_for_pillar[owner].append(link)
        # Also surface inbound bridges on the destination pillar
        bridges_for_pillar[link.to_page_id].append(link)

    def _bridges_cell(page_id: str) -> str:
        return " ; ".join(bridges_from.get(page_id, []))

    # ── Site-level central entity (source-of-context row) ────────────────────
    writer.writerow([
        tm.central_entity.primary,
        "site_entity",
        source_context[:160],
        "",
        "",
        source_context,
        "",
    ])

    # ── Pillars / clusters / supplementary nodes ─────────────────────────────
    for pillar in tm.pillars:
        pillar_queries = " | ".join(q.text for q in pillar.representative_queries[:3])
        writer.writerow([
            pillar.title,
            "h1",
            f"[{pillar.intent.value}] [{pillar.funnel_stage.value}] Priority {pillar.priority}",
            pillar_queries,
            "",
            source_context,
            _bridges_cell(pillar.id),
        ])

        for cluster in pillar.clusters:
            cluster_queries = " | ".join(q.text for q in cluster.represented_queries[:3])
            writer.writerow([
                cluster.title,
                "h2",
                pillar.title,
                cluster_queries,
                "",
                source_context,
                _bridges_cell(cluster.id),
            ])

            for node in cluster.supplementary_nodes:
                angle = f"[{node.angle}] " if node.angle else ""
                # Flag deterministic-fallback nodes so they are visible in
                # the deliverable — a client should never receive template
                # placeholders presented as generated content.
                is_fallback = bool(node.rationale and "fallback" in node.rationale.lower())
                writer.writerow([
                    node.title,
                    "h3",
                    cluster.title,
                    f"{angle}{node.funnel_stage.value}",
                    "",
                    source_context,
                    "FALLBACK PLACEHOLDER — regenerate before publishing" if is_fallback else "",
                ])

        # ── Entity-bridge rows for this pillar ────────────────────────────────
        seen_bridge_keys: set[tuple] = set()
        for link in bridges_for_pillar.get(pillar.id, []):
            key = (link.from_page_id, link.to_page_id)
            if key in seen_bridge_keys:
                continue
            seen_bridge_keys.add(key)
            strength = (
                f" [strength={link.relationship_strength:.2f}]"
                if link.relationship_strength is not None else ""
            )
            writer.writerow([
                f"{link.from_page_id} → {link.to_page_id}",
                "entity_bridge",
                pillar.title,
                link.anchor_text,
                "",
                source_context,
                f"{link.reasoning}{strength}",
            ])

        # Blank separator between pillars
        writer.writerow(["", "", "", "", "", "", ""])

    return out.getvalue()
