"""
Stage 7: Internal Linking — Fully Optimized

Split into two parts:
  DETERMINISTIC (zero tokens):
    - pillar → cluster        (rule: every pillar links all its clusters)
    - cluster → pillar        (rule: every cluster links back to parent)
    - cluster → supplementary (rule: every cluster links its supp nodes)
    - supplementary → cluster (rule: every supp node links back to cluster)
    - homepage → pillar       (rule: priority 1 and 2 pillars)

  LLM — ONE single call for the entire map (not per-pillar):
    - entity bridges only     (requires reasoning — LLM needed)

Before: 10 LLM calls × ~5,000 output tokens = 50,000 tokens
After:  1 LLM call  × ~3,000 output tokens =  3,000 tokens
Saving: ~47,000 output tokens = ~$0.70
"""

import hashlib
import json

from models import Pillar, InternalLink, LinkingPlan, LinkRelationship
from stages._client import call_anthropic_structured, load_prompt
from pydantic import BaseModel


# ── Prompt for entity bridges only ───────────────────────────────────────────
# The full Koray linking prompt (anchor rules, bridge rules, anti-patterns)
# lives in prompts/internal_linking.txt. Hierarchy links are generated
# deterministically in this module, so the LLM is scoped to bridges only.

_BRIDGE_ADDENDUM = """

# THIS RUN — ENTITY BRIDGES ONLY

The hierarchy links (pillar↔cluster, cluster↔supplementary, homepage) are
generated deterministically outside of this call. Your ONLY job here is the
ENTITY_BRIDGE links, following all anchor-text and bridge rules above.

MANDATORY:
- Generate at least 2 bridges per pillar — required, not optional.
- Each bridge: from_page_id is a cluster_id, to_page_id is a DIFFERENT pillar_id.
- reasoning: max 8 words, naming the shared entity.
- relationship_strength: decimal 0.0-1.0 (e.g. 0.85), NEVER text like "strong".
    0.90-1.00 = direct entity overlap (WooCommerce → WooCommerce)
    0.70-0.89 = strong contextual overlap (Speed → Core Web Vitals)
    0.50-0.69 = moderate semantic connection (Security → Maintenance)
    0.30-0.49 = weak but valid connection (Development → Local Business)
- Do NOT return an empty links array.

Output ONLY valid JSON:
{
  "links": [
    {
      "from_page_id": "cluster_id",
      "to_page_id": "pillar_id",
      "anchor_text": "natural anchor text here",
      "relationship": "entity_bridge",
      "reasoning": "Shared WooCommerce entity bridge.",
      "relationship_strength": 0.92
    }
  ]
}"""


def _bridge_system_prompt() -> str:
    return load_prompt("internal_linking") + _BRIDGE_ADDENDUM


class _BridgeResponse(BaseModel):
    links: list[InternalLink]


# ── Deterministic link generators ─────────────────────────────────────────────

# Anchor templates per direction. Rotating templates (seeded by a stable hash
# of the from/to pair) avoids the "always use the destination title as anchor"
# over-optimization anti-pattern that the linking prompt itself forbids —
# while staying deterministic across runs.

_ANCHOR_TEMPLATES_DOWN = [          # parent → child
    "{title}",
    "our guide to {title_lower}",
    "learn about {title_lower}",
    "see {title_lower} in detail",
]

_ANCHOR_TEMPLATES_UP = [            # child → parent
    "{title}",
    "our {title_lower}",
    "explore {title_lower}",
    "back to {title_lower}",
]


def _vary_anchor(from_id: str, to_id: str, title: str, templates: list[str]) -> str:
    """Pick a stable anchor variation for this specific edge."""
    digest = hashlib.md5(f"{from_id}->{to_id}".encode("utf-8")).digest()
    template = templates[digest[0] % len(templates)]
    anchor = template.format(title=title, title_lower=title[:1].lower() + title[1:])
    return anchor[:70]


def _pillar_cluster_links(pillars: list[Pillar]) -> list[InternalLink]:
    """Every pillar links to all its clusters and back. Rule-based, zero tokens."""
    links = []
    for pillar in pillars:
        for cluster in pillar.clusters:
            # pillar → cluster
            links.append(InternalLink(
                from_page_id=pillar.id,
                to_page_id=cluster.id,
                anchor_text=_vary_anchor(pillar.id, cluster.id, cluster.title, _ANCHOR_TEMPLATES_DOWN),
                relationship=LinkRelationship.PILLAR_TO_CLUSTER,
                reasoning="Pillar to cluster.",
            ))
            # cluster → pillar
            links.append(InternalLink(
                from_page_id=cluster.id,
                to_page_id=pillar.id,
                anchor_text=_vary_anchor(cluster.id, pillar.id, pillar.title, _ANCHOR_TEMPLATES_UP),
                relationship=LinkRelationship.CLUSTER_TO_PILLAR,
                reasoning="Cluster to parent pillar.",
            ))
    return links


def _supplementary_links(pillars: list[Pillar]) -> list[InternalLink]:
    """Every cluster links its supplementary nodes and back. Rule-based, zero tokens."""
    links = []
    for pillar in pillars:
        for cluster in pillar.clusters:
            for node in cluster.supplementary_nodes:
                links.append(InternalLink(
                    from_page_id=cluster.id,
                    to_page_id=node.id,
                    anchor_text=_vary_anchor(cluster.id, node.id, node.title, _ANCHOR_TEMPLATES_DOWN),
                    relationship=LinkRelationship.CLUSTER_TO_SUPPLEMENTARY,
                    reasoning="Cluster to supplementary.",
                ))
                links.append(InternalLink(
                    from_page_id=node.id,
                    to_page_id=cluster.id,
                    anchor_text=_vary_anchor(node.id, cluster.id, cluster.title, _ANCHOR_TEMPLATES_UP),
                    relationship=LinkRelationship.SUPPLEMENTARY_TO_CLUSTER,
                    reasoning="Supplementary to parent cluster.",
                ))
    return links


def _homepage_links(pillars: list[Pillar]) -> list[str]:
    """Homepage links to priority 1 and 2 pillars. Rule-based."""
    return [
        p.id for p in sorted(pillars, key=lambda x: x.priority)
        if p.priority <= 2
    ]


def _geo_page_links(geo_pages: list) -> list[InternalLink]:
    """
    Every geo page links back to its parent pillar and the pillar links out
    to each of its regional variants. Without these the geo pages are orphans
    in the link graph — no inbound or outbound links at all.
    """
    links: list[InternalLink] = []
    for geo in geo_pages or []:
        links.append(InternalLink(
            from_page_id=geo.parent_pillar_id,
            to_page_id=geo.id,
            anchor_text=f"{geo.title}"[:70],
            relationship=LinkRelationship.PILLAR_TO_GEO,
            reasoning="Pillar to regional variant.",
        ))
        links.append(InternalLink(
            from_page_id=geo.id,
            to_page_id=geo.parent_pillar_id,
            anchor_text=f"our full {geo.title.split(' in ')[0]} services"[:70],
            relationship=LinkRelationship.GEO_TO_PILLAR,
            reasoning="Geo page to parent pillar.",
        ))
    return links


# ── LLM: entity bridges only ──────────────────────────────────────────────────

def _generate_entity_bridges(pillars: list[Pillar]) -> list[InternalLink]:
    """
    ONE LLM call for the entire map — entity bridges only.
    Sends a compact map (pillar ID + title + top entities + cluster IDs).
    """
    # Build compact map — only what's needed for bridge reasoning
    compact_map = []
    for pillar in pillars:
        compact_map.append({
            "pillar_id":    pillar.id,
            "pillar_title": pillar.title,
            "entities":     pillar.related_entities[:5],
            "clusters": [
                {"id": c.id, "title": c.title, "entities": c.related_entities[:3]}
                for c in pillar.clusters
            ],
        })

    user_message = f"""Generate entity bridge links for this topical map.

```json
{json.dumps(compact_map, indent=2)}
```

MANDATORY: Generate at least 2 entity bridge links per pillar.
Total minimum: {len(pillars) * 2} bridges for {len(pillars)} pillars.
Each bridge: from_page_id must be a cluster_id, to_page_id must be a DIFFERENT pillar_id.
relationship_strength: decimal like 0.85 (NOT text like strong).
Do NOT return empty links array.
Output ONLY valid JSON."""

    try:
        response = call_anthropic_structured(
            system_prompt=_bridge_system_prompt(),
            user_message=user_message,
            response_model=_BridgeResponse,
            max_tokens=4000,
            stage="Stage 7 — Entity bridges",
        )
        # Validate: every bridge must reference real cluster→different pillar
        valid_cluster_ids = {c.id for p in pillars for c in p.clusters}
        valid_pillar_ids  = {p.id for p in pillars}
        cluster_to_pillar = {c.id: p.id for p in pillars for c in p.clusters}

        clean: list[InternalLink] = []
        for link in response.links:
            if link.from_page_id not in valid_cluster_ids:
                continue
            if link.to_page_id not in valid_pillar_ids:
                continue
            if cluster_to_pillar.get(link.from_page_id) == link.to_page_id:
                continue  # not a cross-pillar bridge
            link.relationship = LinkRelationship.ENTITY_BRIDGE
            clean.append(link)
        return clean
    except Exception as e:
        print(f"  [linking] LLM entity-bridge generation failed: {e}. Falling back to deterministic bridges.")
        return []


def _deterministic_entity_bridges(pillars: list[Pillar], min_per_pillar: int = 2) -> list[InternalLink]:
    """
    Build entity bridges purely from shared entities between clusters and other pillars.
    Used as a fallback (or supplement) to LLM bridges so the linking plan is never empty.
    """
    links: list[InternalLink] = []
    seen: set[tuple[str, str]] = set()

    def _norm(e: str) -> str:
        return e.strip().lower()

    pillar_entities = {
        p.id: {_norm(e) for e in (p.related_entities or [])}
        for p in pillars
    }

    for pillar in pillars:
        bridges_for_this_pillar = 0
        target_min = min_per_pillar

        # Pass 1: real entity overlap (cluster ↔ different pillar)
        for cluster in pillar.clusters:
            if bridges_for_this_pillar >= target_min:
                break
            cluster_ents = {_norm(e) for e in (cluster.related_entities or [])} \
                           | {_norm(cluster.title)}
            for other in pillars:
                if other.id == pillar.id:
                    continue
                shared = cluster_ents & pillar_entities[other.id]
                if not shared:
                    continue
                entity = sorted(shared)[0].title()
                key = (cluster.id, other.id)
                if key in seen:
                    continue
                seen.add(key)
                links.append(InternalLink(
                    from_page_id=cluster.id,
                    to_page_id=other.id,
                    anchor_text=f"{entity} considerations"[:55],
                    relationship=LinkRelationship.ENTITY_BRIDGE,
                    reasoning=f"Shared {entity} entity bridge.",
                    relationship_strength=0.85,
                ))
                bridges_for_this_pillar += 1
                if bridges_for_this_pillar >= target_min:
                    break

        # Pass 2: if still under target, force-link first clusters to first other pillars (weak bridges)
        if bridges_for_this_pillar < target_min and pillar.clusters:
            others = [p for p in pillars if p.id != pillar.id]
            ci = 0
            oi = 0
            attempts = 0
            while bridges_for_this_pillar < target_min and attempts < 20:
                attempts += 1
                cluster = pillar.clusters[ci % len(pillar.clusters)]
                other = others[oi % len(others)] if others else None
                if other is None:
                    break
                key = (cluster.id, other.id)
                if key not in seen:
                    seen.add(key)
                    links.append(InternalLink(
                        from_page_id=cluster.id,
                        to_page_id=other.id,
                        anchor_text=other.title[:55],
                        relationship=LinkRelationship.ENTITY_BRIDGE,
                        reasoning="Topical adjacency bridge.",
                        relationship_strength=0.45,
                    ))
                    bridges_for_this_pillar += 1
                ci += 1
                if ci % len(pillar.clusters) == 0:
                    oi += 1
    return links


# ── Main builder ──────────────────────────────────────────────────────────────

def build_linking_plan(pillars: list[Pillar], geo_pages: list | None = None) -> LinkingPlan:
    """
    Build the complete internal linking plan.

    Deterministic (0 tokens): pillar↔cluster, cluster↔supplementary,
                              pillar↔geo, homepage
    LLM (1 call, ~3k tokens): entity bridges across pillars
    """
    all_links: list[InternalLink] = []

    # Deterministic — free
    all_links.extend(_pillar_cluster_links(pillars))
    all_links.extend(_supplementary_links(pillars))
    all_links.extend(_geo_page_links(geo_pages))
    homepage_links = _homepage_links(pillars)

    # LLM — one call only
    bridges = _generate_entity_bridges(pillars)

    # Count bridges per pillar; top up with deterministic bridges where short
    bridge_count: dict[str, int] = {p.id: 0 for p in pillars}
    cluster_to_pillar = {c.id: p.id for p in pillars for c in p.clusters}
    for b in bridges:
        owner = cluster_to_pillar.get(b.from_page_id)
        if owner is not None:
            bridge_count[owner] += 1

    pillars_needing_more = [p for p in pillars if bridge_count[p.id] < 2]
    if pillars_needing_more:
        print(f"  [linking] Topping up entity bridges for {len(pillars_needing_more)} pillars via deterministic fallback")
        fallback = _deterministic_entity_bridges(pillars_needing_more, min_per_pillar=2)
        bridges.extend(fallback)

    all_links.extend(bridges)

    # Deduplicate
    seen: set[tuple] = set()
    deduped: list[InternalLink] = []
    for link in all_links:
        key = (link.from_page_id, link.to_page_id, link.relationship)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(link)

    print(f"  [linking] Total entity bridges: {sum(1 for l in deduped if l.relationship == LinkRelationship.ENTITY_BRIDGE)}")

    return LinkingPlan(links=deduped, homepage_links=homepage_links)
