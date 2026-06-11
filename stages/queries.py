"""
Stage 5: Query Generation — Batched (cost optimized) + real PAA merge

- Uses the full Koray query-semantics prompt (prompts/query_generation.txt)
  with a batch-output addendum, instead of a stripped-down inline prompt.
- Real Serper PAA questions are merged DETERMINISTICALLY into the clusters'
  represented queries (real user queries beat LLM-invented phrasings, $0).
- A failed batch is retried pillar-by-pillar instead of being dropped
  silently; if every pillar still ends up without queries the stage raises
  so the failure cannot be checkpointed as success.
"""

import json
import re
from pydantic import BaseModel, Field

from models import Pillar, Query, QueryType, Intent
from stages._client import call_anthropic_structured, load_prompt
from stages.serp import SerpData


class _RawQuery(BaseModel):
    text: str
    intent: Intent


class _RawClusterQueries(BaseModel):
    cluster_id: str = ""
    represented_queries: list[_RawQuery] = Field(default_factory=list)


class _PillarQueries(BaseModel):
    pillar_id: str
    representative_queries: list[_RawQuery] = Field(default_factory=list)
    clusters: list[_RawClusterQueries] = Field(default_factory=list)


class BatchQueryResponse(BaseModel):
    pillars: list[_PillarQueries]


# The full Koray prompt (definitions, intent classification, anti-patterns)
# is in prompts/query_generation.txt. This addendum adapts its single-pillar
# output schema to the batched multi-pillar call.
_BATCH_ADDENDUM = """

# THIS RUN — BATCHED OUTPUT

You will receive MULTIPLE pillars at once. Apply all the rules above to each
pillar independently, then wrap the per-pillar outputs in this envelope:

{
  "pillars": [
    {
      "pillar_id": "<pillar_id from the input>",
      "representative_queries": [
        {"text": "query text", "intent": "commercial"}
      ],
      "clusters": [
        {
          "cluster_id": "<cluster_id from the input>",
          "represented_queries": [
            {"text": "long tail query", "intent": "informational"}
          ]
        }
      ]
    }
  ]
}

- Include EVERY pillar and EVERY cluster you were given.
- Use the exact pillar_id / cluster_id values from the input.
- If real PAA questions are provided for a pillar, prefer covering their
  intent in your queries rather than inventing similar phrasings.
- Output ONLY valid JSON matching the envelope above."""


def _query_system_prompt() -> str:
    return load_prompt("query_generation") + _BATCH_ADDENDUM


# ── Context builder ───────────────────────────────────────────────────────────

def _build_pillar_context(
    pillar: Pillar,
    serp_data: dict[str, SerpData] | None,
) -> dict:
    """Compact pillar representation for the batch prompt."""
    ctx = {
        "pillar_id":    pillar.id,
        "title":        pillar.title,
        "intent":       pillar.intent.value,
        "entities":     pillar.related_entities[:4],
        "clusters": [
            {"cluster_id": c.id, "title": c.title, "intent": c.intent.value}
            for c in pillar.clusters
        ],
    }
    # Add PAA if available
    if serp_data and pillar.id in serp_data:
        paa = serp_data[pillar.id].paa[:5]
        if paa:
            ctx["paa_questions"] = paa
    return ctx


# ── Deterministic PAA merge ───────────────────────────────────────────────────

_STOPWORDS = {
    "the", "a", "an", "for", "to", "of", "in", "on", "and", "or", "is",
    "are", "what", "how", "why", "do", "does", "can", "should", "my",
    "your", "with", "you", "i", "it",
}


def _tokens(text: str) -> set[str]:
    return {
        w for w in re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()
        if w and w not in _STOPWORDS
    }


def _merge_paa_queries(pillar: Pillar, serp: SerpData, max_per_cluster: int = 2) -> int:
    """
    Attach real PAA questions to their best-matching cluster as represented
    queries. Free, deterministic, and grounded in real search behavior —
    the engine's own anti-pattern list warns against LLM-invented phrasings.
    Returns the number of queries added.
    """
    if not serp.paa or not pillar.clusters:
        return 0

    existing = {
        q.text.lower().strip()
        for c in pillar.clusters for q in c.represented_queries
    }
    added_per_cluster: dict[str, int] = {c.id: 0 for c in pillar.clusters}
    added = 0

    for question in serp.paa:
        q_norm = question.lower().strip().rstrip("?")
        if not q_norm or q_norm in existing:
            continue
        q_tokens = _tokens(question)
        if not q_tokens:
            continue

        best_cluster = None
        best_score = 0.0
        for cluster in pillar.clusters:
            overlap = len(q_tokens & _tokens(cluster.title)) / len(q_tokens)
            if overlap > best_score:
                best_cluster, best_score = cluster, overlap

        # Low-overlap questions still belong to the pillar topic — attach to
        # the first cluster only if nothing matched at all is too noisy; skip.
        if best_cluster is None or best_score < 0.2:
            continue
        if added_per_cluster[best_cluster.id] >= max_per_cluster:
            continue

        best_cluster.represented_queries.append(Query(
            text=q_norm,
            type=QueryType.REPRESENTED,
            intent=Intent.INFORMATIONAL,
            parent_cluster_id=best_cluster.id,
        ))
        existing.add(q_norm)
        added_per_cluster[best_cluster.id] += 1
        added += 1

    return added


# ── LLM result mapping ────────────────────────────────────────────────────────

def _apply_result(pillar: Pillar, result: _PillarQueries) -> None:
    pillar.representative_queries = [
        Query(
            text=q.text,
            type=QueryType.REPRESENTATIVE,
            intent=q.intent,
            parent_cluster_id=pillar.id,
        )
        for q in result.representative_queries
    ]

    cluster_lookup = {c.id: c for c in pillar.clusters}
    for cq in result.clusters:
        cluster = cluster_lookup.get(cq.cluster_id)
        if not cluster:
            continue
        cluster.represented_queries = [
            Query(
                text=q.text,
                type=QueryType.REPRESENTED,
                intent=q.intent,
                parent_cluster_id=cluster.id,
            )
            for q in cq.represented_queries
        ]


def _call_for_pillars(
    batch: list[Pillar],
    serp_data: dict[str, SerpData] | None,
) -> dict[str, _PillarQueries]:
    """One LLM call for a batch of pillars. Returns pillar_id → result."""
    batch_ctx = [_build_pillar_context(p, serp_data) for p in batch]
    user_message = f"""Generate queries for these {len(batch)} pillars.

```json
{json.dumps(batch_ctx, indent=2)}
```

Output ONLY valid JSON with all {len(batch)} pillars in the response."""

    response = call_anthropic_structured(
        system_prompt=_query_system_prompt(),
        user_message=user_message,
        response_model=BatchQueryResponse,
        max_tokens=6000,
        stage="Stage 5 — Queries",
    )
    return {r.pillar_id: r for r in response.pillars}


# ── Main entry ────────────────────────────────────────────────────────────────

def generate_queries_for_all(
    pillars: list[Pillar],
    serp_data: dict[str, SerpData] | None = None,
    batch_size: int = 5,
) -> list[Pillar]:
    """
    Generate queries for all pillars in batches.
    batch_size=5 means 2 calls for 10 pillars instead of 10 calls.

    Failure policy: a failed batch is retried one pillar at a time. If a
    pillar still has no representative queries afterwards a warning is
    printed; if EVERY pillar is empty the stage raises (so the pipeline
    cannot checkpoint a queryless map as success).
    """
    for i in range(0, len(pillars), batch_size):
        batch = pillars[i : i + batch_size]
        try:
            result_map = _call_for_pillars(batch, serp_data)
            for pillar in batch:
                result = result_map.get(pillar.id)
                if result:
                    _apply_result(pillar, result)
        except Exception as e:
            print(f"  [queries] Batch {i//batch_size + 1} failed: {e} — retrying per pillar")
            for pillar in batch:
                try:
                    result_map = _call_for_pillars([pillar], serp_data)
                    result = result_map.get(pillar.id)
                    if result:
                        _apply_result(pillar, result)
                except Exception as e2:
                    print(f"  [queries] Pillar '{pillar.title[:40]}' failed: {e2}")

    # Merge real PAA questions ($0, deterministic)
    if serp_data:
        merged = 0
        for pillar in pillars:
            serp = serp_data.get(pillar.id)
            if serp:
                merged += _merge_paa_queries(pillar, serp)
        if merged:
            print(f"  [queries] Merged {merged} real PAA questions into represented queries")

    # Guard against silent total failure
    empty = [p.title for p in pillars if not p.representative_queries]
    if empty and len(empty) == len(pillars):
        raise RuntimeError(
            "Stage 5 produced no queries for any pillar — refusing to "
            "checkpoint an empty query network. Check API keys / model output."
        )
    if empty:
        print(f"  [queries] WARNING: {len(empty)} pillars have no representative queries: {empty[:3]}")

    return pillars
