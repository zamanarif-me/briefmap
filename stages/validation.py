"""
Stage 4: Topic Validation (data-driven, relevance-based)

Validates each pillar and cluster against SERP data from Stage 3.5.

Old logic counted organic results ("Google returned >= 5 results" → strong),
which is true for almost any string — the signal was noise. The new logic
scores RELEVANCE:

  Pillars:
    - How many of the top organic results actually cover the topic
      (token overlap between the pillar title and result title+snippet)?
    - PAA + related-search counts (evidence of active search behavior).

  Clusters:
    - Token overlap between the cluster title and the parent pillar's
      SERP corpus (titles, snippets, PAA, related searches). A cluster
      whose entities appear in the live SERP for its pillar has real
      footprint — no extra Serper calls needed.

If no SERP data is available at all, falls back to one batched
Gemini Flash call for all pillars (previously one call per pillar).
"""

import re
from typing import Literal

from models import Pillar
from stages.serp import SerpData
from stages._client import call_gemini_flash_structured, load_prompt
from pydantic import BaseModel


class TopicValidation(BaseModel):
    topic_id: str
    topic_title: str
    signal: Literal["strong", "medium", "weak"]
    reasoning: str


class ValidationResponse(BaseModel):
    validations: list[TopicValidation]


# ── Relevance helpers ─────────────────────────────────────────────────────────

_STOPWORDS = {
    "the", "a", "an", "for", "to", "of", "in", "on", "and", "or", "is",
    "are", "with", "your", "my", "how", "what", "why", "services", "service",
    "best", "guide",
}


def _tokens(text: str) -> set[str]:
    return {
        w for w in re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()
        if len(w) > 2 and w not in _STOPWORDS
    }


def _data_driven_signal(serp: SerpData) -> tuple[str, str]:
    """
    Pillar signal from Serper data — relevance-scored, no LLM needed.
    Returns (signal, reasoning).
    """
    topic_tokens = _tokens(serp.query)
    if not topic_tokens:
        return "medium", "Topic title too generic to score — defaulting to medium."

    relevant = 0
    for result in serp.organic[:8]:
        result_tokens = _tokens(f"{result.title} {result.snippet}")
        overlap = len(topic_tokens & result_tokens) / len(topic_tokens)
        if overlap >= 0.5:
            relevant += 1

    paa_count     = len(serp.paa)
    related_count = len(serp.related_searches)

    if relevant >= 4 or (relevant >= 3 and paa_count >= 2):
        signal = "strong"
        reasoning = (
            f"{relevant} of the top results directly cover this topic; "
            f"{paa_count} PAA questions and {related_count} related searches "
            f"indicate active search behavior."
        )
    elif relevant >= 2 or (relevant >= 1 and paa_count >= 3):
        signal = "medium"
        reasoning = (
            f"{relevant} clearly relevant results in the top organic set — "
            f"topic exists but SERP coverage is partial ({paa_count} PAA)."
        )
    else:
        signal = "weak"
        reasoning = (
            f"Only {relevant} of the top results actually cover this topic "
            f"({len(serp.organic)} returned overall) — little direct footprint."
        )

    return signal, reasoning


def _cluster_signal(cluster_title: str, serp: SerpData) -> tuple[str, str]:
    """
    Cluster signal from the PARENT pillar's SERP corpus — zero extra calls.
    """
    cluster_tokens = _tokens(cluster_title)
    if not cluster_tokens:
        return "medium", "Cluster title too generic to score — defaulting to medium."

    corpus_parts = (
        [f"{r.title} {r.snippet}" for r in serp.organic]
        + serp.paa
        + serp.related_searches
        + ([serp.featured_snippet] if serp.featured_snippet else [])
    )
    corpus_tokens = _tokens(" ".join(corpus_parts))

    coverage = len(cluster_tokens & corpus_tokens) / len(cluster_tokens)

    if coverage >= 0.7:
        return "strong", (
            f"{int(coverage*100)}% of the cluster's core terms appear in the live "
            f"SERP for its parent pillar (titles, snippets, PAA)."
        )
    if coverage >= 0.4:
        return "medium", (
            f"{int(coverage*100)}% of the cluster's core terms appear in the parent "
            f"pillar's SERP — partial footprint."
        )
    return "weak", (
        f"Only {int(coverage*100)}% of the cluster's core terms appear in the parent "
        f"pillar's SERP — verify demand before prioritizing."
    )


# ── Main entry ────────────────────────────────────────────────────────────────

def validate_topics(
    pillars: list[Pillar],
    serp_data: dict[str, SerpData] | None = None,
) -> dict[str, TopicValidation]:
    """
    Validate all pillars and clusters.

    If serp_data is provided (from Stage 3.5), uses relevance-scored
    data-driven signals — fast and free.
    If serp_data is None, falls back to ONE batched Gemini Flash call.
    """
    validations: dict[str, TopicValidation] = {}

    if serp_data:
        for pillar in pillars:
            serp = serp_data.get(pillar.id)
            if serp:
                signal, reasoning = _data_driven_signal(serp)
            else:
                signal, reasoning = "medium", "No SERP data available — defaulting to medium."

            validations[pillar.id] = TopicValidation(
                topic_id=pillar.id,
                topic_title=pillar.title,
                signal=signal,
                reasoning=reasoning,
            )

            for cluster in pillar.clusters:
                if serp:
                    c_signal, c_reasoning = _cluster_signal(cluster.title, serp)
                else:
                    c_signal, c_reasoning = "medium", "No SERP data for parent pillar."
                validations[cluster.id] = TopicValidation(
                    topic_id=cluster.id,
                    topic_title=cluster.title,
                    signal=c_signal,
                    reasoning=c_reasoning,
                )
    else:
        # Fallback: one batched Gemini Flash LLM validation
        _validate_with_gemini(pillars, validations)

    # Ensure every topic has an entry
    for pillar in pillars:
        if pillar.id not in validations:
            validations[pillar.id] = TopicValidation(
                topic_id=pillar.id,
                topic_title=pillar.title,
                signal="medium",
                reasoning="Validation not completed — defaulting to medium.",
            )
        for cluster in pillar.clusters:
            if cluster.id not in validations:
                validations[cluster.id] = TopicValidation(
                    topic_id=cluster.id,
                    topic_title=cluster.title,
                    signal="medium",
                    reasoning="Validation not completed — defaulting to medium.",
                )

    return validations


def _validate_with_gemini(
    pillars: list[Pillar],
    validations: dict[str, TopicValidation],
) -> None:
    """Gemini Flash fallback validation — ONE call for the whole map."""
    import json
    system_prompt = load_prompt("web_validation")

    batch: list[dict] = []
    for pillar in pillars:
        batch.append({"id": pillar.id, "title": pillar.title, "type": "pillar"})
        batch.extend(
            {"id": c.id, "title": c.title, "type": "cluster"}
            for c in pillar.clusters
        )

    user_message = f"""Validate these SEO topics. For each one, assess whether it has real-world search demand.
Output ONLY valid JSON.

Topics to validate:
{json.dumps(batch, indent=2)}"""

    try:
        response = call_gemini_flash_structured(
            system_prompt=system_prompt,
            user_message=user_message,
            response_model=ValidationResponse,
            stage="Stage 4 — Validation (LLM fallback)",
        )
        for v in response.validations:
            validations[v.topic_id] = v
    except Exception as e:
        for item in batch:
            if item["id"] not in validations:
                validations[item["id"]] = TopicValidation(
                    topic_id=item["id"],
                    topic_title=item["title"],
                    signal="medium",
                    reasoning=f"Gemini validation failed: {type(e).__name__}",
                )
