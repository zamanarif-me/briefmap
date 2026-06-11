"""
Stage 3: Topic Expansion.

The most important call in the pipeline. Given the central entity, supporting
entity, source context, and intake, generate 8-12 pillars with 6-10 clusters each.

If this produces weak pillars, everything downstream is weak. Evaluate this
stage against the WordPress fixture before trusting downstream stages.
"""

import json

from pydantic import BaseModel

from models import SeedInput, CentralEntity, Pillar
from stages._client import call_structured, load_prompt


class TopicExpansionResponse(BaseModel):
    """LLM response wrapper — just a list of pillars at this stage."""
    pillars: list[Pillar]


def expand_topics(seed: SeedInput, central: CentralEntity) -> list[Pillar]:
    """Run stage 3. Returns the list of pillars (with clusters but no queries/supplementary yet)."""
    system_prompt = load_prompt("topic_expansion")

    user_message = f"""Generate the topical map pillars and clusters for this site.

# Central Entity
{central.primary}

# Supporting Authority Entity
{central.supporting}

# Source Context
{central.source_context}

# Key Entities
{', '.join(central.key_entities)}

# Reasoning Behind These Choices
{central.reasoning}

# User Intake
```json
{json.dumps(seed.intake.model_dump(mode="json"), indent=2)}
```

# Seed Keyword
{seed.seed_keyword}

Generate 8-12 pillars with 6-10 clusters each, following all rules in the system prompt. Pay special attention to:
- At least 60% commercial intent pillars
- Priority assignment based on revenue services and focus areas
- Geographic pillars if geo targeting is specified
- No generic or duplicate pillars

Output ONLY valid JSON matching the schema."""

    # Structural constraints (with tolerance — the prompt asks for 8-12
    # pillars / 6-10 clusters; we accept 6-14 / 4-12 before failing).
    # On violation, retry ONCE with the violation named, instead of
    # throwing away a paid call immediately.
    def _constraint_violation(pillars) -> str | None:
        pillar_count = len(pillars)
        if not (6 <= pillar_count <= 14):
            return (
                f"You returned {pillar_count} pillars but the requirement is 8-12 "
                f"(tolerance 6-14). Pillars: {[p.title for p in pillars]}"
            )
        for pillar in pillars:
            cluster_count = len(pillar.clusters)
            if not (4 <= cluster_count <= 12):
                return (
                    f"Pillar '{pillar.title}' has {cluster_count} clusters but the "
                    f"requirement is 6-10 (tolerance 4-12)."
                )
        return None

    violation: str | None = None
    for attempt in range(2):
        message = user_message
        if violation:
            message += (
                f"\n\nIMPORTANT: Your previous response violated a structural rule: "
                f"{violation} Regenerate the FULL output respecting the pillar and "
                f"cluster count rules."
            )
        response = call_structured(
            system_prompt=system_prompt,
            user_message=message,
            response_model=TopicExpansionResponse,
            max_tokens=16000,  # this stage produces a lot of output
            stage="Stage 3 — Topic expansion",
        )
        violation = _constraint_violation(response.pillars)
        if violation is None:
            return response.pillars

    raise ValueError(
        f"Topic expansion failed structural validation after retry: {violation}"
    )
