"""
Stage 2: Central Entity extraction.

Takes the SeedInput and produces a CentralEntity object that grounds all
downstream stages. This is the most semantically important call in the pipeline
after topic expansion — if the central entity is wrong, the whole map drifts.
"""

import json

from models import SeedInput, CentralEntity
from stages._client import call_structured, load_prompt


def extract_central_entity(seed: SeedInput) -> CentralEntity:
    """Run stage 2."""
    system_prompt = load_prompt("central_entity")

    # We hand the model the seed and intake as structured JSON, not prose.
    # This forces it to read fields explicitly rather than skim.
    user_message = f"""Here is the user's input. Identify the central entity, supporting entity, source context, and key entities.

# Seed Keyword
{seed.seed_keyword}

# Intake Answers
```json
{json.dumps(seed.intake.model_dump(mode="json"), indent=2)}
```

Output the JSON object as specified in the schema. No preamble."""

    return call_structured(
        system_prompt=system_prompt,
        user_message=user_message,
        response_model=CentralEntity,
        stage="Stage 2 — Central entity",
    )
