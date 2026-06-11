# BriefMap

A Python engine that generates Koray-style semantic topical maps, query networks, and SERP-grounded content briefs from a seed keyword and 8 intake answers. Hybrid: Anthropic Sonnet for reasoning, Gemini Flash for validation fallback + supplementary nodes, Serper.dev for SERP/PAA intelligence.

## Deploy to Streamlit Cloud

1. **Push to GitHub** — create a new repo (recommended name: `briefmap`) and push the contents of this directory.
2. **Create a Streamlit Cloud app** at https://share.streamlit.io
   - Repository: `<your-username>/briefmap`
   - Branch: `main`
   - Main file: `app.py`
   - Python version: pinned to `3.11` via `runtime.txt`
3. **Add secrets** — in the app's **Settings → Secrets**, paste:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   GEMINI_API_KEY    = "AIza..."
   SERPER_API_KEY    = "..."
   ```
   See `.streamlit/secrets.toml.example` for the template. Streamlit Cloud reads `st.secrets` as `os.environ` automatically — no code changes needed.
4. **Deploy.** First boot takes ~60 seconds for dependency install.

### Local development

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then fill in real keys
streamlit run app.py
```

### Notes

- Streamlit Cloud has an **ephemeral filesystem** — saved sessions and outputs reset on app restart. For persistent storage, mount S3/GCS later.
- Brief generation runs in a **background thread** to avoid request timeouts; each finished brief is checkpointed to disk, so re-running after a disconnect skips already-generated briefs.
- All three API keys are required: `ANTHROPIC_API_KEY` (stages 2/3/5/7/9), `GEMINI_API_KEY` (stages 4 fallback + 6), `SERPER_API_KEY` (stage 3.5).
- Cost: roughly `$0.05–0.10` per content brief (prompt caching + structured outputs), `$0.25–0.50` per full topical map.

---

## What this engine produces

### What it produces

Three deliverables, generated in eight stages, output as both structured JSON and a human-readable Markdown report:

1. **Topical Map** — pillars (Tier 1), clusters (Tier 2), supplementary nodes (Tier 3), all tagged with intent and funnel stage, plus geographic service pages
2. **Query Network** — representative queries (broad, pillar-level) and represented queries (long-tail, cluster-level)
3. **Internal Linking Plan** — directed graph of pillar↔cluster, cluster↔supplementary, and entity-bridge cross-pillar links, each with anchor text and a one-sentence reasoning

## Build status — all 8 stages complete

| Stage | What it does | Status |
|------|-------------|--------|
| 1. Intake | Load seed + 8 intake answers | ✅ |
| 2. Central Entity | Extract primary entity, supporting entity, source context | ✅ |
| 3. Topic Expansion | Generate 8-12 pillars with 6-10 clusters each | ✅ |
| 4. Web Validation | Confirm topics have real-world footprint via web_search | ✅ |
| 5. Query Generation | Generate representative + represented queries per cluster | ✅ |
| 6. Supplementary Nodes | Generate Tier 3 supporting nodes per cluster | ✅ |
| 7. Internal Linking | Build the directed link graph + anchor text + reasoning | ✅ |
| 8. Render | Serialize to JSON + render Markdown report | ✅ |

## Project structure

```
topical_map_engine/
├── notebook.ipynb              # Colab entry point — runs the full pipeline
├── pipeline.py                 # Top-level orchestrator for all 8 stages
├── models.py                   # Pydantic data model (the spine)
├── prompts/
│   ├── central_entity.txt      # Stage 2
│   ├── topic_expansion.txt     # Stage 3
│   ├── web_validation.txt      # Stage 4
│   ├── query_generation.txt    # Stage 5
│   ├── supplementary.txt       # Stage 6
│   └── internal_linking.txt    # Stage 7
├── stages/
│   ├── _client.py              # Shared Anthropic client + JSON parsing
│   ├── intake.py               # Stage 1
│   ├── central_entity.py       # Stage 2
│   ├── expansion.py            # Stage 3
│   ├── validation.py           # Stage 4
│   ├── queries.py              # Stage 5
│   ├── tiering.py              # Stage 6 (supplementary nodes)
│   ├── linking.py              # Stage 7
│   ├── geo.py                  # Geographic page derivation (deterministic)
│   └── render.py               # Stage 8
├── templates/
│   └── report.md.j2            # Jinja2 template for the Markdown report
└── examples/
    └── wordpress_dev/
        ├── input.json          # WordPress fixture (from the spec)
        └── expected_output.md  # Hand-written reference for evaluation
```

## How to run

### Colab (recommended)

1. Upload `topical_map_engine_v1.zip` to Colab via the Files panel
2. Open `notebook.ipynb`
3. Run all cells (the setup cell unzips and configures everything automatically)

### Locally

```bash
pip install pydantic anthropic jinja2
export ANTHROPIC_API_KEY=sk-ant-...
cd topical_map_engine

# Run the full pipeline
python -m pipeline examples/wordpress_dev/input.json examples/wordpress_dev/output
```

Output appears at `examples/wordpress_dev/output/topical_map.json` and `topical_map_report.md`.

## Cost per run

| Stage | Model | API calls | Approx cost |
|-------|-------|-----------|-------------|
| 2 — Central Entity | Sonnet | 1 | $0.01 |
| 3 — Topic Expansion | Sonnet | 1 | $0.05-0.10 |
| 3.5 — SERP pull | Serper | ~11 (one per pillar) | free tier |
| 4 — Validation | data-driven (free) / Gemini Flash fallback | 0-1 | ~$0.00 |
| 5 — Query Generation | Sonnet (batched, cached) | ~2 | $0.05-0.10 |
| 6 — Supplementary | Gemini Flash | ~11 | $0.01-0.02 |
| 7 — Internal Linking | deterministic + 1 Sonnet call (bridges) | 1 | $0.03-0.05 |
| 8 — Render | none | 0 | free |
| **Total** | | ~17 | **~$0.25-0.50** |

Anthropic calls use prompt caching and forced tool-use (structured outputs), which cuts repeated-context input cost ~90% and eliminates most JSON-retry waste. Skip stage 3.5 (`skip_serp=True`) during prompt iteration to save Serper quota — stage 4 then falls back to one Gemini Flash call.

## Honest limitations

- **No keyword volumes.** v1 has no DataForSEO / Ahrefs / Semrush integration, so cluster/pillar tiering is based on LLM judgment of commercial value, not search volume. Web validation in stage 4 is the strongest free signal we have, but it tells you "topic exists," not "topic has 2,900 searches."
- **No content audit branch.** The existing-site CSV input pathway is not implemented in v1 — we agreed to defer it after deciding the CSV format would be a v2 concern.
- **No content writing.** v1 produces the map, queries, and linking plan. Article briefs and full articles are v2.
- **Humanized content.** v1 doesn't address this directly — that's a v2 problem solved by content brief + style guide + human edit pass, not by clever prompting.

## When something goes wrong

- **Stage 4 fails on a pillar:** the pipeline marks that pillar's topics as `medium` signal and continues. You'll see the issue in the per-stage logs.
- **Stage 7 produces an enormous output:** linking is generated pillar-by-pillar, then merged and deduplicated. Per-pillar calls are capped at 16k tokens. If a pillar has too many clusters, the call may truncate — inspect the per-pillar checkpoint files in the output directory.
- **A stage crashes mid-run:** intermediate state is saved as `_checkpoint_stageN.json` in the output directory. You can resume manually from these.

## What v2 would add

Already designed for additivity (no schema rewrites needed):

- Content audit of existing sites (CSV → gap analysis against the topical map)
- Real keyword volumes via DataForSEO (`Query.volume: int | None` field already in the model)
- Content briefs (new stage 9, doesn't touch stages 1-8)
- AI retrieval scoring (does Claude/ChatGPT surface your URLs for the represented queries?)
- Streamlit UI to replace the Colab notebook

## Evaluation against the WordPress fixture

Stages 1-3 already scored 9/10 against the hand-written reference in `examples/wordpress_dev/expected_output.md`. After running the full pipeline, you should see:

- 8-12 pillars (target: 11)
- 60-80 clusters total
- 200-450 supplementary nodes
- 300-500 queries
- 400-800 internal links

If the numbers are wildly outside these ranges, something went wrong — inspect the checkpoint files to find which stage drifted.
