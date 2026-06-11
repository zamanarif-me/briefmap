"""
Stage 3.5: SERP Intelligence via Serper.dev

For each pillar, pulls:
  - Top 10 organic results (URL + title + snippet)
  - People Also Ask (PAA) questions
  - Related searches
  - Featured snippet (if present)

This data feeds directly into:
  - Stage 5  (query generation — PAA becomes represented queries)
  - Stage 6  (supplementary nodes — related searches suggest topics)
  - Stage 9  (content brief — competitor headings + PAA are DATA 2 + DATA 5)

Serper.dev free tier: 2,500 calls/month.
One topical map with 11 pillars = 11 calls.
Free tier supports ~225 full client maps per month.
"""

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import requests


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class OrganicResult:
    position: int
    title: str
    url: str
    snippet: str


@dataclass
class SerpData:
    """All SERP data for one pillar query."""
    pillar_id: str
    query: str
    organic: list[OrganicResult] = field(default_factory=list)
    paa: list[str] = field(default_factory=list)           # People Also Ask
    related_searches: list[str] = field(default_factory=list)
    featured_snippet: str = ""


# ── Serper client ─────────────────────────────────────────────────────────────

def _get_serper_key() -> str:
    key = os.environ.get("SERPER_API_KEY")
    if not key:
        raise RuntimeError(
            "SERPER_API_KEY not set. "
            "In Colab Secrets: add key named SERPER_API_KEY and enable notebook access."
        )
    return key


def _search_one(query: str, gl: str = "us", hl: str = "en") -> dict:
    """Single Serper.dev search. Returns raw JSON response."""
    url     = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY":    _get_serper_key(),
        "Content-Type": "application/json",
    }
    payload = {
        "q":   query,
        "gl":  gl,      # country (us, gb, au …)
        "hl":  hl,      # language
        "num": 10,
    }
    response = requests.post(url, headers=headers, json=payload, timeout=15)
    response.raise_for_status()
    return response.json()


def _parse_serp(pillar_id: str, query: str, raw: dict) -> SerpData:
    """Convert raw Serper JSON into a clean SerpData object."""
    data = SerpData(pillar_id=pillar_id, query=query)

    # Organic results
    for item in raw.get("organic", []):
        data.organic.append(OrganicResult(
            position=item.get("position", 0),
            title=item.get("title", ""),
            url=item.get("link", ""),
            snippet=item.get("snippet", ""),
        ))

    # People Also Ask
    for item in raw.get("peopleAlsoAsk", []):
        question = item.get("question", "").strip()
        if question:
            data.paa.append(question)

    # Related searches
    for item in raw.get("relatedSearches", []):
        query_text = item.get("query", "").strip()
        if query_text:
            data.related_searches.append(query_text)

    # Featured snippet
    if "answerBox" in raw:
        data.featured_snippet = raw["answerBox"].get("answer", "") or raw["answerBox"].get("snippet", "")

    return data


# ── Main stage function ───────────────────────────────────────────────────────

def pull_serp_for_pillars(
    pillars: list,           # list[Pillar] — typed loosely to avoid circular import
    geo: str = "us",
    lang: str = "en",
    delay: float = 0.5,      # seconds between calls — be polite to the API
) -> dict[str, SerpData]:
    """
    Pull SERP data for every pillar.
    Returns dict keyed by pillar.id → SerpData.

    delay: pause between API calls to avoid rate limits.
    geo:   Serper country code — "us", "gb", "au" etc.
    lang:  language code — "en" etc.
    """
    results: dict[str, SerpData] = {}

    for i, pillar in enumerate(pillars):
        print(f"  [{i+1}/{len(pillars)}] SERP pull: {pillar.title}")
        try:
            raw  = _search_one(pillar.title, gl=geo, hl=lang)
            data = _parse_serp(pillar.id, pillar.title, raw)
            results[pillar.id] = data

            print(f"    organic: {len(data.organic)} | PAA: {len(data.paa)} | related: {len(data.related_searches)}")

            if i < len(pillars) - 1:
                time.sleep(delay)

        except requests.HTTPError as e:
            print(f"    WARNING: Serper returned HTTP {e.response.status_code} for '{pillar.title}'. Skipping.")
            results[pillar.id] = SerpData(pillar_id=pillar.id, query=pillar.title)

        except Exception as e:
            print(f"    WARNING: SERP pull failed for '{pillar.title}': {e}. Skipping.")
            results[pillar.id] = SerpData(pillar_id=pillar.id, query=pillar.title)

    return results


# ── Persistence (so briefs can use real SERP data later) ────────────────────

def save_serp_data(serp_data: dict[str, SerpData], path: str | Path) -> None:
    """Persist SERP data as JSON so Stage 9 (briefs) can ground itself in it."""
    payload = {
        pid: {
            "pillar_id":        s.pillar_id,
            "query":            s.query,
            "organic":          [asdict(o) for o in s.organic],
            "paa":              list(s.paa),
            "related_searches": list(s.related_searches),
            "featured_snippet": s.featured_snippet,
        }
        for pid, s in serp_data.items()
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_serp_data(path: str | Path) -> dict[str, SerpData] | None:
    """Load persisted SERP data. Returns None if missing/corrupt."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    result: dict[str, SerpData] = {}
    for pid, d in payload.items():
        sd = SerpData(pillar_id=d.get("pillar_id", pid), query=d.get("query", ""))
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
        result[pid] = sd
    return result


# ── Helpers used by downstream stages ────────────────────────────────────────

def get_paa_for_pillar(serp_data: dict[str, SerpData], pillar_id: str) -> list[str]:
    """Get PAA questions for a specific pillar. Returns empty list if not available."""
    return serp_data.get(pillar_id, SerpData("", "")).paa


def get_competitor_titles(serp_data: dict[str, SerpData], pillar_id: str) -> list[str]:
    """Get top competitor page titles for a pillar."""
    return [r.title for r in serp_data.get(pillar_id, SerpData("", "")).organic[:5]]


def get_related_searches(serp_data: dict[str, SerpData], pillar_id: str) -> list[str]:
    """Get related searches for a pillar."""
    return serp_data.get(pillar_id, SerpData("", "")).related_searches


def serp_data_to_summary(serp: SerpData) -> str:
    """
    Render SerpData as a compact text block for LLM prompts.
    Used in Stage 5 (queries) and Stage 9 (content brief).
    """
    lines = [f"# SERP Data — {serp.query}"]

    if serp.featured_snippet:
        lines += ["", "## Featured Snippet", serp.featured_snippet]

    if serp.organic:
        lines += ["", "## Top Ranking Pages"]
        for r in serp.organic[:5]:
            lines.append(f"{r.position}. {r.title}")
            lines.append(f"   {r.url}")
            if r.snippet:
                lines.append(f"   → {r.snippet[:150]}")

    if serp.paa:
        lines += ["", "## People Also Ask"]
        for q in serp.paa:
            lines.append(f"- {q}")

    if serp.related_searches:
        lines += ["", "## Related Searches"]
        for s in serp.related_searches:
            lines.append(f"- {s}")

    return "\n".join(lines)
