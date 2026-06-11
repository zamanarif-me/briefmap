"""
Brief Validator (Stage 9.1)

Validates all page IDs referenced inside content briefs against the
actual topical map. Catches hallucinated IDs before a writer follows
a broken internal link plan.

Checks:
  - semantic_bridges[].link_destination
  - next_destination.next_page_id

For each unknown ID:
  - Flags it clearly
  - Suggests the closest real match (fuzzy title match)
  - Returns a corrected brief

Usage:
    from stages.brief_validator import validate_brief, validate_all_briefs
    result = validate_brief(brief, topical_map)
    print(result.report())
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from models import TopicalMap
from stages.brief import ContentBrief, SemanticBridge, NextDestination


# ── Build ID registry from topical map ───────────────────────────────────────

@dataclass
class PageRegistry:
    """All valid page IDs and titles from the topical map."""
    pages: dict[str, str] = field(default_factory=dict)  # id → title

    @classmethod
    def from_topical_map(cls, tm: TopicalMap) -> PageRegistry:
        registry = cls()
        for pillar in tm.pillars:
            registry.pages[pillar.id] = pillar.title
            for cluster in pillar.clusters:
                registry.pages[cluster.id] = cluster.title
                for node in cluster.supplementary_nodes:
                    registry.pages[node.id] = node.title
        for geo in tm.geo_pages:
            registry.pages[geo.id] = geo.title
        return registry

    def is_valid(self, page_id: str) -> bool:
        return page_id in self.pages

    def find_closest(self, unknown_id: str, top_n: int = 3) -> list[tuple[str, str, float]]:
        """
        Find the closest real pages to an unknown ID.
        Returns list of (id, title, similarity_score).
        """
        # Strategy 1: slug similarity (compare ID strings)
        slug_scores: list[tuple[str, str, float]] = []
        for real_id, real_title in self.pages.items():
            score = SequenceMatcher(None, unknown_id, real_id).ratio()
            slug_scores.append((real_id, real_title, score))

        # Strategy 2: title keyword overlap
        # Extract keywords from the unknown ID
        unknown_words = set(re.sub(r'[_\-]', ' ', unknown_id).lower().split())
        unknown_words.discard('pillar')
        unknown_words.discard('cluster')
        unknown_words.discard('supp')

        title_scores: list[tuple[str, str, float]] = []
        for real_id, real_title in self.pages.items():
            real_words = set(real_title.lower().split())
            overlap = len(unknown_words & real_words)
            score = overlap / max(len(unknown_words), 1)
            title_scores.append((real_id, real_title, score))

        # Combine: take max of both strategies per page
        combined: dict[str, tuple[str, str, float]] = {}
        for real_id, real_title, score in slug_scores + title_scores:
            if real_id not in combined or score > combined[real_id][2]:
                combined[real_id] = (real_id, real_title, score)

        sorted_pages = sorted(combined.values(), key=lambda x: x[2], reverse=True)
        return sorted_pages[:top_n]


# ── Validation result ─────────────────────────────────────────────────────────

@dataclass
class ValidationIssue:
    field: str           # e.g. "semantic_bridges[2].link_destination"
    bad_id: str
    suggestions: list[tuple[str, str, float]]  # (id, title, score)


@dataclass
class BriefValidationResult:
    brief_id: str
    brief_title: str
    issues: list[ValidationIssue] = field(default_factory=list)
    corrected_brief: ContentBrief | None = None

    @property
    def is_valid(self) -> bool:
        return len(self.issues) == 0

    def report(self) -> str:
        lines = [
            f"Brief: {self.brief_title} ({self.brief_id})",
            f"Status: {'✅ VALID' if self.is_valid else f'⚠️  {len(self.issues)} ISSUE(S)'}",
        ]
        if self.issues:
            lines.append("")
            for issue in self.issues:
                lines.append(f"  Field: {issue.field}")
                lines.append(f"  Bad ID: '{issue.bad_id}'")
                if issue.suggestions:
                    lines.append(f"  Suggestions:")
                    for sid, stitle, score in issue.suggestions:
                        lines.append(f"    [{score:.2f}] {sid}")
                        lines.append(f"           \"{stitle}\"")
                lines.append("")
        return "\n".join(lines)


# ── Validator ─────────────────────────────────────────────────────────────────

def validate_brief(
    brief: ContentBrief,
    topical_map: TopicalMap,
    auto_correct: bool = True,
) -> BriefValidationResult:
    """
    Validate all page ID references in a brief against the topical map.

    auto_correct: if True, replace bad IDs with the top suggestion (score > 0.3).
    """
    registry = PageRegistry.from_topical_map(topical_map)
    issues: list[ValidationIssue] = []

    # Check semantic bridges
    for i, bridge in enumerate(brief.semantic_bridges):
        if not registry.is_valid(bridge.link_destination):
            suggestions = registry.find_closest(bridge.link_destination)
            issues.append(ValidationIssue(
                field=f"semantic_bridges[{i}].link_destination",
                bad_id=bridge.link_destination,
                suggestions=suggestions,
            ))

    # Check next destination
    if not registry.is_valid(brief.next_destination.next_page_id):
        suggestions = registry.find_closest(brief.next_destination.next_page_id)
        issues.append(ValidationIssue(
            field="next_destination.next_page_id",
            bad_id=brief.next_destination.next_page_id,
            suggestions=suggestions,
        ))

    # Auto-correct if requested
    corrected = None
    if auto_correct and issues:
        corrected = _auto_correct(brief, issues, registry)

    return BriefValidationResult(
        brief_id=brief.page_id,
        brief_title=brief.page_title,
        issues=issues,
        corrected_brief=corrected,
    )


def _auto_correct(
    brief: ContentBrief,
    issues: list[ValidationIssue],
    registry: PageRegistry,
) -> ContentBrief:
    """
    Replace bad IDs with the best suggestion if confidence > 0.3.

    When NO suggestion clears the confidence bar, the reference is DROPPED
    (bridge removed / next_destination cleared) rather than silently
    rewritten to an arbitrary page — a confidently wrong internal link is
    worse for the writer than an explicitly missing one. The issue stays
    in the validation report either way.
    """
    # Build correction map: bad_id → real_id, or None meaning "drop it"
    corrections: dict[str, str | None] = {}
    for issue in issues:
        if issue.suggestions and issue.suggestions[0][2] > 0.30:
            corrections[issue.bad_id] = issue.suggestions[0][0]
        else:
            corrections[issue.bad_id] = None

    # Deep copy via model_dump + reconstruct
    data = brief.model_dump(mode="json")

    # Fix bridges — drop those with no credible correction
    fixed_bridges = []
    for bridge in data["semantic_bridges"]:
        bad = bridge["link_destination"]
        if bad in corrections:
            replacement = corrections[bad]
            if replacement is None:
                continue  # drop the bridge entirely
            bridge["link_destination"] = replacement
        fixed_bridges.append(bridge)
    data["semantic_bridges"] = fixed_bridges

    # Fix next destination — clear it if no credible correction
    bad_next = data["next_destination"]["next_page_id"]
    if bad_next in corrections:
        replacement = corrections[bad_next]
        if replacement is None:
            data["next_destination"]["next_page_id"] = ""
            data["next_destination"]["next_page_title"] = ""
            data["next_destination"]["transition_reason"] = (
                "REMOVED — the model referenced a page that does not exist "
                "in the topical map. Pick a destination manually."
            )
        else:
            data["next_destination"]["next_page_id"] = replacement
            data["next_destination"]["next_page_title"] = registry.pages[replacement]

    return ContentBrief.model_validate(data)


# ── Batch validator ───────────────────────────────────────────────────────────

def validate_all_briefs(
    briefs: dict[str, ContentBrief],
    topical_map: TopicalMap,
    auto_correct: bool = True,
) -> dict[str, BriefValidationResult]:
    """Validate all briefs and return results keyed by page_id."""
    results: dict[str, BriefValidationResult] = {}
    for page_id, brief in briefs.items():
        result = validate_brief(brief, topical_map, auto_correct=auto_correct)
        results[page_id] = result
    return results


def print_validation_summary(results: dict[str, BriefValidationResult]) -> None:
    """Print a summary of validation results."""
    total      = len(results)
    valid      = sum(1 for r in results.values() if r.is_valid)
    with_issues = total - valid
    total_issues = sum(len(r.issues) for r in results.values())

    print(f"{'='*55}")
    print(f"  BRIEF VALIDATION SUMMARY")
    print(f"{'='*55}")
    print(f"  Total briefs:    {total}")
    print(f"  Valid:           {valid} ✅")
    print(f"  With issues:     {with_issues} ⚠️")
    print(f"  Total bad IDs:   {total_issues}")
    print(f"{'='*55}")
    print()
    for result in results.values():
        print(result.report())


def get_corrected_briefs(
    results: dict[str, BriefValidationResult],
) -> dict[str, ContentBrief]:
    """Return corrected briefs where available, original otherwise."""
    corrected: dict[str, ContentBrief] = {}
    for page_id, result in results.items():
        if result.corrected_brief is not None:
            corrected[page_id] = result.corrected_brief
        # If no issues, we need the original brief — caller must merge
    return corrected
