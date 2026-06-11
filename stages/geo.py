"""
Geographic page derivation.

The intake says what geos the user targets. The pillar list already includes
1-2 geographic pillars (created in stage 3). This helper expands the geo
coverage by producing regional variants of the top commercial pillars.

This is deterministic — no LLM call needed.
"""

from models import Pillar, GeoPage, GeoTargeting, GeoScope, Intent


def derive_geo_pages(pillars: list[Pillar], geo: GeoTargeting) -> list[GeoPage]:
    """
    For each priority-1 commercial pillar that ISN'T already a geographic pillar,
    create one geo variant per country/city in the targeting.
    """
    if geo.scope == GeoScope.GLOBAL:
        return []

    geos: list[str] = []
    if geo.scope == GeoScope.COUNTRY:
        geos = geo.countries
    elif geo.scope == GeoScope.LOCAL:
        geos = geo.cities

    if not geos:
        return []

    # Find pillars that already mention one of the TARGET geos in their
    # title — skip those. (Previously this also matched any title containing
    # " in ", which false-positived on titles like "WordPress Maintenance
    # in 2026".)
    def is_geographic(pillar: Pillar) -> bool:
        t = pillar.title.lower()
        return any(g.lower() in t for g in geos)

    eligible = [
        p for p in pillars
        if p.priority == 1
        and p.intent == Intent.COMMERCIAL
        and not is_geographic(p)
    ]

    geo_pages: list[GeoPage] = []
    for pillar in eligible[:3]:  # cap to 3 pillars × N geos to avoid sprawl
        for region in geos:
            # Derive a sensible title
            title = f"{pillar.title} in {region}"
            geo_pages.append(GeoPage(
                id=f"geo_{pillar.id}_{region.lower().replace(' ', '_')}",
                title=title,
                parent_pillar_id=pillar.id,
                geography=region,
            ))

    return geo_pages
