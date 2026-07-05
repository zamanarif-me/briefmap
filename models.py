"""
Data model for BriefMap.

Every stage of the pipeline produces or consumes these objects.
The JSON serialization is what gets passed between stages and ultimately
rendered into the Markdown report.

Designed so v2 features (content audit, keyword volumes, briefs) are
additive — optional fields, new models, no rewrites needed.
"""

from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ---------- Enums ----------

class BusinessFocus(str, Enum):
    CUSTOM_DEV = "custom_wordpress_development"
    SECURITY = "wordpress_security_services"
    MAINTENANCE = "website_maintenance"
    MANAGED = "managed_wordpress"
    AGENCY = "agency_for_businesses"
    FREELANCER = "personal_freelancer_brand"
    ENTERPRISE = "enterprise_wordpress"
    OTHER = "other"


class SiteStage(str, Enum):
    BRAND_NEW = "brand_new"
    HAS_BLOGS = "existing_with_blogs"
    ESTABLISHED = "established_with_traffic"


class GeoScope(str, Enum):
    GLOBAL = "global"
    COUNTRY = "country_specific"
    LOCAL = "city_local"


class ContentMix(str, Enum):
    SERVICE_HEAVY = "mostly_service_pages"
    BLOG_HEAVY = "mostly_informational_blogs"
    BALANCED = "balanced_authority_site"


class Intent(str, Enum):
    """Search intent classification."""
    COMMERCIAL = "commercial"
    INFORMATIONAL = "informational"
    NAVIGATIONAL = "navigational"
    TRANSACTIONAL = "transactional"


class FunnelStage(str, Enum):
    """Extended funnel stages — Koray lifecycle model."""
    TOFU = "TOFU"                          # awareness — broad, informational
    MOFU = "MOFU"                          # evaluation — comparison, how-to
    BOFU = "BOFU"                          # decision — commercial, service pages
    IMPLEMENTATION = "IMPLEMENTATION"      # how to actually do it
    TROUBLESHOOTING = "TROUBLESHOOTING"    # fix problems
    SCALING = "SCALING"                    # grow beyond initial setup
    OPTIMIZATION = "OPTIMIZATION"          # refine after launch
    RECOVERY = "RECOVERY"                  # fix after failure/hack/migration


class SupplementaryAngle(str, Enum):
    """Content angle — drives information gain and retrieval differentiation."""
    CONTRADICTION = "contradiction"        # myth-busting, why X is wrong
    INFORMATION_GAIN = "information_gain"  # rare attributes, hidden mechanisms
    PERSPECTIVE = "perspective"            # different stakeholder viewpoint
    LIFECYCLE = "lifecycle"               # specific stage/state of maturity


class QueryType(str, Enum):
    REPRESENTATIVE = "representative"  # broad intent, e.g. "wordpress development"
    REPRESENTED = "represented"        # specific path, e.g. "wordpress development for startups"


class LinkRelationship(str, Enum):
    PILLAR_TO_CLUSTER = "pillar_to_cluster"
    CLUSTER_TO_PILLAR = "cluster_to_pillar"
    CLUSTER_TO_SUPPLEMENTARY = "cluster_to_supplementary"
    SUPPLEMENTARY_TO_CLUSTER = "supplementary_to_cluster"
    ENTITY_BRIDGE = "entity_bridge"          # links between pages sharing a key entity
    CONTEXTUAL = "contextual"                # general topical relevance
    HOMEPAGE_TO_PILLAR = "homepage_to_pillar"
    PILLAR_TO_GEO = "pillar_to_geo"          # pillar links its regional variants
    GEO_TO_PILLAR = "geo_to_pillar"          # geo page links back to parent pillar


# ---------- Input ----------

class GeoTargeting(BaseModel):
    scope: GeoScope
    countries: list[str] = Field(default_factory=list)
    cities: list[str] = Field(default_factory=list)


class IntakeAnswers(BaseModel):
    """The 8 questions from the spec, structured."""
    business_focus: BusinessFocus
    business_focus_detail: Optional[str] = None  # if OTHER
    target_audience: list[str]
    revenue_services: list[str]
    geo: GeoTargeting
    site_stage: SiteStage
    positioning: list[str]            # e.g. ["seo_focused", "affordable", "enterprise_grade"]
    focus_areas: list[str]            # 2-4 areas to dominate first
    content_mix: ContentMix


class SeedInput(BaseModel):
    """Top-level input to the engine."""
    seed_keyword: str = Field(..., description="The user's raw seed, e.g. 'WordPress development and security service'")
    intake: IntakeAnswers
    language: str = "en-US"


# ---------- Stage 2 output ----------

class CentralEntity(BaseModel):
    """Output of the central entity stage. Grounds everything downstream."""
    primary: str = Field(..., description="The single most important entity, e.g. 'WordPress Website Development Services'")
    supporting: str = Field(..., description="The secondary authority entity that creates the semantic bridge")
    source_context: str = Field(..., description="One-sentence positioning statement that should appear semantically across the site")
    key_entities: list[str] = Field(..., description="5-10 related entities that anchor the topical authority")
    reasoning: str = Field(..., description="Why these entities were chosen, in 2-3 sentences")


# ---------- Stage 3-6 outputs ----------

class SupplementaryNode(BaseModel):
    """Tier 3 — supporting topics that reinforce topical authority but aren't money pages."""
    id: str
    title: str
    intent: Intent
    funnel_stage: FunnelStage
    parent_cluster_id: str
    angle: Optional[str] = Field(default=None, description="contradiction | information_gain | perspective | lifecycle")
    rationale: Optional[str] = None


class Query(BaseModel):
    text: str
    type: QueryType
    intent: Intent
    parent_cluster_id: str


class Cluster(BaseModel):
    """Tier 2 — major subtopics under a pillar."""
    id: str
    title: str
    parent_pillar_id: str
    intent: Intent
    funnel_stage: FunnelStage
    supplementary_nodes: list[SupplementaryNode] = Field(default_factory=list)
    represented_queries: list[Query] = Field(default_factory=list)
    related_entities: list[str] = Field(default_factory=list)
    validation_signal: Optional[str] = Field(default=None, description="strong | medium | weak — set by stage 4")
    validation_reasoning: Optional[str] = None


class Pillar(BaseModel):
    """Tier 1 — money + authority nodes. The pages everything else points to."""
    id: str
    title: str
    intent: Intent
    funnel_stage: FunnelStage
    priority: int = Field(..., ge=1, le=3, description="1 = highest priority, publish first")
    clusters: list[Cluster] = Field(default_factory=list)
    representative_queries: list[Query] = Field(default_factory=list)
    related_entities: list[str] = Field(default_factory=list)
    commercial_value: str = Field(..., description="Why this is a money page — one sentence")
    validation_signal: Optional[str] = Field(default=None, description="strong | medium | weak — set by stage 4")
    validation_reasoning: Optional[str] = None


class GeoPage(BaseModel):
    """Regional service page derived from a pillar + a geography."""
    id: str
    title: str                # e.g. "WordPress Developer USA"
    parent_pillar_id: str
    geography: str            # e.g. "USA", "Europe", "London"
    intent: Intent = Intent.COMMERCIAL


class TopicalMap(BaseModel):
    """The complete topical structure. Stage 6 output."""
    central_entity: CentralEntity
    pillars: list[Pillar]
    geo_pages: list[GeoPage] = Field(default_factory=list)


# ---------- Stage 7 output ----------

class InternalLink(BaseModel):
    from_page_id: str
    to_page_id: str
    anchor_text: str
    relationship: LinkRelationship
    reasoning: str = Field(..., description="One sentence on why this link exists.")
    relationship_strength: Optional[float] = Field(
        default=None,
        description="Semantic edge weight 0.0-1.0. 0.9+ = direct entity overlap, 0.7-0.89 = strong contextual, 0.5-0.69 = moderate, 0.3-0.49 = weak but valid. Only set for entity_bridge links."
    )


class LinkingPlan(BaseModel):
    links: list[InternalLink]
    homepage_links: list[str] = Field(..., description="page_ids the homepage should link to directly")


# ---------- Final output ----------

class EngineOutput(BaseModel):
    """What the engine returns at the end of a run. Serialized to JSON and rendered to Markdown."""
    input: SeedInput
    topical_map: TopicalMap
    linking_plan: LinkingPlan
    version: str = "1.0.0"
