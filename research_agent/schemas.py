"""Validated data contracts used throughout ResearchPilot."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator, model_validator

ResearchMode = Literal["academic", "competitive", "content", "general"]
ResearchDepth = Literal["brief", "standard", "deep"]


class ResearchRequest(BaseModel):
    query: str = Field(min_length=5, max_length=1000)
    mode: ResearchMode = "general"
    language: str = "English"
    audience: str = "General professional audience"
    depth: ResearchDepth = "standard"
    max_sources: int = Field(default=8, ge=1, le=20)
    recency_days: int | None = Field(default=None, ge=1, le=3650)
    include_domains: list[str] = Field(default_factory=list, max_length=20)
    exclude_domains: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("query", "language", "audience")
    @classmethod
    def strip_non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("This field cannot be empty or whitespace-only.")
        return value

    @field_validator("include_domains", "exclude_domains")
    @classmethod
    def normalize_domains(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        for raw in values:
            domain = raw.strip().lower()
            domain = domain.removeprefix("https://").removeprefix("http://").strip("/")
            if not domain or "/" in domain or " " in domain:
                raise ValueError(f"Invalid domain value: {raw!r}")
            if domain not in normalized:
                normalized.append(domain)
        return normalized


class ResearchPlan(BaseModel):
    objective: str
    subquestions: list[str] = Field(min_length=1, max_length=10)
    search_queries: list[str] = Field(min_length=1, max_length=10)
    report_outline: list[str] = Field(min_length=3, max_length=12)


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str = ""
    domain: str
    published_date: str | None = None
    query: str

    @field_validator("url")
    @classmethod
    def valid_http_url(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Search result URL must be an absolute HTTP(S) URL.")
        return value


class SourceDocument(BaseModel):
    source_id: str
    title: str
    url: str
    domain: str
    snippet: str = ""
    content: str = ""
    published_date: str | None = None
    retrieval_status: Literal["full_text", "snippet_only", "failed"]
    retrieval_note: str | None = None
    retrieved_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class EvidenceClaim(BaseModel):
    claim: str = Field(min_length=1, max_length=2000)
    supporting_excerpt: str | None = Field(default=None, max_length=1200)
    caveat: str | None = Field(default=None, max_length=1000)


class EvidenceExtraction(BaseModel):
    source_summary: str
    claims: list[EvidenceClaim] = Field(default_factory=list, max_length=15)
    source_limitations: list[str] = Field(default_factory=list, max_length=10)


class EvidenceRecord(BaseModel):
    source_id: str
    source_title: str
    source_url: str
    evidence_type: Literal["full_text", "snippet_only"]
    source_summary: str
    claims: list[EvidenceClaim] = Field(default_factory=list)
    source_limitations: list[str] = Field(default_factory=list)


class SufficiencyAssessment(BaseModel):
    sufficient: bool
    rationale: str
    missing_questions: list[str] = Field(default_factory=list, max_length=10)
    additional_queries: list[str] = Field(default_factory=list, max_length=5)


class CitationAudit(BaseModel):
    valid_ids: list[str] = Field(default_factory=list)
    unknown_ids: list[str] = Field(default_factory=list)
    cited_ids: list[str] = Field(default_factory=list)
