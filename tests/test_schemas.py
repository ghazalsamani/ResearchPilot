import pytest
from pydantic import ValidationError

from research_agent.schemas import ResearchRequest, SearchResult


def test_request_strips_query_and_deduplicates_domains():
    request = ResearchRequest(query="  How do AI agents work?  ", include_domains=["https://example.com/", "example.com"])
    assert request.query == "How do AI agents work?"
    assert request.include_domains == ["example.com"]


def test_request_rejects_empty_query_and_excessive_sources():
    with pytest.raises(ValidationError):
        ResearchRequest(query="   ")
    with pytest.raises(ValidationError):
        ResearchRequest(query="A meaningful query", max_sources=100)


def test_search_result_requires_absolute_http_url():
    with pytest.raises(ValidationError):
        SearchResult(title="bad", url="javascript:alert(1)", domain="example.com", query="test")
