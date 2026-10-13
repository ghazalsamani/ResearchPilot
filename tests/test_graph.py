from pathlib import Path
from types import SimpleNamespace

import pytest
pytest.importorskip("langgraph")

from research_agent.graph import ResearchRunner, normalize_url
from research_agent.schemas import EvidenceClaim, EvidenceExtraction, ResearchPlan, SearchResult, SourceDocument, SufficiencyAssessment


class FakeStructured:
    def __init__(self, schema):
        self.schema = schema
    def invoke(self, prompt):
        if self.schema is ResearchPlan:
            return ResearchPlan(objective="Compare the approaches", subquestions=["What are the differences?"],
                search_queries=["LangGraph and CrewAI production agents", "LangGraph official docs"],
                report_outline=["Executive summary", "Comparison", "Conclusion"])
        if self.schema is EvidenceExtraction:
            return EvidenceExtraction(source_summary="The source discusses graph workflows.", claims=[
                EvidenceClaim(claim="The source describes graph-based workflows.", supporting_excerpt="graph-based workflows")])
        if self.schema is SufficiencyAssessment:
            return SufficiencyAssessment(sufficient=True, rationale="The fake evidence covers the test question.")
        raise AssertionError(f"Unexpected structured schema: {self.schema}")


class FakeLLM:
    def with_structured_output(self, schema, **kwargs):
        return FakeStructured(schema)
    def invoke(self, prompt):
        return SimpleNamespace(content="# Research Findings\n\nThe collected source describes graph-based workflows [S1].")


class FakeSearch:
    def search(self, query, num=10):
        return [SearchResult(title="Example docs", url="https://example.com/docs", snippet="Graph-based workflows",
            domain="example.com", query=query)]
    def close(self):
        pass


class FakeReader:
    def fetch(self, result, source_id):
        return SourceDocument(source_id=source_id, title=result.title, url=result.url, domain=result.domain,
            snippet=result.snippet, content="The documentation describes graph-based workflows.", retrieval_status="full_text")
    def close(self):
        pass


def test_normalize_url_removes_tracking_parameters():
    assert normalize_url("https://Example.com/page/?utm_source=test&x=1#section") == "https://example.com/page?x=1"


def test_graph_runs_end_to_end_with_fake_dependencies(tmp_path: Path):
    from research_agent.schemas import ResearchRequest
    runner = ResearchRunner(FakeLLM(), FakeSearch(), FakeReader(), tmp_path)
    result = runner.run(ResearchRequest(query="Compare LangGraph and CrewAI for production AI agents", max_sources=4))
    assert result["report_path"].exists()
    assert result["trace_path"].exists()
    markdown = result["report_path"].read_text(encoding="utf-8")
    assert "## References" in markdown
    assert "[S1]" in markdown
    assert result["source_count"] == 1
