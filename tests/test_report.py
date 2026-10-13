from research_agent.report import assemble_report, audit_citations
from research_agent.schemas import SourceDocument


def source(source_id="S1"):
    return SourceDocument(source_id=source_id, title="Example source", url="https://example.com/article",
        domain="example.com", snippet="A snippet", content="Some source text", retrieval_status="full_text")


def test_citation_audit_finds_unknown_citations():
    audit = audit_citations("Evidence [S1] and invented [S99].", ["S1"])
    assert audit.cited_ids == ["S1"]
    assert audit.unknown_ids == ["S99"]


def test_assembler_removes_unknown_citations_and_builds_references():
    report, audit = assemble_report("# Findings\n\nSupported statement [S1]. Unsupported ref [S99].", [source()], "Fallback title")
    assert "[citation removed: unknown source ID]" in report
    assert "## References" in report
    assert "https://example.com/article" in report
    assert audit["cited_source_ids"] == ["S1"]
    assert audit["unknown_source_ids_removed"] == ["S99"]


def test_assembler_warns_when_no_source_ids_are_cited():
    report, _ = assemble_report("A report without citations.", [source()], "Fallback title")
    assert "did not contain valid source citations" in report
