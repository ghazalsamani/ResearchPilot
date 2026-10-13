"""Markdown report assembly and citation auditing."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from research_agent.schemas import CitationAudit, SourceDocument

CITATION_PATTERN = re.compile(r"\[(S\d+)\]")


def audit_citations(text: str, valid_source_ids: Iterable[str]) -> CitationAudit:
    valid = set(valid_source_ids)
    encountered = CITATION_PATTERN.findall(text)
    used = list(dict.fromkeys(encountered))
    unknown = list(dict.fromkeys(source_id for source_id in encountered if source_id not in valid))
    return CitationAudit(
        valid_ids=sorted(valid),
        unknown_ids=unknown,
        cited_ids=[source_id for source_id in used if source_id in valid],
    )


def remove_unknown_citations(text: str, allowed: set[str]) -> tuple[str, list[str]]:
    removed: list[str] = []
    def replace(match: re.Match[str]) -> str:
        source_id = match.group(1)
        if source_id not in allowed:
            removed.append(source_id)
            return "[citation removed: unknown source ID]"
        return match.group(0)
    return CITATION_PATTERN.sub(replace, text), list(dict.fromkeys(removed))


def build_references(sources: list[SourceDocument], cited_ids: list[str]) -> str:
    by_id = {source.source_id: source for source in sources}
    selected = [by_id[source_id] for source_id in cited_ids if source_id in by_id]
    if not selected and sources:
        selected = sources
    if not selected:
        return "## References\n\nNo sources were successfully collected.\n"
    lines = ["## References", ""]
    for source in selected:
        date = f" Publication date: {source.published_date}." if source.published_date else ""
        status = "Full page text retrieved" if source.retrieval_status == "full_text" else "Search snippet only; page text was not verified"
        lines.append(f"- **[{source.source_id}] {source.title}** — {source.domain}. {status}.{date} URL: {source.url}")
    return "\n".join(lines) + "\n"


def assemble_report(
    report_body: str,
    sources: list[SourceDocument],
    title_fallback: str,
) -> tuple[str, dict[str, list[str]]]:
    body = report_body.strip()
    if not body.startswith("#"):
        body = f"# {title_fallback}\n\n{body}"
    usable_sources = [source for source in sources if source.retrieval_status != "failed" and (source.content.strip() or source.snippet.strip())]
    allowed = {source.source_id for source in usable_sources}
    body, removed = remove_unknown_citations(body, allowed)
    audit = audit_citations(body, allowed)
    if removed:
        body += "\n\n> **Citation audit warning:** One or more citations referenced unknown source IDs. Those markers were removed; review the affected statements before relying on this report.\n"
    if not audit.cited_ids and sources:
        body += "\n\n> **Citation audit warning:** The draft did not contain valid source citations. The collected sources are listed below, but individual claims need manual citation review.\n"
    references = build_references(usable_sources, audit.cited_ids)
    complete = body.rstrip() + "\n\n" + references
    return complete, {"valid_source_ids": audit.valid_ids, "cited_source_ids": audit.cited_ids, "unknown_source_ids_removed": removed}


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
