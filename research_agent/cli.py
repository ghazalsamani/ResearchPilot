"""Command-line entry point for ResearchPilot."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from langchain_openai import ChatOpenAI

from research_agent.config import Settings
from research_agent.fetcher import PageReader
from research_agent.graph import ResearchRunner
from research_agent.schemas import ResearchRequest
from research_agent.search import SerperClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="researchpilot",
        description="ResearchPilot: autonomous web research with evidence-backed Markdown reports.",
    )
    parser.add_argument("--query", help="Research question or topic. Omit to be prompted interactively.")
    parser.add_argument("--mode", choices=["academic", "competitive", "content", "general"], default="general")
    parser.add_argument("--language", default="English", help="Report language (default: English).")
    parser.add_argument("--audience", default="General professional audience")
    parser.add_argument("--depth", choices=["brief", "standard", "deep"], default="standard")
    parser.add_argument("--max-sources", type=int, default=8, help="Maximum source pages/snippets to inspect (1-20).")
    parser.add_argument("--recency-days", type=int, default=None, help="Prefer recent information when relevant (1-3650).")
    parser.add_argument("--include-domain", action="append", default=[], help="Limit results to this domain; repeat as needed.")
    parser.add_argument("--exclude-domain", action="append", default=[], help="Exclude this domain; repeat as needed.")
    parser.add_argument("--output-dir", default=None, help="Override OUTPUT_DIR from .env (default: output).")
    parser.add_argument("--show-report", action="store_true", help="Print the full generated report after saving.")
    return parser


def _interactive_query() -> str:
    try:
        query = input("What would you like to research? ").strip()
    except (EOFError, KeyboardInterrupt) as exc:
        raise ValueError("No research query was provided.") from exc
    if not query:
        raise ValueError("Research query cannot be empty.")
    return query


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        query = args.query.strip() if args.query else _interactive_query()
        request = ResearchRequest(
            query=query, mode=args.mode, language=args.language, audience=args.audience,
            depth=args.depth, max_sources=args.max_sources, recency_days=args.recency_days,
            include_domains=args.include_domain, exclude_domains=args.exclude_domain,
        )
        settings = Settings.from_env(require_keys=True)
        logging.basicConfig(
            level=getattr(logging, settings.log_level, logging.INFO),
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
        output_dir = Path(args.output_dir) if args.output_dir else settings.output_dir
        llm = ChatOpenAI(model=settings.openai_model, api_key=settings.openai_api_key, temperature=0)
        search_client = SerperClient(settings.serper_api_key)
        page_reader = PageReader()
        try:
            print("\nResearchPilot is starting your research...\n")
            result = ResearchRunner(llm, search_client, page_reader, output_dir).run(request)
        finally:
            search_client.close()
            page_reader.close()
        print("Research completed.\n")
        print(f"Sources collected: {result['source_count']}")
        print(f"Evidence records: {result['evidence_count']}")
        print(f"Markdown report: {result['report_path']}")
        print(f"JSON research trace: {result['trace_path']}")
        if result["errors"]:
            print(f"Warnings/errors recorded in trace: {len(result['errors'])}")
        audit = result.get("citation_audit", {})
        if audit.get("unknown_source_ids_removed"):
            print("Warning: unknown citation markers were removed; review the report.")
        if args.show_report:
            print("\n" + result["report_markdown"])
        return 0
    except Exception as exc:
        # Keep the CLI friendly without exposing environment variable values or stack traces by default.
        print(f"ResearchPilot failed: {exc}", file=sys.stderr)
        logging.getLogger(__name__).debug("CLI exception", exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
