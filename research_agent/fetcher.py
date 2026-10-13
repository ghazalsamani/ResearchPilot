"""Safe, bounded retrieval and extraction of public web pages."""
from __future__ import annotations

import ipaddress
import socket
from contextlib import AbstractContextManager
from urllib.parse import urljoin, urlparse

import httpx
try:
    import trafilatura
except ImportError:  # optional at runtime: BeautifulSoup provides the fallback extractor
    trafilatura = None
from bs4 import BeautifulSoup

from research_agent.schemas import SearchResult, SourceDocument

REDIRECT_CODES = {301, 302, 303, 307, 308}


class FetchError(RuntimeError):
    """A web page could not be safely retrieved or parsed."""


def validate_public_url(url: str) -> None:
    """Reject non-HTTP URLs and hosts resolving to non-public IP addresses."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise FetchError("Only absolute HTTP(S) URLs are allowed.")
    if parsed.username or parsed.password:
        raise FetchError("URLs containing embedded credentials are not allowed.")
    host = parsed.hostname.rstrip(".").lower()
    if host in {"localhost", "localhost.localdomain"} or host.endswith(".localhost") or host.endswith(".local"):
        raise FetchError("Local or internal hostnames are not allowed.")
    try:
        addresses = [ipaddress.ip_address(host)]
    except ValueError:
        try:
            addresses = [ipaddress.ip_address(info[4][0]) for info in socket.getaddrinfo(host, None)]
        except (OSError, ValueError) as exc:
            raise FetchError(f"Could not safely resolve hostname: {host}") from exc
    if not addresses:
        raise FetchError(f"No IP addresses found for hostname: {host}")
    if any(not address.is_global for address in addresses):
        raise FetchError(f"Non-public IP destination is blocked: {host}")


class PageReader:
    def __init__(
        self,
        timeout_seconds: float = 12.0,
        max_bytes: int = 1_500_000,
        max_redirects: int = 4,
        client: httpx.Client | None = None,
        url_validator=validate_public_url,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_bytes = max_bytes
        self.max_redirects = max_redirects
        self._client = client or httpx.Client(timeout=timeout_seconds, follow_redirects=False)
        self._owns_client = client is None
        self._url_validator = url_validator

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    @staticmethod
    def _metadata(html: str, final_url: str) -> tuple[str | None, str | None]:
        soup = BeautifulSoup(html, "html.parser")
        title = soup.title.get_text(" ", strip=True) if soup.title else None
        canonical = soup.find("link", rel=lambda value: value and "canonical" in value)
        canonical_url = urljoin(final_url, canonical.get("href")) if canonical and canonical.get("href") else None
        return title, canonical_url

    @staticmethod
    def extract_content(html: str, url: str) -> tuple[str, str | None, str | None, str | None]:
        """Return (main text, title, canonical URL, publication date)."""
        title, canonical_url = PageReader._metadata(html, url)
        soup = BeautifulSoup(html, "html.parser")
        date = None
        date_meta = (
            soup.find("meta", attrs={"property": "article:published_time"})
            or soup.find("meta", attrs={"name": "date"})
            or soup.find("meta", attrs={"property": "og:published_time"})
        )
        if date_meta and date_meta.get("content"):
            date = str(date_meta.get("content"))
        extracted = trafilatura.extract(html, url=url, include_comments=False, include_tables=True) if trafilatura else None
        if not extracted:
            for element in soup(["script", "style", "noscript", "nav", "footer", "header", "aside", "form"]):
                element.decompose()
            main = soup.find("main") or soup.find("article") or soup.body or soup
            extracted = main.get_text("\n", strip=True)
        return (extracted or "").strip(), title, canonical_url, date

    def _read_html(self, url: str) -> tuple[bytes, str]:
        current = url
        for redirect_index in range(self.max_redirects + 1):
            self._url_validator(current)
            with self._client.stream(
                "GET",
                current,
                headers={"User-Agent": "ResearchPilot/0.1 (+CLI research tool)"},
                timeout=self.timeout_seconds,
                follow_redirects=False,
            ) as response:
                if response.status_code in REDIRECT_CODES:
                    if redirect_index >= self.max_redirects:
                        raise FetchError("Maximum redirect count exceeded.")
                    location = response.headers.get("location")
                    if not location:
                        raise FetchError("Redirect response did not include a Location header.")
                    current = urljoin(current, location)
                    continue
                response.raise_for_status()
                content_type = response.headers.get("content-type", "").lower()
                if content_type and not any(kind in content_type for kind in ("text/html", "application/xhtml+xml", "text/plain")):
                    raise FetchError(f"Unsupported content type: {content_type.split(';')[0]}")
                chunks: list[bytes] = []
                byte_count = 0
                for chunk in response.iter_bytes():
                    byte_count += len(chunk)
                    if byte_count > self.max_bytes:
                        raise FetchError(f"Page exceeded the {self.max_bytes}-byte content limit.")
                    chunks.append(chunk)
                return b"".join(chunks), str(response.url or current)
        raise FetchError("Redirect handling ended unexpectedly.")

    def fetch(self, result: SearchResult, source_id: str) -> SourceDocument:
        try:
            raw_bytes, final_url = self._read_html(result.url)
            html = raw_bytes.decode("utf-8", errors="replace")
            content, page_title, canonical, published_date = self.extract_content(html, final_url)
            if not content:
                return SourceDocument(
                    source_id=source_id, title=page_title or result.title, url=canonical or final_url,
                    domain=result.domain, snippet=result.snippet, content="", 
                    published_date=published_date or result.published_date,
                    retrieval_status="snippet_only" if result.snippet else "failed",
                    retrieval_note="The page loaded, but no readable main content could be extracted.",
                )
            selected_url = canonical or final_url
            return SourceDocument(
                source_id=source_id, title=page_title or result.title, url=selected_url,
                domain=result.domain, snippet=result.snippet, content=content[:100_000],
                published_date=published_date or result.published_date,
                retrieval_status="full_text",
            )
        except Exception as exc:
            return SourceDocument(
                source_id=source_id, title=result.title, url=result.url, domain=result.domain,
                snippet=result.snippet, content="", published_date=result.published_date,
                retrieval_status="snippet_only" if result.snippet else "failed",
                retrieval_note=f"Page retrieval failed: {type(exc).__name__}: {exc}",
            )
