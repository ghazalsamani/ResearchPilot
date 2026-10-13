"""Serper Google Search API client with bounded retries."""
from __future__ import annotations

import time
from urllib.parse import urlparse

import httpx

from research_agent.schemas import SearchResult

SERPER_ENDPOINT = "https://google.serper.dev/search"


class SearchError(RuntimeError):
    """Raised when Serper cannot return a usable search response."""


class SerperClient:
    def __init__(
        self,
        api_key: str,
        timeout_seconds: float = 20.0,
        max_retries: int = 2,
        client: httpx.Client | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("Serper API key is required.")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(0, max_retries)
        self._client = client or httpx.Client(timeout=timeout_seconds)
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def search(self, query: str, num: int = 10) -> list[SearchResult]:
        query = query.strip()
        if not query:
            raise ValueError("Search query cannot be empty.")
        num = max(1, min(int(num), 20))
        payload = {"q": query, "num": num}
        headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}
        response: httpx.Response | None = None
        last_error: Exception | None = None
        succeeded = False

        for attempt in range(self.max_retries + 1):
            try:
                candidate = self._client.post(
                    SERPER_ENDPOINT,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout_seconds,
                )
                if candidate.status_code in {429, 500, 502, 503, 504}:
                    last_error = httpx.HTTPStatusError(
                        f"Retryable Serper response: {candidate.status_code}",
                        request=candidate.request,
                        response=candidate,
                    )
                    response = candidate
                    if attempt < self.max_retries:
                        time.sleep(0.25 * (attempt + 1))
                        continue
                    break
                candidate.raise_for_status()
                response = candidate
                succeeded = True
                break
            except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError) as exc:
                last_error = exc
                response = getattr(exc, "response", response)
                retryable = not isinstance(exc, httpx.HTTPStatusError) or (
                    exc.response.status_code in {429, 500, 502, 503, 504}
                )
                if attempt >= self.max_retries or not retryable:
                    break
                time.sleep(0.25 * (attempt + 1))

        if not succeeded or response is None:
            status = getattr(getattr(last_error, "response", None), "status_code", None)
            if status in {401, 403}:
                message = "Serper rejected the API key. Check SERPER_API_KEY."
            elif status == 429:
                message = "Serper rate limit reached. Try again later."
            else:
                message = f"Serper request failed: {last_error or 'no response received'}"
            raise SearchError(message) from last_error

        try:
            data = response.json()
        except ValueError as exc:
            raise SearchError("Serper returned malformed JSON.") from exc
        organic = data.get("organic", [])
        if not isinstance(organic, list):
            raise SearchError("Serper response field 'organic' was not a list.")

        results: list[SearchResult] = []
        for item in organic:
            if not isinstance(item, dict) or not item.get("link"):
                continue
            url = str(item["link"]).strip()
            parsed = urlparse(url)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                continue
            try:
                results.append(SearchResult(
                    title=str(item.get("title") or url),
                    url=url,
                    snippet=str(item.get("snippet") or ""),
                    domain=parsed.hostname.lower(),
                    published_date=str(item.get("date")) if item.get("date") else None,
                    query=query,
                ))
            except ValueError:
                continue
        return results
