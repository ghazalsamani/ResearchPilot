import httpx
import pytest

from research_agent.search import SearchError, SerperClient


class DummyResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {"organic": []}
        self.headers = {}

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("POST", "https://google.serper.dev/search")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("bad status", request=request, response=response)

    def json(self):
        return self._payload


class DummyClient:
    def __init__(self, response):
        self.response = response
        self.calls = []
    def post(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.response
    def close(self):
        pass


def test_serper_parses_organic_results_and_sends_expected_payload():
    client = DummyClient(DummyResponse(payload={"organic": [{
        "title": "Example article", "link": "https://www.example.com/story?utm_source=x",
        "snippet": "A useful snippet", "date": "2026-09-01"
    }]}))
    search = SerperClient("test-key", client=client)
    results = search.search("example research", num=7)
    assert len(results) == 1
    assert results[0].title == "Example article"
    assert results[0].domain == "www.example.com"
    _, kwargs = client.calls[0]
    assert kwargs["json"] == {"q": "example research", "num": 7}
    assert kwargs["headers"]["X-API-KEY"] == "test-key"


def test_serper_rejects_empty_query():
    search = SerperClient("test-key", client=DummyClient(DummyResponse()))
    with pytest.raises(ValueError):
        search.search(" ")


def test_serper_handles_invalid_credentials():
    search = SerperClient("test-key", max_retries=0, client=DummyClient(DummyResponse(status_code=401)))
    with pytest.raises(SearchError, match="rejected the API key"):
        search.search("some query")
