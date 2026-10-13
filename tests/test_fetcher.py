import pytest

from research_agent.fetcher import FetchError, PageReader, validate_public_url


def test_extracts_main_text_and_metadata():
    html = '''<!doctype html><html><head><title>Test article</title>
    <link rel="canonical" href="https://example.com/canonical">
    <meta property="article:published_time" content="2026-08-15"></head>
    <body><nav>Menu should disappear</nav><main><article><h1>Research results</h1>
    <p>This is the main article text with enough useful information for extraction.</p>
    <p>The second paragraph adds context to this research topic.</p></article></main></body></html>'''
    text, title, canonical, date = PageReader.extract_content(html, "https://example.com/page")
    assert "main article text" in text
    assert title == "Test article"
    assert canonical == "https://example.com/canonical"
    assert date == "2026-08-15"


def test_blocks_localhost_and_private_ip_destinations():
    with pytest.raises(FetchError):
        validate_public_url("http://127.0.0.1:8000/admin")
    with pytest.raises(FetchError):
        validate_public_url("http://localhost/private")


def test_rejects_non_http_scheme():
    with pytest.raises(FetchError):
        validate_public_url("file:///etc/passwd")
