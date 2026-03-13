from __future__ import annotations

import pytest
import requests

from miniprophet.exceptions import SearchAuthError, SearchNetworkError, SearchRateLimitError
from miniprophet.tools.search.brave import BraveSearchBackend


@pytest.fixture
def backend(monkeypatch: pytest.MonkeyPatch) -> BraveSearchBackend:
    monkeypatch.setenv("BRAVE_API_KEY", "test-key")
    return BraveSearchBackend(max_retries=1)


def _mock_response(status_code: int = 200, json_data: dict | None = None, text: str = ""):
    resp = requests.models.Response()
    resp.status_code = status_code
    resp._content = b""
    if json_data is not None:
        import json

        resp._content = json.dumps(json_data).encode()
    else:
        resp._content = text.encode()
    return resp


class TestBraveGetLinks:
    def test_missing_api_key_raises_auth_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("BRAVE_API_KEY", raising=False)
        b = BraveSearchBackend()
        with pytest.raises(SearchAuthError, match="BRAVE_API_KEY"):
            b._get_links("test", 5)

    def test_successful_search_returns_links(
        self, backend: BraveSearchBackend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        api_response = {
            "web": {
                "results": [
                    {
                        "url": "https://example.com/1",
                        "title": "Result 1",
                        "description": "First result",
                        "age": "2d",
                    },
                    {
                        "url": "https://example.com/2",
                        "title": "Result 2",
                        "description": "Second result",
                    },
                ]
            }
        }
        monkeypatch.setattr(
            "miniprophet.tools.search.brave.requests.get",
            lambda *a, **kw: _mock_response(200, api_response),
        )

        links = backend._get_links("test query", 5)
        assert len(links) == 2
        assert links[0]["url"] == "https://example.com/1"
        assert links[0]["date"] == "2d"
        assert links[1]["date"] is None

    @pytest.mark.parametrize(
        "status_code,exc_type,match",
        [
            (401, SearchAuthError, "authentication failed"),
            (429, SearchRateLimitError, "rate limit"),
            (500, SearchNetworkError, "HTTP 500"),
        ],
    )
    def test_http_error_raises_expected(
        self, backend: BraveSearchBackend, monkeypatch: pytest.MonkeyPatch,
        status_code: int, exc_type: type, match: str,
    ) -> None:
        resp = _mock_response(status_code, text="Internal Server Error")
        monkeypatch.setattr("miniprophet.tools.search.brave.requests.get", lambda *a, **kw: resp)
        with pytest.raises(exc_type, match=match):
            backend._get_links("test", 5)

    def test_connection_error_raises_network_error(
        self, backend: BraveSearchBackend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "miniprophet.tools.search.brave.requests.get",
            lambda *a, **kw: (_ for _ in ()).throw(requests.exceptions.ConnectionError("connection refused")),
        )
        with pytest.raises(SearchNetworkError, match="request failed"):
            backend._get_links("test", 5)

    def test_freshness_param_forwarded(
        self, backend: BraveSearchBackend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict = {}

        def mock_get(url, *, headers, params, timeout):
            captured.update(params)
            return _mock_response(200, {"web": {"results": []}})

        monkeypatch.setattr("miniprophet.tools.search.brave.requests.get", mock_get)
        backend._get_links("test", 5, freshness="pw")
        assert captured["freshness"] == "pw"


class TestBraveSearch:
    def test_search_extracts_articles(
        self, backend: BraveSearchBackend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            backend,
            "_get_links",
            lambda q, limit, **kw: [
                {"url": "https://a.com", "title": "A", "snippet": "s", "date": None},
                {"url": "https://b.com", "title": "B", "snippet": "s2", "date": "1d"},
            ],
        )
        monkeypatch.setattr(
            backend, "_fetch_article_text", lambda url: "extracted" if "a.com" in url else None
        )

        result = backend.search("query")
        assert len(result.sources) == 1
        assert result.sources[0].url == "https://a.com"
        assert result.cost == 0.0

    def test_search_strips_date_filter_kwargs(
        self, backend: BraveSearchBackend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(backend, "_get_links", lambda q, limit, **kw: [])
        monkeypatch.setattr(backend, "_fetch_article_text", lambda url: None)

        result = backend.search(
            "query", search_date_before="01/01/2026", search_date_after="12/01/2025"
        )
        assert result.sources == []


class TestBraveSerialize:
    def test_serialize_returns_config(self, backend: BraveSearchBackend) -> None:
        s = backend.serialize()
        assert s["info"]["config"]["search"]["search_class"] == "brave"
        assert s["info"]["config"]["search"]["max_retries"] == 1
