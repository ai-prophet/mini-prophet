from __future__ import annotations

from typing import Any

import pytest

from miniprophet.exceptions import SearchAuthError, SearchNetworkError, SearchRateLimitError
from miniprophet.tools.search.tavily import TavilySearchBackend

# --- Fake client ---


class _FakeTavilyClient:
    def __init__(self, **kwargs: Any) -> None:
        self._response: dict = {}
        self._raise: Exception | None = None
        self.last_payload: dict = {}

    def search(self, **kwargs: Any) -> dict:
        self.last_payload = kwargs
        if self._raise is not None:
            raise self._raise
        return self._response


def _make_result(
    url: str = "https://example.com",
    title: str = "Title",
    content: str = "body text",
    score: float = 0.9,
    published_date: str | None = "2026-01-15",
) -> dict:
    return {
        "url": url,
        "title": title,
        "content": content,
        "score": score,
        "published_date": published_date,
    }


@pytest.fixture
def backend_and_client(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[TavilySearchBackend, _FakeTavilyClient]:
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    fake_client = _FakeTavilyClient()
    monkeypatch.setattr(
        "miniprophet.tools.search.tavily.TavilyClient",
        lambda api_key: fake_client,
    )
    backend = TavilySearchBackend()
    return backend, fake_client


# --- Init tests ---


class TestTavilyInit:
    def test_missing_api_key_raises_auth_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        with pytest.raises(SearchAuthError, match="TAVILY_API_KEY"):
            TavilySearchBackend()

    def test_custom_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TAVILY_API_KEY", "key")
        monkeypatch.setattr(
            "miniprophet.tools.search.tavily.TavilyClient",
            lambda api_key: _FakeTavilyClient(),
        )
        backend = TavilySearchBackend(search_depth="advanced", topic="news", max_characters=5000)
        assert backend._search_depth == "advanced"
        assert backend._topic == "news"
        assert backend._max_characters == 5000


# --- Search tests ---


class TestTavilySearch:
    def test_successful_search_returns_sources(
        self, backend_and_client: tuple[TavilySearchBackend, _FakeTavilyClient]
    ) -> None:
        backend, client = backend_and_client
        client._response = {
            "results": [_make_result(), _make_result(url="https://b.com", title="B")],
            "usage": {"credits": 1},
        }

        result = backend.search("test query", limit=5)
        assert len(result.sources) == 2
        assert result.sources[0].url == "https://example.com"
        assert result.sources[0].title == "Title"
        assert result.sources[0].snippet == "body text"
        assert result.sources[0].date == "2026-01-15"
        assert result.cost == pytest.approx(1.0)

    def test_search_skips_empty_url_results(
        self, backend_and_client: tuple[TavilySearchBackend, _FakeTavilyClient]
    ) -> None:
        backend, client = backend_and_client
        client._response = {
            "results": [_make_result(url=""), _make_result(url="https://valid.com")],
        }

        result = backend.search("query")
        assert len(result.sources) == 1
        assert result.sources[0].url == "https://valid.com"

    def test_date_filters_converted_and_forwarded(
        self, backend_and_client: tuple[TavilySearchBackend, _FakeTavilyClient]
    ) -> None:
        backend, client = backend_and_client
        client._response = {"results": []}

        backend.search("test", search_date_after="01/01/2026", search_date_before="03/01/2026")
        assert client.last_payload["start_date"] == "2026-01-01"
        assert client.last_payload["end_date"] == "2026-03-01"

    def test_time_range_forwarded(
        self, backend_and_client: tuple[TavilySearchBackend, _FakeTavilyClient]
    ) -> None:
        backend, client = backend_and_client
        client._response = {"results": []}

        backend.search("test", time_range="week")
        assert client.last_payload["time_range"] == "week"

    def test_invalid_date_raises_network_error(
        self, backend_and_client: tuple[TavilySearchBackend, _FakeTavilyClient]
    ) -> None:
        backend, client = backend_and_client
        with pytest.raises(SearchNetworkError, match="Expected MM/DD/YYYY"):
            backend.search("test", search_date_after="2026-01-01")

    def test_invalid_api_key_raises_auth_error(
        self, backend_and_client: tuple[TavilySearchBackend, _FakeTavilyClient]
    ) -> None:
        from tavily.errors import InvalidAPIKeyError

        backend, client = backend_and_client
        client._raise = InvalidAPIKeyError("invalid key")

        with pytest.raises(SearchAuthError, match="authentication failed"):
            backend.search("test")

    def test_missing_key_exception_raises_auth_error(
        self, backend_and_client: tuple[TavilySearchBackend, _FakeTavilyClient]
    ) -> None:
        from tavily.errors import MissingAPIKeyError

        backend, client = backend_and_client
        client._raise = MissingAPIKeyError()

        with pytest.raises(SearchAuthError, match="authentication failed"):
            backend.search("test")

    def test_usage_limit_raises_rate_limit_error(
        self, backend_and_client: tuple[TavilySearchBackend, _FakeTavilyClient]
    ) -> None:
        from tavily.errors import UsageLimitExceededError

        backend, client = backend_and_client
        client._raise = UsageLimitExceededError("limit exceeded")

        with pytest.raises(SearchRateLimitError, match="usage limit"):
            backend.search("test")

    @pytest.mark.parametrize(
        "status_code,exc_type,match",
        [
            (401, SearchAuthError, "authentication failed"),
            (429, SearchRateLimitError, "rate limit"),
        ],
    )
    def test_http_error_raises_expected(
        self, backend_and_client: tuple[TavilySearchBackend, _FakeTavilyClient],
        status_code: int, exc_type: type, match: str,
    ) -> None:
        backend, client = backend_and_client
        exc = Exception("error")
        exc.status_code = status_code  # type: ignore[attr-defined]
        client._raise = exc
        with pytest.raises(exc_type, match=match):
            backend.search("test")

    def test_generic_error_raises_network_error(
        self, backend_and_client: tuple[TavilySearchBackend, _FakeTavilyClient]
    ) -> None:
        backend, client = backend_and_client
        client._raise = RuntimeError("connection failed")
        with pytest.raises(SearchNetworkError, match="request failed"):
            backend.search("test")

    def test_content_truncated_to_max_characters(
        self, backend_and_client: tuple[TavilySearchBackend, _FakeTavilyClient]
    ) -> None:
        backend, client = backend_and_client
        backend._max_characters = 10
        client._response = {
            "results": [_make_result(content="a" * 100)],
        }

        result = backend.search("test")
        assert len(result.sources[0].snippet) == 10

    def test_payload_includes_expected_defaults(
        self, backend_and_client: tuple[TavilySearchBackend, _FakeTavilyClient]
    ) -> None:
        backend, client = backend_and_client
        client._response = {"results": []}

        backend.search("test query", limit=3)
        p = client.last_payload
        assert p["query"] == "test query"
        assert p["max_results"] == 3
        assert p["search_depth"] == "basic"
        assert p["topic"] == "general"
        assert p["include_usage"] is True
        assert p["include_raw_content"] is False


# --- Cost extraction ---


class TestTavilyCostExtraction:
    @pytest.mark.parametrize(
        "response,expected",
        [
            ({"usage": {"credits": 2}}, 2.0),
            ({}, 0.0),
            ({"usage": {"credits": "bad"}}, 0.0),
            ({"usage": "invalid"}, 0.0),
        ],
    )
    def test_extract_cost(self, response: dict, expected: float) -> None:
        assert TavilySearchBackend._extract_cost(response) == expected


# --- Date conversion ---


class TestTavilyDateConversion:
    def test_valid_date(self) -> None:
        assert TavilySearchBackend._date_mmddyyyy_to_iso("01/15/2026") == "2026-01-15"

    def test_invalid_date_raises(self) -> None:
        with pytest.raises(SearchNetworkError, match="Expected MM/DD/YYYY"):
            TavilySearchBackend._date_mmddyyyy_to_iso("2026-01-15")


# --- Serialize ---


class TestTavilySerialize:
    def test_serialize_returns_config(
        self, backend_and_client: tuple[TavilySearchBackend, _FakeTavilyClient]
    ) -> None:
        backend, _ = backend_and_client
        s = backend.serialize()
        assert s["info"]["config"]["search"]["search_class"] == "tavily"
        assert s["info"]["config"]["search"]["search_depth"] == "basic"
        assert s["info"]["config"]["search"]["topic"] == "general"
