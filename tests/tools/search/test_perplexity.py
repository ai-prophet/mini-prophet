from __future__ import annotations

import asyncio
import types
from typing import Any

import pytest

from miniprophet.exceptions import SearchAuthError, SearchNetworkError, SearchRateLimitError
from miniprophet.tools.search.perplexity import (
    PERPLEXITY_PER_SEARCH_COST,
    PerplexitySearchBackend,
)


class _FakeClient:
    """Stub for the Perplexity SDK client."""

    def __init__(self) -> None:
        self.search = types.SimpleNamespace(create=self._create)
        self._response: Any = None
        self._raise: Exception | None = None
        self.last_payload: dict = {}

    async def _create(self, **kwargs) -> Any:
        self.last_payload = kwargs
        if self._raise is not None:
            raise self._raise
        return self._response


def _make_result(url: str, title: str, snippet: str, date: str | None, last_updated: str | None):
    return types.SimpleNamespace(
        url=url, title=title, snippet=snippet, date=date, last_updated=last_updated
    )


@pytest.fixture
def backend_and_client(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[PerplexitySearchBackend, _FakeClient]:
    monkeypatch.setenv("PERPLEXITY_API_KEY", "test-key")
    fake_client = _FakeClient()
    monkeypatch.setattr(
        "miniprophet.tools.search.perplexity.Perplexity",
        lambda api_key, timeout: fake_client,
    )
    backend = PerplexitySearchBackend()
    backend._async_client = fake_client
    return backend, fake_client


class TestPerplexityInit:
    def test_missing_api_key_raises_auth_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
        with pytest.raises(SearchAuthError, match="PERPLEXITY_API_KEY"):
            PerplexitySearchBackend()


class TestPerplexitySearch:
    def test_successful_search_returns_sources(
        self, backend_and_client: tuple[PerplexitySearchBackend, _FakeClient]
    ) -> None:
        backend, client = backend_and_client
        client._response = types.SimpleNamespace(
            results=[
                _make_result("https://a.com", "A", "snippet a", "2026-01-01", None),
                _make_result("https://b.com", "B", "snippet b", "2026-01-01", "2026-02-01"),
            ]
        )

        result = asyncio.run(backend.search("test query", limit=5))
        assert len(result.sources) == 2
        assert result.cost == pytest.approx(PERPLEXITY_PER_SEARCH_COST)
        assert result.sources[0].url == "https://a.com"
        # When both dates present, takes the latest
        assert result.sources[1].date == "2026-02-01"

    def test_date_filters_forwarded(
        self, backend_and_client: tuple[PerplexitySearchBackend, _FakeClient]
    ) -> None:
        backend, client = backend_and_client
        client._response = types.SimpleNamespace(results=[])

        asyncio.run(
            backend.search("test", search_date_after="01/01/2026", search_date_before="03/01/2026")
        )
        assert client.last_payload["search_after_date_filter"] == "01/01/2026"
        assert client.last_payload["search_before_date_filter"] == "03/01/2026"

    @pytest.mark.parametrize(
        "status_code,exc_type,match",
        [
            (401, SearchAuthError, "authentication failed"),
            (429, SearchRateLimitError, "rate limit"),
        ],
    )
    def test_http_error_raises_expected(
        self,
        backend_and_client: tuple[PerplexitySearchBackend, _FakeClient],
        status_code: int,
        exc_type: type,
        match: str,
    ) -> None:
        backend, client = backend_and_client
        exc = Exception("error")
        exc.status_code = status_code  # type: ignore[attr-defined]
        client._raise = exc
        with pytest.raises(exc_type, match=match):
            asyncio.run(backend.search("test"))

    def test_network_error_raises_search_network_error(
        self, backend_and_client: tuple[PerplexitySearchBackend, _FakeClient]
    ) -> None:
        backend, client = backend_and_client
        client._raise = ConnectionError("connection refused")
        with pytest.raises(SearchNetworkError, match="request failed"):
            asyncio.run(backend.search("test"))

    def test_status_code_from_response_attribute(
        self, backend_and_client: tuple[PerplexitySearchBackend, _FakeClient]
    ) -> None:
        backend, client = backend_and_client
        exc = Exception("error")
        exc.response = types.SimpleNamespace(status_code=401)  # type: ignore[attr-defined]
        client._raise = exc

        with pytest.raises(SearchAuthError):
            asyncio.run(backend.search("test"))


class TestPerplexitySerialize:
    def test_serialize_returns_config(
        self, backend_and_client: tuple[PerplexitySearchBackend, _FakeClient]
    ) -> None:
        backend, _ = backend_and_client
        s = backend.serialize()
        assert s["info"]["config"]["search"]["search_class"] == "perplexity"
        assert s["info"]["config"]["search"]["timeout"] == 30
