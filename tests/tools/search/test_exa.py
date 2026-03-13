from __future__ import annotations

import types
from typing import Any

import pytest

from miniprophet.exceptions import SearchAuthError, SearchNetworkError, SearchRateLimitError
from miniprophet.tools.search.exa import CONTENT_NOT_AVAILABLE, ExaSearchBackend

# --- Date utility tests (existing) ---


def test_exa_date_mmddyyyy_to_iso_start_and_end() -> None:
    assert (
        ExaSearchBackend._date_mmddyyyy_to_iso("01/02/2026", end_of_day=False)
        == "2026-01-02T00:00:00Z"
    )
    assert (
        ExaSearchBackend._date_mmddyyyy_to_iso("01/02/2026", end_of_day=True)
        == "2026-01-02T23:59:59Z"
    )


def test_exa_date_mmddyyyy_to_iso_rejects_invalid_date() -> None:
    with pytest.raises(SearchNetworkError, match="Expected MM/DD/YYYY"):
        ExaSearchBackend._date_mmddyyyy_to_iso("2026-01-02", end_of_day=False)


# --- Backend tests ---


class _FakeExaClient:
    def __init__(self) -> None:
        self._response: Any = None
        self._raise: Exception | None = None
        self.last_payload: dict = {}

    def search(self, **kwargs) -> Any:
        self.last_payload = kwargs
        if self._raise is not None:
            raise self._raise
        return self._response


def _make_result(
    url: str = "https://example.com",
    title: str = "Title",
    text: str = "body text",
    published_date: str | None = "2026-01-01",
    highlights: list[str] | None = None,
    summary: str = "",
) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        url=url,
        title=title,
        text=text,
        published_date=published_date,
        highlights=highlights,
        summary=summary,
    )


@pytest.fixture
def backend_and_client(monkeypatch: pytest.MonkeyPatch) -> tuple[ExaSearchBackend, _FakeExaClient]:
    monkeypatch.setenv("EXA_API_KEY", "test-key")
    fake_client = _FakeExaClient()
    monkeypatch.setattr("miniprophet.tools.search.exa.Exa", lambda api_key: fake_client)
    backend = ExaSearchBackend()
    return backend, fake_client


class TestExaInit:
    def test_missing_api_key_raises_auth_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("EXA_API_KEY", raising=False)
        with pytest.raises(SearchAuthError, match="EXA_API_KEY"):
            ExaSearchBackend()

    def test_invalid_content_mode_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EXA_API_KEY", "key")
        monkeypatch.setattr("miniprophet.tools.search.exa.Exa", lambda api_key: None)
        with pytest.raises(ValueError, match="content_mode"):
            ExaSearchBackend(content_mode="invalid")


class TestExaSearch:
    def test_successful_search_returns_sources(
        self, backend_and_client: tuple[ExaSearchBackend, _FakeExaClient]
    ) -> None:
        backend, client = backend_and_client
        client._response = types.SimpleNamespace(
            results=[_make_result(), _make_result(url="https://b.com", title="B")],
            cost_dollars=types.SimpleNamespace(total=0.05),
        )

        result = backend.search("test query", limit=5)
        assert len(result.sources) == 2
        assert result.sources[0].url == "https://example.com"
        assert result.cost == pytest.approx(0.05)

    def test_search_skips_empty_url_results(
        self, backend_and_client: tuple[ExaSearchBackend, _FakeExaClient]
    ) -> None:
        backend, client = backend_and_client
        client._response = types.SimpleNamespace(
            results=[_make_result(url=""), _make_result(url="https://valid.com")],
            cost_dollars=None,
        )

        result = backend.search("query")
        assert len(result.sources) == 1
        assert result.sources[0].url == "https://valid.com"
        assert result.cost == 0.0

    def test_date_filters_converted_and_forwarded(
        self, backend_and_client: tuple[ExaSearchBackend, _FakeExaClient]
    ) -> None:
        backend, client = backend_and_client
        client._response = types.SimpleNamespace(results=[], cost_dollars=None)

        backend.search("test", search_date_after="01/01/2026", search_date_before="03/01/2026")
        assert client.last_payload["start_published_date"] == "2026-01-01T00:00:00Z"
        assert client.last_payload["end_published_date"] == "2026-03-01T23:59:59Z"

    @pytest.mark.parametrize(
        "status_code,exc_type,match",
        [
            (401, SearchAuthError, "authentication failed"),
            (429, SearchRateLimitError, "rate limit"),
        ],
    )
    def test_http_error_raises_expected(
        self,
        backend_and_client: tuple[ExaSearchBackend, _FakeExaClient],
        status_code: int,
        exc_type: type,
        match: str,
    ) -> None:
        backend, client = backend_and_client
        exc = Exception("error")
        exc.status_code = status_code  # type: ignore[attr-defined]
        client._raise = exc
        with pytest.raises(exc_type, match=match):
            backend.search("test")

    def test_generic_error_raises_network_error(
        self, backend_and_client: tuple[ExaSearchBackend, _FakeExaClient]
    ) -> None:
        backend, client = backend_and_client
        client._raise = RuntimeError("connection failed")
        with pytest.raises(SearchNetworkError, match="request failed"):
            backend.search("test")


class TestExaSnippetExtraction:
    def test_text_mode_prefers_text(
        self, backend_and_client: tuple[ExaSearchBackend, _FakeExaClient]
    ) -> None:
        backend, _ = backend_and_client
        item = _make_result(text="body text", highlights=["highlight"])
        snippet = backend._extract_snippet(item)
        assert snippet == "body text"

    def test_text_mode_falls_back_to_highlights(
        self, backend_and_client: tuple[ExaSearchBackend, _FakeExaClient]
    ) -> None:
        backend, _ = backend_and_client
        item = _make_result(text="", highlights=["h1", "h2"])
        snippet = backend._extract_snippet(item)
        assert snippet == "h1\nh2"

    def test_text_mode_falls_back_to_content_not_available(
        self, backend_and_client: tuple[ExaSearchBackend, _FakeExaClient]
    ) -> None:
        backend, _ = backend_and_client
        item = _make_result(text="", highlights=None, summary="")
        snippet = backend._extract_snippet(item)
        assert snippet == CONTENT_NOT_AVAILABLE

    def test_highlights_mode_prefers_highlights(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EXA_API_KEY", "key")
        fake_client = _FakeExaClient()
        monkeypatch.setattr("miniprophet.tools.search.exa.Exa", lambda api_key: fake_client)
        backend = ExaSearchBackend(content_mode="highlights")

        item = _make_result(text="body", highlights=["h1"])
        snippet = backend._extract_snippet(item)
        assert snippet == "h1"


class TestExaHelpers:
    @pytest.mark.parametrize(
        "obj,key,expected",
        [
            ({"a": 1}, "a", 1),
            ({"a": 1}, "b", None),
            (types.SimpleNamespace(x=42), "x", 42),
            (types.SimpleNamespace(x=42), "y", None),
            (None, "x", None),
        ],
    )
    def test_get_field(self, obj, key, expected) -> None:
        assert ExaSearchBackend._get_field(obj, key) == expected

    @pytest.mark.parametrize("val,expected", [("hello", "hello"), (123, ""), (None, "")])
    def test_as_str(self, val, expected) -> None:
        assert ExaSearchBackend._as_str(val) == expected

    def test_build_contents_payload_text(
        self, backend_and_client: tuple[ExaSearchBackend, _FakeExaClient]
    ) -> None:
        backend, _ = backend_and_client
        payload = backend._build_contents_payload()
        assert "text" in payload

    def test_build_contents_payload_highlights(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EXA_API_KEY", "key")
        monkeypatch.setattr("miniprophet.tools.search.exa.Exa", lambda api_key: None)
        backend = ExaSearchBackend(content_mode="highlights")
        payload = backend._build_contents_payload()
        assert "highlights" in payload

    def test_extract_cost_missing(
        self, backend_and_client: tuple[ExaSearchBackend, _FakeExaClient]
    ) -> None:
        backend, _ = backend_and_client
        resp = types.SimpleNamespace(cost_dollars=None)
        assert backend._extract_cost(resp) == 0.0

    def test_extract_cost_invalid(
        self, backend_and_client: tuple[ExaSearchBackend, _FakeExaClient]
    ) -> None:
        backend, _ = backend_and_client
        resp = types.SimpleNamespace(cost_dollars=types.SimpleNamespace(total="invalid"))
        assert backend._extract_cost(resp) == 0.0


class TestExaSerialize:
    def test_serialize_returns_config(
        self, backend_and_client: tuple[ExaSearchBackend, _FakeExaClient]
    ) -> None:
        backend, _ = backend_and_client
        s = backend.serialize()
        assert s["info"]["config"]["search"]["search_class"] == "exa"
        assert s["info"]["config"]["search"]["content_mode"] == "text"
