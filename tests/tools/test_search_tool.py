from __future__ import annotations

import asyncio

import pytest
from conftest import DummySearchTool

from miniprophet.environment.source_registry import Source, SourceRegistry
from miniprophet.exceptions import SearchAuthError, SearchNetworkError
from miniprophet.tools.search_tool import SearchForecastTool


def test_search_tool_execute_success_assigns_ids(two_sources: list[Source]) -> None:
    registry = SourceRegistry(max_gist_chars=200)
    backend = DummySearchTool(two_sources, cost=0.015)
    tool = SearchForecastTool(backend, registry, search_limit=5)

    output = asyncio.run(tool.execute({"query": "nba finals"}))

    assert output["search_cost"] == pytest.approx(0.015)
    assert [sid for sid, _ in output["search_results"]] == ["S1", "S2"]
    assert '<search_results count="2">' in output["output"]
    # Verify sources are in the registry
    entry = asyncio.run(registry.get("S1"))
    assert entry.source.title == "A"


def test_search_tool_execute_rejects_missing_query() -> None:
    tool = SearchForecastTool(DummySearchTool(), SourceRegistry())
    output = asyncio.run(tool.execute({"query": "   "}))
    assert output["error"] is True
    assert "'query' is required" in output["output"]


def test_search_tool_execute_handles_backend_error() -> None:
    tool = SearchForecastTool(
        DummySearchTool(error=SearchNetworkError("timeout")),
        SourceRegistry(),
        search_limit=2,
    )
    output = asyncio.run(tool.execute({"query": "x"}))
    assert output["error"] is True
    assert "Search failed" in output["output"]


def test_search_tool_execute_bubbles_auth_error() -> None:
    tool = SearchForecastTool(DummySearchTool(error=SearchAuthError("bad key")), SourceRegistry())
    with pytest.raises(SearchAuthError):
        asyncio.run(tool.execute({"query": "x"}))


def test_search_tool_search_limit_reached() -> None:
    tool = SearchForecastTool(DummySearchTool(), SourceRegistry(), search_limit=1)
    tool.n_searches = 1
    output = asyncio.run(tool.execute({"query": "x"}))
    assert output["error"] is True
    assert "Search limit reached" in output["output"]


def test_search_tool_no_results_message() -> None:
    tool = SearchForecastTool(DummySearchTool(sources=[]), SourceRegistry(), search_limit=5)
    output = asyncio.run(tool.execute({"query": "x"}))
    assert "No sources found" in output["output"]


def test_search_tool_get_schema_with_backend_params() -> None:
    backend = DummySearchTool()
    tool = SearchForecastTool(backend, SourceRegistry())
    schema = tool.get_schema()
    assert schema["function"]["name"] == "search"
    assert schema["function"]["parameters"]["properties"]["query"]["type"] == "string"


def test_search_tool_get_schema_without_backend_params() -> None:
    class NoSchemaBackend:
        async def search(self, query, limit=5, **kw):
            from miniprophet.tools.search import SearchResult

            return SearchResult(sources=[], cost=0.0)

        def serialize(self):
            return {}

    tool = SearchForecastTool(NoSchemaBackend(), SourceRegistry())
    schema = tool.get_schema()
    assert schema["function"]["parameters"]["required"] == ["query"]
