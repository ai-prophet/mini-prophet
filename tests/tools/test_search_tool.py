from __future__ import annotations

import pytest
from conftest import DummySearchTool

from miniprophet.environment.source_board import Source
from miniprophet.exceptions import SearchAuthError, SearchNetworkError
from miniprophet.tools.search_tool import SearchForecastTool


def test_search_tool_execute_success_assigns_ids(two_sources: list[Source]) -> None:
    registry: dict[str, Source] = {}
    backend = DummySearchTool(two_sources, cost=0.015)
    tool = SearchForecastTool(backend, registry, search_limit=5)

    output = tool.execute({"query": "nba finals"})

    assert output["search_cost"] == pytest.approx(0.015)
    assert [sid for sid, _ in output["search_results"]] == ["S1", "S2"]
    assert '<search_results count="2">' in output["output"]
    assert registry["S1"].title == "A"


def test_search_tool_execute_rejects_missing_query() -> None:
    tool = SearchForecastTool(DummySearchTool(), {})
    output = tool.execute({"query": "   "})
    assert output["error"] is True
    assert "'query' is required" in output["output"]


def test_search_tool_execute_handles_backend_error() -> None:
    tool = SearchForecastTool(
        DummySearchTool(error=SearchNetworkError("timeout")),
        {},
        search_limit=2,
    )
    output = tool.execute({"query": "x"})
    assert output["error"] is True
    assert "Search failed" in output["output"]


def test_search_tool_execute_bubbles_auth_error() -> None:
    tool = SearchForecastTool(DummySearchTool(error=SearchAuthError("bad key")), {})
    with pytest.raises(SearchAuthError):
        tool.execute({"query": "x"})


def test_search_tool_search_limit_reached() -> None:
    tool = SearchForecastTool(DummySearchTool(), {}, search_limit=1)
    tool.n_searches = 1
    output = tool.execute({"query": "x"})
    assert output["error"] is True
    assert "Search limit reached" in output["output"]


def test_search_tool_no_results_message() -> None:
    tool = SearchForecastTool(DummySearchTool(sources=[]), {}, search_limit=5)
    output = tool.execute({"query": "x"})
    assert "No sources found" in output["output"]


def test_search_tool_get_schema_with_backend_params() -> None:
    backend = DummySearchTool()
    tool = SearchForecastTool(backend, {})
    schema = tool.get_schema()
    assert schema["function"]["name"] == "search"
    # DummySearchTool has search_parameters_schema so it should be used
    assert schema["function"]["parameters"]["properties"]["query"]["type"] == "string"


def test_search_tool_get_schema_without_backend_params() -> None:
    class NoSchemaBackend:
        def search(self, query, limit=5, **kw):
            from miniprophet.tools.search import SearchResult

            return SearchResult(sources=[], cost=0.0)

        def serialize(self):
            return {}

    tool = SearchForecastTool(NoSchemaBackend(), {})
    schema = tool.get_schema()
    assert schema["function"]["parameters"]["required"] == ["query"]
