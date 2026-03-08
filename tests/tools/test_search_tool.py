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
