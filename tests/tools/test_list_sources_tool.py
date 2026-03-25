from __future__ import annotations

import asyncio

from miniprophet.environment.source_board import Source
from miniprophet.environment.source_registry import SourceRegistry
from miniprophet.tools.list_sources_tool import ListSourcesTool


def test_list_all_sources() -> None:
    registry = SourceRegistry(max_gist_chars=200)
    asyncio.run(registry.add(Source(url="u1", title="A", snippet="first"), problem_id="main"))
    asyncio.run(registry.add(Source(url="u2", title="B", snippet="second"), problem_id="sub1"))

    tool = ListSourcesTool(registry=registry)
    output = asyncio.run(tool.execute({}))
    assert '<sources count="2">' in output["output"]
    assert "S1" in output["output"]
    assert "S2" in output["output"]


def test_list_empty() -> None:
    tool = ListSourcesTool(registry=SourceRegistry())
    output = asyncio.run(tool.execute({}))
    assert "No sources found" in output["output"]


def test_list_filter_problem_id() -> None:
    registry = SourceRegistry()
    asyncio.run(registry.add(Source(url="u1", title="A", snippet="a"), problem_id="main"))
    asyncio.run(registry.add(Source(url="u2", title="B", snippet="b"), problem_id="sub1"))

    tool = ListSourcesTool(registry=registry)
    output = asyncio.run(tool.execute({"problem_id": "sub1"}))
    assert "S2" in output["output"]
    assert "S1" not in output["output"]


def test_list_filter_no_match() -> None:
    registry = SourceRegistry()
    asyncio.run(registry.add(Source(url="u1", title="A", snippet="a"), problem_id="main"))

    tool = ListSourcesTool(registry=registry)
    output = asyncio.run(tool.execute({"problem_id": "nonexistent"}))
    assert "No sources found" in output["output"]
