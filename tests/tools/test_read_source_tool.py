from __future__ import annotations

import asyncio

from miniprophet.environment.source_board import Source
from miniprophet.environment.source_registry import SourceRegistry
from miniprophet.tools.read_source_tool import ReadSourceTool


def test_read_valid_source() -> None:
    registry = SourceRegistry()
    src = Source(
        url="https://example.com", title="Example", snippet="full content here", date="2026-01-01"
    )
    asyncio.run(registry.add(src))

    tool = ReadSourceTool(registry=registry)
    output = asyncio.run(tool.execute({"source_id": "S1"}))
    assert "full content here" in output["output"]
    assert 'id="S1"' in output["output"]
    assert not output.get("error")


def test_read_unknown_source() -> None:
    tool = ReadSourceTool(registry=SourceRegistry())
    output = asyncio.run(tool.execute({"source_id": "S99"}))
    assert output["error"] is True
    assert "unknown source_id" in output["output"]


def test_read_source_converts_int_id() -> None:
    registry = SourceRegistry()
    asyncio.run(registry.add(Source(url="u", title="t", snippet="s")))
    tool = ReadSourceTool(registry=registry)
    output = asyncio.run(tool.execute({"source_id": 1}))
    assert 'id="S1"' in output["output"]


def test_read_source_empty_id() -> None:
    tool = ReadSourceTool(registry=SourceRegistry())
    output = asyncio.run(tool.execute({"source_id": ""}))
    assert output["error"] is True
    assert "'source_id' is required" in output["output"]
