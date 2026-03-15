from __future__ import annotations

import asyncio

from miniprophet.environment.source_board import Source, SourceBoard
from miniprophet.tools.source_board_tools import AddSourceTool, EditNoteTool


def test_add_source_tool_adds_valid_source(dummy_source: Source) -> None:
    board = SourceBoard()
    registry = {"S1": dummy_source}
    tool = AddSourceTool(source_registry=registry, board=board, outcomes=["Yes", "No"])

    output = asyncio.run(
        tool.execute({"source_id": "s1", "note": "good", "reaction": {"Yes": "positive"}})
    )

    assert "added to board as #1" in output["output"]
    assert len(board.serialize()) == 1


def test_add_source_tool_rejects_unknown_source_id() -> None:
    tool = AddSourceTool(source_registry={}, board=SourceBoard(), outcomes=["Yes", "No"])
    output = asyncio.run(tool.execute({"source_id": "S9", "note": "x"}))
    assert output["error"] is True
    assert "unknown source_id" in output["output"]


def test_edit_note_tool_updates_note_and_reaction() -> None:
    board = SourceBoard()
    board.add(Source(url="u", title="t", snippet="s"), "old", source_id="S1")
    tool = EditNoteTool(board=board, outcomes=["Yes", "No"])

    output = asyncio.run(
        tool.execute({"board_id": 1, "new_note": "new", "reaction": {"No": "negative"}})
    )

    assert output["output"] == "Note for #1 updated."
    assert board.get(1).reaction == {"No": "negative"}


def test_edit_note_tool_rejects_missing_entry() -> None:
    tool = EditNoteTool(board=SourceBoard(), outcomes=["Yes", "No"])
    output = asyncio.run(tool.execute({"board_id": 1, "new_note": "x"}))
    assert output["error"] is True
    assert "no board entry" in output["output"]


def test_add_source_tool_rejects_missing_source_id() -> None:
    tool = AddSourceTool(source_registry={}, board=SourceBoard(), outcomes=["Yes", "No"])
    output = asyncio.run(tool.execute({"source_id": "", "note": "x"}))
    assert output["error"] is True
    assert "'source_id' is required" in output["output"]


def test_add_source_tool_rejects_missing_note(dummy_source: Source) -> None:
    tool = AddSourceTool(
        source_registry={"S1": dummy_source}, board=SourceBoard(), outcomes=["Yes", "No"]
    )
    output = asyncio.run(tool.execute({"source_id": "S1", "note": ""}))
    assert output["error"] is True
    assert "'note' is required" in output["output"]


def test_add_source_tool_converts_int_source_id(dummy_source: Source) -> None:
    board = SourceBoard()
    tool = AddSourceTool(source_registry={"S3": dummy_source}, board=board, outcomes=["Yes", "No"])
    output = asyncio.run(tool.execute({"source_id": 3, "note": "good"}))
    assert "added to board" in output["output"]


def test_add_source_tool_rejects_bad_reaction(dummy_source: Source) -> None:
    tool = AddSourceTool(
        source_registry={"S1": dummy_source}, board=SourceBoard(), outcomes=["Yes", "No"]
    )
    output = asyncio.run(
        tool.execute({"source_id": "S1", "note": "ok", "reaction": {"Unknown": "positive"}})
    )
    assert output["error"] is True
    assert "Unknown outcome" in output["output"]


def test_add_source_tool_rejects_bad_sentiment(dummy_source: Source) -> None:
    tool = AddSourceTool(
        source_registry={"S1": dummy_source}, board=SourceBoard(), outcomes=["Yes", "No"]
    )
    output = asyncio.run(
        tool.execute({"source_id": "S1", "note": "ok", "reaction": {"Yes": "invalid_sentiment"}})
    )
    assert output["error"] is True
    assert "Invalid sentiment" in output["output"]


def test_edit_note_tool_rejects_missing_board_id() -> None:
    tool = EditNoteTool(board=SourceBoard(), outcomes=["Yes", "No"])
    output = asyncio.run(tool.execute({"new_note": "x"}))
    assert output["error"] is True
    assert "'board_id' is required" in output["output"]


def test_edit_note_tool_rejects_missing_note() -> None:
    tool = EditNoteTool(board=SourceBoard(), outcomes=["Yes", "No"])
    output = asyncio.run(tool.execute({"board_id": 1, "new_note": ""}))
    assert output["error"] is True
    assert "'new_note' is required" in output["output"]


def test_edit_note_tool_rejects_bad_reaction() -> None:
    board = SourceBoard()
    board.add(Source(url="u", title="t", snippet="s"), "old", source_id="S1")
    tool = EditNoteTool(board=board, outcomes=["Yes", "No"])
    output = asyncio.run(
        tool.execute({"board_id": 1, "new_note": "x", "reaction": {"Yes": "bad_val"}})
    )
    assert output["error"] is True
    assert "Invalid sentiment" in output["output"]
