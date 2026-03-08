from __future__ import annotations

from miniprophet.environment.source_board import Source, SourceBoard
from miniprophet.tools.source_board_tools import AddSourceTool, EditNoteTool


def test_add_source_tool_adds_valid_source(dummy_source: Source) -> None:
    board = SourceBoard()
    registry = {"S1": dummy_source}
    tool = AddSourceTool(source_registry=registry, board=board, outcomes=["Yes", "No"])

    output = tool.execute({"source_id": "s1", "note": "good", "reaction": {"Yes": "positive"}})

    assert "added to board as #1" in output["output"]
    assert len(board.serialize()) == 1


def test_add_source_tool_rejects_unknown_source_id() -> None:
    tool = AddSourceTool(source_registry={}, board=SourceBoard(), outcomes=["Yes", "No"])
    output = tool.execute({"source_id": "S9", "note": "x"})
    assert output["error"] is True
    assert "unknown source_id" in output["output"]


def test_edit_note_tool_updates_note_and_reaction() -> None:
    board = SourceBoard()
    board.add(Source(url="u", title="t", snippet="s"), "old", source_id="S1")
    tool = EditNoteTool(board=board, outcomes=["Yes", "No"])

    output = tool.execute({"board_id": 1, "new_note": "new", "reaction": {"No": "negative"}})

    assert output["output"] == "Note for #1 updated."
    assert board.get(1).reaction == {"No": "negative"}


def test_edit_note_tool_rejects_missing_entry() -> None:
    tool = EditNoteTool(board=SourceBoard(), outcomes=["Yes", "No"])
    output = tool.execute({"board_id": 1, "new_note": "x"})
    assert output["error"] is True
    assert "no board entry" in output["output"]
