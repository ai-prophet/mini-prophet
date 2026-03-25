from __future__ import annotations

import asyncio

import pytest

from miniprophet.environment.source_board import Source, SourceBoard
from miniprophet.exceptions import Submitted
from miniprophet.tools.submit import SubmitTool


def test_submit_tool_raises_submitted_on_valid_probability(dummy_source: Source) -> None:
    board = SourceBoard()
    board.add(dummy_source, "note", source_id="S1")
    tool = SubmitTool(board=board)

    with pytest.raises(Submitted) as exc:
        asyncio.run(tool.execute({"probability": 0.7}))

    payload = exc.value.messages[0]
    assert payload["extra"]["exit_status"] == "submitted"
    assert payload["extra"]["submission"]["Yes"] == pytest.approx(0.7)
    assert payload["extra"]["submission"]["No"] == pytest.approx(0.3)


def test_submit_tool_accepts_zero_probability() -> None:
    tool = SubmitTool(board=SourceBoard())
    with pytest.raises(Submitted) as exc:
        asyncio.run(tool.execute({"probability": 0}))

    payload = exc.value.messages[0]
    assert payload["extra"]["submission"]["Yes"] == 0.0
    assert payload["extra"]["submission"]["No"] == pytest.approx(1.0)


def test_submit_tool_rejects_invalid_probability() -> None:
    tool = SubmitTool(board=SourceBoard())

    output = asyncio.run(tool.execute({"probability": 2.0}))
    assert output["error"] is True
    assert "must be between 0 and 1" in output["output"]


def test_submit_tool_rejects_non_number() -> None:
    tool = SubmitTool(board=SourceBoard())

    output = asyncio.run(tool.execute({"probability": "high"}))
    assert output["error"] is True
    assert "must be a number" in output["output"]
