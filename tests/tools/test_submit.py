from __future__ import annotations

import asyncio

import pytest

from miniprophet.environment.source_board import Source, SourceBoard
from miniprophet.exceptions import Submitted
from miniprophet.tools.submit import SubmitTool


def test_submit_tool_raises_submitted_on_valid_probs(dummy_source: Source) -> None:
    board = SourceBoard()
    board.add(dummy_source, "note", source_id="S1")
    tool = SubmitTool(outcomes=["Yes", "No"], board=board)

    with pytest.raises(Submitted) as exc:
        asyncio.run(tool.execute({"probabilities": {"Yes": 0.7, "No": 0.3}}))

    payload = exc.value.messages[0]
    assert payload["extra"]["exit_status"] == "submitted"
    assert payload["extra"]["submission"] == {"Yes": 0.7, "No": 0.3}


def test_submit_tool_rejects_invalid_probabilities() -> None:
    tool = SubmitTool(outcomes=["Yes", "No"], board=SourceBoard())
    output = asyncio.run(tool.execute({"probabilities": {"Yes": 2.0}}))

    assert output["error"] is True
    assert "Missing probability" in output["output"]
    assert "must be a number between 0 and 1" in output["output"]
