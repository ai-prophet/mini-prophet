from __future__ import annotations

import asyncio

import pytest

from miniprophet.environment.source_registry import Source, SourceRegistry
from miniprophet.exceptions import Submitted
from miniprophet.tools.submit import SubmitTool


def test_submit_tool_raises_submitted_on_valid_probability(dummy_source: Source) -> None:
    registry = SourceRegistry()
    asyncio.run(registry.add(dummy_source, problem_id="main"))
    tool = SubmitTool(registry=registry)

    with pytest.raises(Submitted) as exc:
        asyncio.run(tool.execute({"probability": 0.7, "rationale": "evidence supports yes"}))

    payload = exc.value.messages[0]
    assert payload["extra"]["exit_status"] == "submitted"
    assert payload["extra"]["submission"]["Yes"] == pytest.approx(0.7)
    assert payload["extra"]["submission"]["No"] == pytest.approx(0.3)
    assert payload["extra"]["rationale"] == "evidence supports yes"
    assert "sources" in payload["extra"]
    assert "S1" in payload["extra"]["sources"]


def test_submit_tool_accepts_zero_probability() -> None:
    tool = SubmitTool(registry=SourceRegistry())
    with pytest.raises(Submitted) as exc:
        asyncio.run(tool.execute({"probability": 0, "rationale": "not binary"}))

    payload = exc.value.messages[0]
    assert payload["extra"]["submission"]["Yes"] == 0.0
    assert payload["extra"]["submission"]["No"] == pytest.approx(1.0)


def test_submit_tool_rejects_invalid_probability() -> None:
    tool = SubmitTool(registry=SourceRegistry())

    output = asyncio.run(tool.execute({"probability": 2.0}))
    assert output["error"] is True
    assert "must be between 0 and 1" in output["output"]


def test_submit_tool_rejects_non_number() -> None:
    tool = SubmitTool(registry=SourceRegistry())

    output = asyncio.run(tool.execute({"probability": "high"}))
    assert output["error"] is True
    assert "must be a number" in output["output"]
