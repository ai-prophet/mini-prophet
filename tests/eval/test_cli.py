from __future__ import annotations

import json
from pathlib import Path

import pytest
import typer

from miniprophet.eval.cli import _build_eval_config, _parse_agent_kwargs, _resolve_resume_state
from miniprophet.eval.types import ForecastProblem


def test_build_batch_config_applies_overrides() -> None:
    cfg = _build_eval_config(
        config_spec=None,
        max_cost_per_run=0.5,
        model_name="openai/gpt-4o-mini",
        model_class="litellm",
    )
    assert cfg["agent"]["cost_limit"] == 0.5
    assert cfg["model"]["model_name"] == "openai/gpt-4o-mini"
    assert cfg["model"]["model_class"] == "litellm"


def test_resolve_resume_state_disabled_returns_inputs(tmp_path: Path) -> None:
    problems = [ForecastProblem(task_id="r1", title="T", outcomes=["A", "B"])]
    filtered, results, total = _resolve_resume_state(
        enabled=False, output=tmp_path, problems=problems
    )
    assert filtered == problems
    assert results == {}
    assert total == 0.0


def test_resolve_resume_state_filters_completed_runs(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "total_cost": 1.0,
                "runs": [
                    {"task_id": "r1", "title": "t", "status": "submitted", "cost": {"total": 0.2}}
                ],
            }
        )
    )
    problems = [
        ForecastProblem(task_id="r1", title="T1", outcomes=["A", "B"]),
        ForecastProblem(task_id="r2", title="T2", outcomes=["A", "B"]),
    ]

    filtered, results, total = _resolve_resume_state(
        enabled=True, output=tmp_path, problems=problems
    )

    assert [p.task_id for p in filtered] == ["r2"]
    assert "r1" in results
    assert total == pytest.approx(1.0)


def test_resolve_resume_state_rejects_unexpected_run_ids(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "total_cost": 1.0,
                "runs": [{"task_id": "unknown", "title": "t", "status": "submitted"}],
            }
        )
    )
    problems = [ForecastProblem(task_id="r1", title="T1", outcomes=["A", "B"])]

    with pytest.raises(typer.Exit):
        _resolve_resume_state(enabled=True, output=tmp_path, problems=problems)


class TestParseAgentKwargs:
    def test_empty_input(self) -> None:
        assert _parse_agent_kwargs(None) == {}
        assert _parse_agent_kwargs([]) == {}

    def test_valid_string_value(self) -> None:
        result = _parse_agent_kwargs(["name=hello"])
        assert result == {"name": "hello"}

    def test_json_value_parsed(self) -> None:
        result = _parse_agent_kwargs(["count=42", "flag=true"])
        assert result == {"count": 42, "flag": True}

    def test_missing_equals_raises(self) -> None:
        with pytest.raises(ValueError, match="expected key=value"):
            _parse_agent_kwargs(["no-equals-here"])

    def test_empty_key_raises(self) -> None:
        with pytest.raises(ValueError, match="empty key"):
            _parse_agent_kwargs(["=value"])


def test_resolve_resume_state_no_summary_starts_fresh(tmp_path: Path) -> None:
    problems = [ForecastProblem(task_id="r1", title="T", outcomes=["A", "B"])]
    filtered, results, total = _resolve_resume_state(
        enabled=True, output=tmp_path, problems=problems
    )
    assert filtered == problems
    assert results == {}
    assert total == 0.0
