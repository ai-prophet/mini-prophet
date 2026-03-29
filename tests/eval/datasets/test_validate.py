from __future__ import annotations

import json
from pathlib import Path

import pytest

from miniprophet.eval.datasets.validate import load_problems


def test_load_problems_assigns_auto_task_ids(tmp_path: Path) -> None:
    path = tmp_path / "problems.jsonl"
    path.write_text(
        '{"title": "Q1", "outcomes": ["A", "B"]}\n'
        '{"title": "Q2", "outcomes": ["C", "D"], "task_id": "custom"}\n'
    )

    probs = load_problems(path)
    assert [p.task_id for p in probs] == ["task_0", "custom"]


def test_load_problems_rejects_duplicate_run_ids(tmp_path: Path) -> None:
    path = tmp_path / "problems.jsonl"
    path.write_text(
        '{"title": "Q1", "outcomes": ["A", "B"], "task_id": "x"}\n'
        '{"title": "Q2", "outcomes": ["C", "D"], "task_id": "x"}\n'
    )

    with pytest.raises(ValueError, match="duplicate task_id"):
        load_problems(path)


def test_load_problems_validates_schema_and_extra_fields(tmp_path: Path) -> None:
    path = tmp_path / "tasks.jsonl"
    path.write_text(
        json.dumps(
            {
                "task_id": "t1",
                "title": "Will B happen?",
                "outcomes": ["Yes", "No"],
                "random_col": "kept",
            }
        )
        + "\n"
    )

    problems = load_problems(path)
    assert problems[0].task_id == "t1"
    assert problems[0].metadata["_extra_fields"]["random_col"] == "kept"


def test_load_problems_defaults_outcomes_to_binary(tmp_path: Path) -> None:
    """Outcomes default to ["Yes", "No"] when not specified."""
    path = tmp_path / "binary.jsonl"
    path.write_text(json.dumps({"title": "Will it rain?"}) + "\n")

    problems = load_problems(path)
    assert problems[0].outcomes == ["Yes", "No"]
