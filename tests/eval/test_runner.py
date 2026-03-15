from __future__ import annotations

import json
from pathlib import Path

import pytest

from miniprophet.eval.batch import _is_auth_error, _is_rate_limit_error
from miniprophet.eval.runner import load_existing_summary
from miniprophet.eval.types import ForecastProblem, to_mm_dd_yyyy
from miniprophet.exceptions import SearchRateLimitError


def test_to_mm_dd_yyyy_applies_offset() -> None:
    assert to_mm_dd_yyyy("2026-01-10", offset=2) == "01/08/2026"


def test_forecast_problem_invalid_end_time_becomes_none() -> None:
    p = ForecastProblem(task_id="r1", title="t", outcomes=["a", "b"], predict_by="not-a-date")
    assert p.predict_by is None


def test_load_existing_summary_reads_results_and_total_cost(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "total_cost": 1.2,
                "runs": [
                    {
                        "task_id": "r1",
                        "title": "t",
                        "status": "submitted",
                        "cost": {"model": 0.2, "search": 0.3, "total": 0.5},
                    }
                ],
            }
        )
    )

    runs, total = load_existing_summary(summary)
    assert total == pytest.approx(1.2)
    assert runs["r1"].status == "submitted"


def test_load_existing_summary_rejects_bad_shape(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text('{"runs": {"not": "a-list"}}')
    with pytest.raises(ValueError, match="must be a list"):
        load_existing_summary(summary)


@pytest.mark.parametrize(
    "exc,expected",
    [
        (SearchRateLimitError("x"), True),
        (RuntimeError("HTTP 429 from provider"), True),
        (RuntimeError("other"), False),
    ],
)
def test_is_rate_limit_error(exc: Exception, expected: bool) -> None:
    assert _is_rate_limit_error(exc) is expected


@pytest.mark.parametrize(
    "exc,expected",
    [
        (RuntimeError("Authentication failed"), True),
        (RuntimeError("status code 401"), True),
        (RuntimeError("something else"), False),
    ],
)
def test_is_auth_error(exc: Exception, expected: bool) -> None:
    assert _is_auth_error(exc) is expected
