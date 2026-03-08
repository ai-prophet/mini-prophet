from __future__ import annotations

from miniprophet.eval.datasets.schema import ForecastTaskRow


def test_forecast_task_row_parses_task_id_and_predict_by() -> None:
    row = ForecastTaskRow.model_validate(
        {
            "task_id": "r1",
            "title": "Will A happen?",
            "outcomes": ["Yes", "No"],
            "predict_by": "2026-03-01",
        }
    )
    assert row.task_id == "r1"
    assert row.predict_by == "2026-03-01"


def test_forecast_task_row_allows_single_outcome() -> None:
    row = ForecastTaskRow.model_validate(
        {
            "task_id": "single",
            "title": "Will A happen?",
            "outcomes": ["Yes"],
        }
    )
    assert row.outcomes == ["Yes"]
