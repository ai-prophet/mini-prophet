"""Forecast task row schemas and conversions."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from miniprophet.eval.types import ForecastProblem


class ForecastTaskRow(BaseModel):
    """Standardized schema for one forecast-eval task row."""

    model_config = ConfigDict(extra="allow")

    task_id: str | None = None
    title: str
    context: str | None = None
    outcomes: list[str] = Field(default_factory=lambda: ["Yes", "No"])
    ground_truth: dict[str, int] | None = None
    predict_by: str | None = None
    source: str | None = None
    criteria: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _merge_extra_into_metadata(cls, raw: Any) -> Any:
        if not isinstance(raw, dict):
            return raw

        known = {
            "task_id",
            "title",
            "context",
            "outcomes",
            "ground_truth",
            "predict_by",
            "source",
            "criteria",
            "metadata",
        }

        data = dict(raw)

        extra_fields = {k: v for k, v in data.items() if k not in known}
        if extra_fields:
            base_metadata = data.get("metadata")
            metadata = dict(base_metadata) if isinstance(base_metadata, dict) else {}
            metadata["_extra_fields"] = extra_fields
            data["metadata"] = metadata

        return data

    @field_validator("title")
    @classmethod
    def _validate_title(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("title must be a non-empty string")
        return value

    @field_validator("outcomes")
    @classmethod
    def _validate_outcomes(cls, value: list[str]) -> list[str]:
        cleaned = [o.strip() for o in value if isinstance(o, str) and o.strip()]
        if len(cleaned) < 1:
            raise ValueError("outcomes must contain at least 1 non-empty string")
        return cleaned

    @field_validator("predict_by")
    @classmethod
    def _validate_predict_by(cls, value: str | None) -> str | None:
        if value is None:
            return None
        # Accept common date/datetime formats.
        from dateutil import parser

        parsed = parser.parse(value)
        if parsed.tzinfo is None:
            # keep naive values as-is; only validate parseability.
            _ = datetime(parsed.year, parsed.month, parsed.day)
        return value

    @model_validator(mode="after")
    def _validate_ground_truth(self) -> ForecastTaskRow:
        if self.ground_truth is None:
            return self
        gt_keys = set(self.ground_truth)
        outcomes = set(self.outcomes)
        if gt_keys != outcomes:
            missing = outcomes - gt_keys
            extra = gt_keys - outcomes
            errors: list[str] = []
            if missing:
                errors.append(f"missing outcomes: {sorted(missing)}")
            if extra:
                errors.append(f"unknown outcomes: {sorted(extra)}")
            raise ValueError("ground_truth keys mismatch: " + "; ".join(errors))
        invalid = [k for k, v in self.ground_truth.items() if v not in (0, 1)]
        if invalid:
            raise ValueError(f"ground_truth values must be 0/1 for outcomes: {invalid}")
        return self


def row_to_problem(row: ForecastTaskRow, task_id: str, offset: int = 0) -> ForecastProblem:
    return ForecastProblem(
        task_id=task_id,
        title=row.title,
        outcomes=row.outcomes,
        ground_truth=row.ground_truth,
        predict_by=row.predict_by,
        context=row.context,
        source=row.source,
        criteria=row.criteria,
        metadata=row.metadata,
        offset=offset,
    )
