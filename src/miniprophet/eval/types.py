"""Core data types for prophet eval."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


def to_mm_dd_yyyy(dt_str: str, offset: int = 0) -> str:
    """Convert an arbitrary date string to MM/DD/YYYY with optional day offset."""
    from datetime import timedelta

    from dateutil import parser

    dt = parser.parse(dt_str)
    return (dt - timedelta(days=offset)).strftime("%m/%d/%Y")


@dataclass
class ForecastProblem:
    """A single standardized forecast-eval problem."""

    task_id: str
    title: str
    outcomes: list[str]
    ground_truth: dict[str, int] | None = None
    predict_by: str | None = None
    context: str | None = None
    source: str | None = None
    criteria: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    retries: int = 0
    offset: int = 0

    def __post_init__(self) -> None:
        if self.predict_by:
            try:
                self.predict_by = to_mm_dd_yyyy(self.predict_by, self.offset)
            except ValueError:
                self.predict_by = None


@dataclass
class ForecastResult:
    """Result of a single forecast run from the public batch API."""

    task_id: str
    title: str
    status: str = "pending"
    submission: dict[str, float] | None = None
    evaluation: dict[str, float] | None = None
    cost: dict[str, float] = field(
        default_factory=lambda: {"model": 0.0, "search": 0.0, "total": 0.0}
    )
    error: str | None = None
    trajectory: dict | None = None


@runtime_checkable
class BatchProgressCallback(Protocol):
    """Protocol for receiving progress updates from batch_forecast."""

    def on_run_start(self, task_id: str) -> None: ...

    def update_run_status(self, task_id: str, message: str, cost_delta: float = 0.0) -> None: ...

    def on_run_end(self, task_id: str, exit_status: str | None) -> None: ...
