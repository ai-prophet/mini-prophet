"""Tests for the public batch_forecast API."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from miniprophet.eval.batch import (
    _is_auth_error,
    _is_rate_limit_error,
    _load_config,
    batch_forecast_sync,
)
from miniprophet.eval.types import (
    BatchProgressCallback,
    ForecastProblem,
    ForecastResult,
)
from miniprophet.exceptions import SearchRateLimitError


class _SimpleAgent:
    """Minimal agent that returns a fixed forecast."""

    def __init__(self, model: Any, env: Any, **kwargs: Any) -> None:
        self.model = model
        self.env = env
        self.model_cost = 0.1
        self.search_cost = 0.05
        self.total_cost = 0.15
        self.n_calls = 1

    async def step(self) -> list[dict]:
        return [{"role": "tool", "content": "ok"}]

    async def run(self, *, title: str, ground_truth: Any = None, **kw: Any) -> dict:
        return {
            "exit_status": "submitted",
            "submission": {"Yes": 0.5, "No": 0.5},
        }

    def run_sync(self, **kw: Any) -> dict:
        import asyncio

        return asyncio.run(self.run(**kw))

    def save(self, path: Any, *extra: Any) -> dict:
        return {}


class _ProgressTracker:
    """Simple progress tracker that records callbacks."""

    def __init__(self) -> None:
        self.started: list[str] = []
        self.statuses: list[tuple[str, str, float]] = []
        self.ended: list[tuple[str, str | None]] = []

    def on_run_start(self, task_id: str) -> None:
        self.started.append(task_id)

    def update_run_status(self, task_id: str, message: str, cost_delta: float = 0.0) -> None:
        self.statuses.append((task_id, message, cost_delta))

    def on_run_end(self, task_id: str, exit_status: str | None) -> None:
        self.ended.append((task_id, exit_status))


def _make_problems(n: int = 2) -> list[ForecastProblem]:
    return [ForecastProblem(task_id=f"q{i}", title=f"Question {i}") for i in range(1, n + 1)]


def test_forecast_result_defaults() -> None:
    r = ForecastResult(task_id="x", title="T")
    assert r.status == "pending"
    assert r.submission is None
    assert r.cost == {"model": 0.0, "search": 0.0, "total": 0.0}
    assert r.trajectory is None


def test_progress_callback_protocol() -> None:
    tracker = _ProgressTracker()
    assert isinstance(tracker, BatchProgressCallback)


def test_load_config_defaults() -> None:
    cfg = _load_config(None)
    assert "model" in cfg
    assert "search" in cfg
    assert "agent" in cfg


def test_load_config_dict_merge() -> None:
    cfg = _load_config({"model": {"model_name": "test/model"}})
    assert cfg["model"]["model_name"] == "test/model"
    # Should still have defaults for other sections
    assert "search" in cfg


@patch("miniprophet.tools.search.get_search_backend")
@patch("miniprophet.models.get_model")
def test_batch_forecast_with_agent_class(
    mock_get_model: MagicMock, mock_get_search: MagicMock
) -> None:
    mock_get_model.return_value = MagicMock()
    mock_get_search.return_value = MagicMock()

    problems = _make_problems(2)
    results = batch_forecast_sync(
        problems,
        config={"model": {"model_name": "test/m"}},
        agent_class=_SimpleAgent,
        workers=1,
        timeout_seconds=10.0,
    )

    assert len(results) == 2
    assert results[0].task_id == "q1"
    assert results[1].task_id == "q2"
    assert results[0].status == "submitted"
    assert results[0].submission is not None
    assert results[0].cost["total"] == pytest.approx(0.15)


@patch("miniprophet.tools.search.get_search_backend")
@patch("miniprophet.models.get_model")
def test_batch_forecast_with_progress_callback(
    mock_get_model: MagicMock, mock_get_search: MagicMock
) -> None:
    mock_get_model.return_value = MagicMock()
    mock_get_search.return_value = MagicMock()

    tracker = _ProgressTracker()
    problems = _make_problems(1)
    results = batch_forecast_sync(
        problems,
        config={"model": {"model_name": "test/m"}},
        agent_class=_SimpleAgent,
        on_progress=tracker,
        timeout_seconds=10.0,
    )

    assert len(results) == 1
    assert "q1" in tracker.started
    assert len(tracker.ended) == 1
    assert tracker.ended[0] == ("q1", "submitted")


@patch("miniprophet.tools.search.get_search_backend")
@patch("miniprophet.models.get_model")
def test_batch_forecast_preserves_order(
    mock_get_model: MagicMock, mock_get_search: MagicMock
) -> None:
    mock_get_model.return_value = MagicMock()
    mock_get_search.return_value = MagicMock()

    problems = _make_problems(5)
    results = batch_forecast_sync(
        problems,
        config={"model": {"model_name": "test/m"}},
        agent_class=_SimpleAgent,
        workers=2,
        timeout_seconds=10.0,
    )

    assert [r.task_id for r in results] == ["q1", "q2", "q3", "q4", "q5"]


class _FailAgent:
    """Agent that raises an error."""

    def __init__(self, model: Any, env: Any, **kwargs: Any) -> None:
        self.model_cost = 0.0
        self.search_cost = 0.0
        self.total_cost = 0.0
        self.n_calls = 0

    async def step(self) -> list[dict]:
        return []

    async def run(self, **kw: Any) -> dict:
        raise RuntimeError("Something went wrong")


@patch("miniprophet.tools.search.get_search_backend")
@patch("miniprophet.models.get_model")
def test_batch_forecast_handles_agent_errors(
    mock_get_model: MagicMock, mock_get_search: MagicMock
) -> None:
    mock_get_model.return_value = MagicMock()
    mock_get_search.return_value = MagicMock()

    problems = _make_problems(1)
    results = batch_forecast_sync(
        problems,
        config={"model": {"model_name": "test/m"}},
        agent_class=_FailAgent,
        timeout_seconds=10.0,
    )

    assert len(results) == 1
    assert results[0].status == "RuntimeError"
    assert results[0].error == "Something went wrong"


@patch("miniprophet.tools.search.get_search_backend")
@patch("miniprophet.models.get_model")
def test_batch_forecast_cost_limit(mock_get_model: MagicMock, mock_get_search: MagicMock) -> None:
    mock_get_model.return_value = MagicMock()
    mock_get_search.return_value = MagicMock()

    # First run costs 0.15, so second should be skipped with a 0.10 limit
    problems = _make_problems(2)
    results = batch_forecast_sync(
        problems,
        config={"model": {"model_name": "test/m"}},
        agent_class=_SimpleAgent,
        workers=1,
        max_total_cost=0.10,
        timeout_seconds=10.0,
    )

    assert len(results) == 2
    # First runs fine, second hits cost limit
    assert results[0].status == "submitted"
    assert results[1].status == "skipped_cost_limit"


class TestIsRateLimitError:
    @pytest.mark.parametrize(
        "exc,expected",
        [
            (SearchRateLimitError("limit"), True),
            (Exception("got 429 from server"), True),
            (Exception("rate limit exceeded"), True),
            (Exception("something else"), False),
            (RuntimeError("timeout"), False),
        ],
    )
    def test_detection(self, exc: Exception, expected: bool) -> None:
        assert _is_rate_limit_error(exc) == expected


class TestIsAuthError:
    @pytest.mark.parametrize(
        "exc,expected",
        [
            (type("AuthenticationError", (Exception,), {})("fail"), True),
            (Exception("authentication failed"), True),
            (Exception("unauthorized access"), True),
            (Exception("invalid api key"), True),
            (Exception("HTTP 401"), True),
            (Exception("status code 401"), True),
            (Exception("something else"), False),
        ],
    )
    def test_detection(self, exc: Exception, expected: bool) -> None:
        assert _is_auth_error(exc) == expected


def test_load_config_invalid_type_raises() -> None:
    with pytest.raises(TypeError, match="config must be"):
        _load_config(42)  # type: ignore[arg-type]


@patch("miniprophet.tools.search.get_search_backend")
def test_batch_forecast_with_prebuilt_model(mock_get_search: MagicMock) -> None:
    mock_get_search.return_value = MagicMock()

    class FakeModel:
        class config:
            model_name = "fake"

        async def query(self, *a, **kw):
            return {}

        def format_message(self, **kw):
            return kw

        def format_observation_messages(self, msg, out):
            return []

        def serialize(self):
            return {"info": {"config": {"model": {}}}}

    problems = _make_problems(1)
    results = batch_forecast_sync(
        problems,
        config={"model": {"model_name": "test/m"}},
        agent_class=_SimpleAgent,
        model=FakeModel(),
        timeout_seconds=10.0,
    )
    assert len(results) == 1
    assert results[0].status == "submitted"
