"""Tests for miniprophet.eval.progress module."""

from __future__ import annotations

from miniprophet.eval.progress import EvalProgressManager, _shorten


class TestShorten:
    def test_no_truncation_needed(self) -> None:
        result = _shorten("hi", 10)
        assert result == "hi        "

    def test_right_truncation(self) -> None:
        result = _shorten("a" * 20, 10)
        assert result.strip().endswith("...")
        assert len(result) == 10

    def test_left_truncation(self) -> None:
        result = _shorten("a" * 20, 10, left=True)
        assert result.strip().startswith("...")
        assert len(result) == 10


class TestEvalProgressManager:
    def test_init_and_n_completed(self) -> None:
        pm = EvalProgressManager(total=5)
        assert pm.n_completed == 0

    def test_run_lifecycle(self) -> None:
        pm = EvalProgressManager(total=2)
        pm.on_run_start("task_1")
        assert "task_1" in pm._spinner_tasks

        pm.update_run_status("task_1", "step 1", cost_delta=0.05)
        assert pm._total_cost > 0

        pm.on_run_end("task_1", "submitted")
        assert pm.n_completed == 1
        assert "submitted" in pm._instances_by_status

    def test_on_uncaught_exception(self) -> None:
        pm = EvalProgressManager(total=1)
        pm.on_run_start("task_1")
        pm.on_uncaught_exception("task_1", ValueError("boom"))
        assert pm.n_completed == 1
        assert "Uncaught ValueError" in pm._instances_by_status

    def test_update_cost(self) -> None:
        pm = EvalProgressManager(total=1)
        pm._update_cost(0.5)
        assert pm._total_cost == 0.5
        pm._update_cost(0.3)
        assert pm._total_cost == 0.8
