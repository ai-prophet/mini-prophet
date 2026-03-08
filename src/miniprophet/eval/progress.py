"""Rich progress display for eval forecasting runs."""

from __future__ import annotations

import collections
import time
from pathlib import Path
from threading import Lock

from rich.console import Group
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table


def _shorten(s: str, max_len: int, *, left: bool = False) -> str:
    if len(s) <= max_len:
        return f"{s:<{max_len}}"
    if left:
        return f"{'...' + s[-max_len + 3 :]:<{max_len}}"
    return f"{s[: max_len - 3] + '...':<{max_len}}"


class EvalProgressManager:
    """Thread-safe progress tracker for eval forecast runs.

    Renders an overall progress bar, a per-instance spinner table,
    and an exit-status summary table via Rich Live.
    """

    def __init__(self, total: int, report_path: Path | None = None) -> None:
        self._lock = Lock()
        self._start_time = time.time()
        self._total = total
        self._total_cost = 0.0
        self._report_path = report_path

        self._instances_by_status: dict[str, list[str]] = collections.defaultdict(list)
        self._spinner_tasks: dict[str, TaskID] = {}

        self._main_bar = Progress(
            SpinnerColumn(spinner_name="dots2"),
            TextColumn("[progress.description]{task.description} (${task.fields[total_cost]})"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            speed_estimate_period=60 * 5,
        )
        self._task_bar = Progress(
            SpinnerColumn(spinner_name="dots2"),
            TextColumn("{task.fields[run_id]}"),
            TextColumn("{task.fields[status]}"),
            TimeElapsedColumn(),
        )

        self._main_task_id = self._main_bar.add_task(
            "[cyan]Eval Progress", total=total, total_cost="0.00"
        )

        self._status_table = Table()
        self.render_group = Group(self._main_bar, self._status_table, self._task_bar)

    @property
    def n_completed(self) -> int:
        return sum(len(v) for v in self._instances_by_status.values())

    def _refresh_status_table(self) -> None:
        t = Table()
        t.add_column("Exit Status")
        t.add_column("Count", justify="right", style="bold cyan")
        t.add_column("Recent runs")
        t.show_header = True
        with self._lock:
            for status, runs in sorted(
                self._instances_by_status.items(), key=lambda x: len(x[1]), reverse=True
            ):
                t.add_row(status, str(len(runs)), _shorten(", ".join(reversed(runs)), 55))
        self.render_group.renderables[1] = t

    def _update_cost(self, cost_delta: float = 0.0) -> None:
        with self._lock:
            self._total_cost += cost_delta
            self._main_bar.update(
                self._main_task_id,
                total_cost=f"{self._total_cost:.2f}",
            )

    def on_run_start(self, task_id: str) -> None:
        with self._lock:
            self._spinner_tasks[task_id] = self._task_bar.add_task(
                description=f"Run {task_id}",
                status="initializing",
                total=None,
                run_id=_shorten(task_id, 25, left=True),
            )

    def update_run_status(self, task_id: str, message: str, cost_delta: float = 0.0) -> None:
        with self._lock:
            if task_id in self._spinner_tasks:
                self._task_bar.update(
                    self._spinner_tasks[task_id],
                    status=_shorten(message, 30),
                    run_id=_shorten(task_id, 25, left=True),
                )
        if cost_delta:
            self._update_cost(cost_delta)

    def on_run_end(self, task_id: str, exit_status: str | None) -> None:
        with self._lock:
            self._instances_by_status[exit_status or "unknown"].append(task_id)
            if task_id in self._spinner_tasks:
                try:
                    self._task_bar.remove_task(self._spinner_tasks[task_id])
                except KeyError:
                    pass
            self._main_bar.update(TaskID(0), advance=1)
        self._refresh_status_table()
        self._update_cost()

    def on_uncaught_exception(self, task_id: str, exc: Exception) -> None:
        self.on_run_end(task_id, f"Uncaught {type(exc).__name__}")


# Backward-compatible internal alias.
BatchProgressManager = EvalProgressManager
