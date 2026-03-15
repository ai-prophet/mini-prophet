"""CLI eval orchestration — wraps the public batch API with Rich display and disk I/O."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from miniprophet.eval.agent_factory import EvalAgentFactory
from miniprophet.eval.agent_runtime import RateLimitCoordinator
from miniprophet.eval.batch import (
    MAX_RETRIES,
    _is_auth_error,
    _is_rate_limit_error,
)
from miniprophet.eval.progress import EvalProgressManager
from miniprophet.eval.types import ForecastProblem
from miniprophet.exceptions import (
    BatchFatalError,
    SearchAuthError,
)

logger = logging.getLogger("miniprophet.eval")


@dataclass
class RunResult:
    """Result summary for one eval run."""

    task_id: str
    title: str
    status: str = "pending"
    cost: dict[str, float] = field(
        default_factory=lambda: {"model": 0.0, "search": 0.0, "total": 0.0}
    )
    submission: dict[str, float] | None = None
    evaluation: dict[str, float] | None = None
    error: str | None = None
    output_dir: str = ""

    @classmethod
    def from_dict(cls, payload: dict) -> RunResult:
        cost = payload.get("cost", {}) if isinstance(payload.get("cost", {}), dict) else {}
        return cls(
            task_id=str(payload.get("task_id", "")),
            title=str(payload.get("title", "")),
            status=str(payload.get("status", "pending")),
            cost={
                "model": float(cost.get("model", 0.0) or 0.0),
                "search": float(cost.get("search", 0.0) or 0.0),
                "total": float(cost.get("total", 0.0) or 0.0),
            },
            submission=payload.get("submission"),
            evaluation=payload.get("evaluation"),
            error=payload.get("error"),
            output_dir=str(payload.get("output_dir", "")),
        )

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "title": self.title,
            "status": self.status,
            "cost": self.cost,
            "submission": self.submission,
            "evaluation": self.evaluation,
            "error": self.error,
            "output_dir": self.output_dir,
        }


@dataclass
class EvalRunArgs:
    """Runtime arguments for an eval execution."""

    output_dir: Path
    config: dict
    workers: int = 1
    max_cost: float = 0.0
    timeout_seconds: float = 180.0
    initial_results: dict[str, RunResult] | None = None
    initial_total_cost: float = 0.0
    search_date_before: str | None = None
    search_date_after: str | None = None
    dataset_info: dict[str, Any] = field(default_factory=dict)
    agent_name: str | None = None
    agent_class: type | None = None
    agent_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalRunState:
    """Shared mutable state for workers in a single eval execution."""

    coordinator: RateLimitCoordinator
    progress: EvalProgressManager
    summary_lock: asyncio.Lock
    summary_path: Path
    results: dict[str, RunResult]
    total_cost_ref: list[float]
    fatal_event: asyncio.Event
    model: Any
    search_backend: Any
    fatal_error: str | None = None


def load_existing_summary(path: Path) -> tuple[dict[str, RunResult], float]:
    """Load existing summary.json into RunResult mapping and total_cost."""
    if not path.exists():
        return {}, 0.0

    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid summary JSON at {path}: {exc}") from exc

    runs = data.get("runs", [])
    if not isinstance(runs, list):
        raise ValueError(f"Invalid summary format at {path}: 'runs' must be a list")

    results: dict[str, RunResult] = {}
    for item in runs:
        if not isinstance(item, dict):
            continue
        run = RunResult.from_dict(item)
        if run.task_id:
            results[run.task_id] = run

    total_cost = float(data.get("total_cost", 0.0) or 0.0)
    return results, total_cost


def _write_summary(args: EvalRunArgs, state: EvalRunState) -> None:
    summary = {
        "eval_config": {k: v for k, v in args.config.items() if k != "agent"},
        "eval": {
            **args.dataset_info,
            "agent_name": args.agent_name or "default",
        },
        "total_cost": state.total_cost_ref[0],
        "runs": [r.to_dict() for r in state.results.values()],
    }
    state.summary_path.parent.mkdir(parents=True, exist_ok=True)
    state.summary_path.write_text(json.dumps(summary, indent=2))


async def process_problem(
    problem: ForecastProblem,
    args: EvalRunArgs,
    state: EvalRunState,
) -> bool:
    """Process a single forecasting problem. Returns True if complete/permanent failure."""
    from miniprophet.agent.context import get_context_manager
    from miniprophet.environment.forecast_env import ForecastEnvironment, create_default_tools
    from miniprophet.environment.source_board import SourceBoard

    task_id = problem.task_id
    run_dir = args.output_dir / "runs" / task_id
    result = state.results.setdefault(task_id, RunResult(task_id=task_id, title=problem.title))
    result.output_dir = f"runs/{task_id}"

    state.progress.on_run_start(task_id)

    agent: Any = None
    timed_out = False

    try:
        if args.max_cost > 0 and state.total_cost_ref[0] >= args.max_cost:
            result.status = "skipped_cost_limit"
            result.error = f"Total eval cost limit (${args.max_cost:.2f}) reached."
            state.progress.on_run_end(task_id, result.status)
            return True

        search_cfg = args.config.get("search", {})
        agent_cfg = args.config.get("agent", {})
        agent_search_limit = int(agent_cfg.get("search_limit", 10) or 10)
        board = SourceBoard()
        tools = create_default_tools(
            search_tool=state.search_backend,
            outcomes=problem.outcomes,
            board=board,
            search_limit=agent_search_limit,
            search_results_limit=search_cfg.get("search_results_limit", 5),
            max_source_display_chars=search_cfg.get("max_source_display_chars", 2000),
        )
        env = ForecastEnvironment(tools, board=board)

        cm_cfg = args.config.get("context_manager", {})
        ctx_mgr = get_context_manager(cm_cfg)

        agent_kwargs = dict(args.agent_kwargs)
        if args.agent_class is None:
            agent_kwargs = {**agent_cfg, **agent_kwargs}

        agent = EvalAgentFactory.create(
            model=state.model,
            env=env,
            context_manager=ctx_mgr,
            agent_name=args.agent_name,
            agent_class=args.agent_class,
            agent_kwargs=agent_kwargs,
            task_id=task_id,
            coordinator=state.coordinator,
            progress_manager=state.progress,
        )

        runtime_kwargs = {
            "search_date_before": args.search_date_before or problem.predict_by,
            "search_date_after": args.search_date_after,
        }

        forecast = await asyncio.wait_for(
            agent.run(
                title=problem.title,
                outcomes=problem.outcomes,
                ground_truth=problem.ground_truth,
                **runtime_kwargs,
            ),
            timeout=args.timeout_seconds if args.timeout_seconds > 0 else None,
        )

        result.status = forecast.get("exit_status", "unknown")
        result.submission = forecast.get("submission") or None
        result.evaluation = forecast.get("evaluation") or None
        result.cost = {
            "model": float(getattr(agent, "model_cost", 0.0) or 0.0),
            "search": float(getattr(agent, "search_cost", 0.0) or 0.0),
            "total": float(getattr(agent, "total_cost", 0.0) or 0.0),
        }

        async with state.summary_lock:
            state.total_cost_ref[0] += result.cost["total"]

        state.progress.on_run_end(task_id, result.status)
        return True

    except SearchAuthError as exc:
        result.status = "auth_error"
        result.error = str(exc)
        state.progress.on_run_end(task_id, result.status)
        raise BatchFatalError(f"Run {task_id} failed with auth error: {exc}") from exc

    except TimeoutError:
        timed_out = True
        result.status = "BatchRunTimeoutError"
        result.error = f"Run timed out after {args.timeout_seconds:.1f}s"
        state.progress.on_run_end(task_id, result.status)
        return True

    except Exception as exc:
        if _is_rate_limit_error(exc):
            await state.coordinator.signal_rate_limit()
            if problem.retries < MAX_RETRIES:
                problem.retries += 1
                logger.warning(
                    "Rate limit for %s (attempt %d/%d) -- will retry.",
                    task_id,
                    problem.retries,
                    MAX_RETRIES,
                )
                state.progress.on_run_end(task_id, f"rate_limited (retry {problem.retries})")
                return False

        if _is_auth_error(exc):
            result.status = "auth_error"
            result.error = str(exc)
            state.progress.on_run_end(task_id, result.status)
            raise BatchFatalError(f"Run {task_id} failed with auth error: {exc}") from exc

        result.status = type(exc).__name__
        result.error = str(exc)
        logger.error("Run %s failed: %s", task_id, exc, exc_info=True)
        state.progress.on_run_end(task_id, result.status)
        return True

    finally:
        if agent is not None and not timed_out:
            try:
                agent.save(run_dir)
            except Exception:
                logger.error("Failed to save trajectory for %s", task_id, exc_info=True)
        _write_summary(args, state)


async def run_eval(
    problems: list[ForecastProblem],
    args: EvalRunArgs,
) -> dict[str, RunResult]:
    """Execute eval with async parallel workers."""
    from rich.live import Live

    from miniprophet.models import get_model
    from miniprophet.tools.search import get_search_backend

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = get_model(config=args.config.get("model", {}))
    search_backend = get_search_backend(search_cfg=args.config.get("search", {}))
    state = EvalRunState(
        coordinator=RateLimitCoordinator(),
        progress=EvalProgressManager(len(problems)),
        summary_lock=asyncio.Lock(),
        summary_path=args.output_dir / "summary.json",
        results=dict(args.initial_results or {}),
        total_cost_ref=[float(args.initial_total_cost)],
        fatal_event=asyncio.Event(),
        model=model,
        search_backend=search_backend,
    )

    semaphore = asyncio.Semaphore(args.workers)

    async def run_one(problem: ForecastProblem) -> None:
        async with semaphore:
            for _attempt in range(MAX_RETRIES + 1):
                if state.fatal_event.is_set():
                    return
                try:
                    done = await process_problem(problem, args, state)
                    if done:
                        return
                except BatchFatalError as exc:
                    state.fatal_error = str(exc)
                    state.fatal_event.set()
                    return
                except Exception as exc:
                    result = state.results.setdefault(
                        problem.task_id,
                        RunResult(task_id=problem.task_id, title=problem.title),
                    )
                    result.status = type(exc).__name__
                    result.error = str(exc)
                    logger.error(
                        "Unhandled worker exception for %s",
                        problem.task_id,
                        exc_info=True,
                    )
                    state.progress.on_run_end(problem.task_id, result.status)
                    _write_summary(args, state)
                    return

    tasks = [asyncio.create_task(run_one(p)) for p in problems]

    with Live(state.progress.render_group, refresh_per_second=4):
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except KeyboardInterrupt:
            logger.info("Eval interrupted by user. Cancelling active runs...")
            for t in tasks:
                t.cancel()

    _write_summary(args, state)

    if state.fatal_event.is_set():
        raise BatchFatalError(state.fatal_error or "Eval terminated due to a fatal error.")

    return state.results


def run_eval_sync(
    problems: list[ForecastProblem],
    args: EvalRunArgs,
) -> dict[str, RunResult]:
    """Execute eval with parallel workers. Sync wrapper."""
    return asyncio.run(run_eval(problems, args))
