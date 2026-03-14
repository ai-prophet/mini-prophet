"""DefaultForecastAgent: the core agent loop for mini-prophet."""

from __future__ import annotations

import json
import logging
import threading
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict

from pydantic import BaseModel

from miniprophet import ContextManager, Environment, Model, __version__
from miniprophet.agent.trajectory import TrajectoryRecorder
from miniprophet.exceptions import BatchRunTimeoutError, InterruptAgentFlow, LimitsExceeded
from miniprophet.utils.metrics import evaluate_submission, validate_ground_truth
from miniprophet.utils.serialize import recursive_merge


class ForecastResult(TypedDict, total=False):
    exit_status: str
    submission: dict[str, float]
    evaluation: dict[str, float]
    board: list[dict]


class AgentConfig(BaseModel):
    system_template: str
    instance_template: str
    step_limit: int = 30
    cost_limit: float = 3.0
    search_limit: int = 10
    max_outcomes: int = 20
    context_window: int = 6
    output_path: Path | None = None
    show_current_time: bool = False
    enable_grace_period: bool = False
    grace_period_prompt: str = ""
    grace_period_extra_turns: int = 3


class DefaultForecastAgent:
    def __init__(
        self,
        model: Model,
        env: Environment,
        *,
        context_manager: ContextManager | None = None,
        cancel_event: threading.Event | None = None,
        config_class: type = AgentConfig,
        **kwargs,
    ) -> None:
        self.config = config_class(**kwargs)
        self._cancel_event = cancel_event
        self.messages: list[dict] = []
        self.model = model
        self.env = env
        self.context_manager = context_manager
        self.logger = logging.getLogger("miniprophet.agent")
        self.model_cost = 0.0
        self.search_cost = 0.0
        self.n_searches = 0
        self.n_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.max_context_tokens: int | None = None
        self._in_grace_period = False
        self._grace_period_turns = 0
        self._trajectory = TrajectoryRecorder()

        # Resolve max context length from model (best-effort)
        if hasattr(self.model, "get_max_context_tokens"):
            try:
                self.max_context_tokens = self.model.get_max_context_tokens()
            except Exception:
                pass

    @property
    def total_cost(self) -> float:
        return self.model_cost + self.search_cost

    def _render(self, template: str, **extra_vars) -> str:
        return template.format_map(extra_vars)

    def add_messages(self, *messages: dict) -> list[dict]:
        self.messages.extend(messages)
        return list(messages)

    def handle_uncaught_exception(self, exc: Exception) -> list[dict]:
        return self.add_messages(
            self.model.format_message(
                role="exit",
                content=str(exc),
                extra={
                    "exit_status": type(exc).__name__,
                    "submission": "",
                    "exception_str": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
        )

    # Hook methods (no-ops; overridden by subclasses like CliForecastAgent)
    def on_run_start(self, title: str, outcomes: str, config: AgentConfig) -> None:
        pass

    def on_step_start(self) -> None:
        pass

    def on_model_response(self, message: dict) -> None:
        pass

    def on_observation(self, action: dict, output: dict) -> None:
        pass

    def on_run_end(self, result: ForecastResult) -> None:
        pass

    # Core loop
    def run(
        self,
        title: str,
        outcomes: list[str],
        ground_truth: dict[str, int] | None = None,
        **runtime_kwargs,
    ) -> ForecastResult:
        if len(outcomes) > self.config.max_outcomes:
            raise ValueError(
                f"Too many outcomes ({len(outcomes)} > {self.config.max_outcomes}). "
                "Increase `max_outcomes` in config if intentional."
            )
        if ground_truth is not None:
            validate_ground_truth(outcomes, ground_truth)

        self.runtime_kwargs = runtime_kwargs

        outcomes_formatted = ", ".join(outcomes)
        current_time = (
            datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S %Z")
            if self.config.show_current_time
            else "Not Available"
        )
        self.messages = []
        self.add_messages(
            self.model.format_message(
                role="system",
                content=self._render(
                    self.config.system_template, title=title, outcomes_formatted=outcomes_formatted
                ),
            ),
            self.model.format_message(
                role="user",
                content=self._render(
                    self.config.instance_template,
                    title=title,
                    outcomes_formatted=outcomes_formatted,
                    current_time=current_time,
                ),
            ),
        )

        self.on_run_start(title, outcomes_formatted, self.config)

        while True:
            try:
                self.step()
            except InterruptAgentFlow as exc:
                self.add_messages(*exc.messages)
            except Exception as exc:
                self.handle_uncaught_exception(exc)
                raise
            finally:
                self.save(self.config.output_path)
            if self.messages[-1].get("role") == "exit":
                break

        last_extra = self.messages[-1].get("extra", {})

        result: ForecastResult = {
            "exit_status": last_extra.get("exit_status", "unknown"),
            "submission": last_extra.get("submission", {}),
            "board": last_extra.get("board", []),
        }

        if ground_truth is not None and result.get("submission"):
            evaluation = evaluate_submission(result["submission"], ground_truth)
            result["evaluation"] = evaluation

        self.on_run_end(result)
        return result

    def _prepare_messages_for_step(self) -> None:
        """Strip stale board state, apply context manager, inject fresh board state."""
        self.messages = [m for m in self.messages if not m.get("extra", {}).get("is_board_state")]

        if self.context_manager is not None:
            self.messages = self.context_manager.manage(self.messages, step=self.n_calls)

        # Inject fresh board state as invariant at position 2 (after system + user)
        if hasattr(self.env, "board"):
            self.messages.insert(
                2,
                {
                    "role": "system",
                    "content": self.env.board.render(),  # type: ignore
                    "extra": {"is_board_state": True},
                },
            )

    def step(self) -> list[dict]:
        if self._cancel_event is not None and self._cancel_event.is_set():
            raise BatchRunTimeoutError("Run cancelled (timeout).")
        self._prepare_messages_for_step()
        return self.execute_actions(self.query())

    def query(self) -> dict:
        step_limit_hit = 0 < self.config.step_limit <= self.n_calls
        cost_limit_hit = 0 < self.config.cost_limit <= self.total_cost

        if step_limit_hit or cost_limit_hit:
            if self.config.enable_grace_period and not self._in_grace_period:
                self._in_grace_period = True
                self._grace_period_turns = 0

            if self._in_grace_period:
                if self._grace_period_turns >= self.config.grace_period_extra_turns:
                    raise LimitsExceeded(
                        {
                            "role": "exit",
                            "content": "Grace period exhausted.",
                            "extra": {"exit_status": "LimitsExceeded", "submission": ""},
                        }
                    )
                self.add_messages(
                    self.model.format_message(
                        role="user",
                        content=self.config.grace_period_prompt,
                    )
                )
                self._grace_period_turns += 1
            else:
                limit_type = "Step" if step_limit_hit else "Cost"
                raise LimitsExceeded(
                    {
                        "role": "exit",
                        "content": f"{limit_type} limit exceeded.",
                        "extra": {"exit_status": "LimitsExceeded", "submission": ""},
                    }
                )

        self.n_calls += 1
        # Snapshot the messages the model will actually see (post-truncation, post-board inject)
        input_snapshot = list(self.messages)
        tools = self.env.get_tool_schemas()
        message = self.model.query(self.messages, tools)
        extra = message.get("extra", {})
        self.model_cost += extra.get("cost", 0.0)
        self.prompt_tokens = extra.get("prompt_tokens", self.prompt_tokens)
        self.completion_tokens = extra.get("completion_tokens", self.completion_tokens)
        self.add_messages(message)
        self._trajectory.record_step(input_snapshot, message)

        self.on_step_start()
        self.on_model_response(message)
        return message

    def execute_actions(self, message: dict) -> list[dict]:
        actions = message.get("extra", {}).get("actions", [])
        outputs: list[dict] = []
        for action in actions:
            if self._in_grace_period and action.get("name") != "submit":
                outputs.append(
                    {
                        "output": self.config.grace_period_prompt,
                        "error": True,
                    }
                )
            else:
                outputs.append(self.env.execute(action, **self.runtime_kwargs))
        for action, output in zip(actions, outputs):
            sc = output.get("search_cost", 0.0)
            if sc:
                self.search_cost += sc
                self.n_searches += 1
            if action.get("name") == "search" and self.context_manager is not None:
                raw = action.get("arguments", "{}")
                args = json.loads(raw) if isinstance(raw, str) else raw
                query = args.get("query", "")
                if query and hasattr(self.context_manager, "record_query"):
                    self.context_manager.record_query(query)  # type: ignore
            self.on_observation(action, output)
        return self.add_messages(*self.model.format_observation_messages(message, outputs))

    # Serialization / save
    def serialize_info(self, *extra_dicts: dict) -> dict:
        """Serialize run metadata (config, costs, status, submission, evaluation)."""
        last_message = self.messages[-1] if self.messages else {}
        last_extra = last_message.get("extra", {})
        agent_data = {
            "cost_stats": {
                "model_cost": self.model_cost,
                "search_cost": self.search_cost,
                "total_cost": self.total_cost,
                "n_api_calls": self.n_calls,
                "n_searches": self.n_searches,
            },
            "token_usage": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "max_context_tokens": self.max_context_tokens,
            },
            "config": {
                "agent": self.config.model_dump(mode="json"),
                "agent_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
            },
            "version": __version__,
            "exit_status": last_extra.get("exit_status", ""),
            "submission": last_extra.get("submission", ""),
            "evaluation": last_extra.get("evaluation", {}),
        }
        model_info = self.model.serialize().get("info", {})
        env_info = self.env.serialize().get("info", {})
        extra_merged = recursive_merge(*extra_dicts) if extra_dicts else {}
        return recursive_merge(agent_data, model_info, env_info, extra_merged)

    def serialize(self, *extra_dicts: dict) -> dict:
        """Serialize run artifacts into a single dict (legacy compat)."""
        res = dict()
        res["info"] = self.serialize_info(*extra_dicts)
        res["trajectory"] = self._trajectory.serialize()
        res["trajectory"]["trajectory_format"] = f"mini-prophet-v{__version__}"
        if hasattr(self.env, "serialize_sources_state"):
            res["sources"] = self.env.serialize_sources_state()
        return res

    def save(self, path: Path | None, *extra_dicts: dict) -> dict:
        """Save run artifacts to a directory (info.json + trajectory.json + sources.json).

        ``path`` is treated as a directory. If None, serialization is
        still performed but nothing is written to disk.
        """
        res = self.serialize(*extra_dicts)
        if path:
            path.mkdir(parents=True, exist_ok=True)
            for key, value in res.items():
                (path / f"{key}.json").write_text(json.dumps(value, indent=2))
        return res
