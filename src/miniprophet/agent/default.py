"""DefaultForecastAgent: the core agent loop for mini-prophet."""

from __future__ import annotations

import asyncio
import json
import logging
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict

from pydantic import BaseModel

from miniprophet import ContextManager, Environment, Model, __version__
from miniprophet.agent.trajectory import TrajectoryRecorder
from miniprophet.exceptions import InterruptAgentFlow, LimitsExceeded
from miniprophet.utils.metrics import (
    evaluate_submission,
    normalize_ground_truth,
    validate_ground_truth,
)
from miniprophet.utils.serialize import recursive_merge


class ForecastResult(TypedDict, total=False):
    exit_status: str
    submission: dict[str, float]
    rationale: str
    evaluation: dict[str, float]
    sources: dict


class AgentConfig(BaseModel):
    system_template: str
    instance_template: str
    step_limit: int = 30
    cost_limit: float = 3.0
    search_limit: int = 10
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
        config_class: type = AgentConfig,
        **kwargs,
    ) -> None:
        self.config = config_class(**kwargs)
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
        self.cached_tokens: int | None = None
        self.cache_creation_tokens: int | None = None
        self.total_prompt_tokens = 0
        self.total_cached_tokens = 0
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
    def on_run_start(self, title: str, config: AgentConfig) -> None:
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
    def run_sync(
        self,
        title: str,
        ground_truth: dict[str, int] | None = None,
        **runtime_kwargs,
    ) -> ForecastResult:
        return asyncio.run(self.run(title, ground_truth, **runtime_kwargs))

    async def run(
        self,
        title: str,
        ground_truth: dict[str, int] | None = None,
        **runtime_kwargs,
    ) -> ForecastResult:
        if ground_truth is not None:
            ground_truth = normalize_ground_truth(ground_truth)
            validate_ground_truth(ground_truth)

        self.runtime_kwargs = runtime_kwargs

        current_time = (
            datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S %Z")
            if self.config.show_current_time
            else "Not Available"
        )
        self.messages = []
        self.add_messages(
            self.model.format_message(
                role="system",
                content=self._render(self.config.system_template, title=title),
            ),
            self.model.format_message(
                role="user",
                content=self._render(
                    self.config.instance_template,
                    title=title,
                    current_time=current_time,
                ),
            ),
        )

        self.on_run_start(title, self.config)

        while True:
            try:
                await self.step()
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
            "rationale": last_extra.get("rationale", ""),
            "sources": last_extra.get("sources", {}),
        }

        if ground_truth is not None and result.get("submission"):
            evaluation = evaluate_submission(result["submission"], ground_truth)
            result["evaluation"] = evaluation

        self.on_run_end(result)
        return result

    def _prepare_messages_for_step(self) -> None:
        """Apply context manager if enabled. No board injection — context is append-only."""
        if self.context_manager is not None:
            self.messages = self.context_manager.manage(self.messages, step=self.n_calls)

    async def step(self) -> list[dict]:
        self._prepare_messages_for_step()
        return await self.execute_actions(await self.query())

    async def query(self) -> dict:
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
        input_snapshot = list(self.messages)
        tools = self.env.get_tool_schemas()
        message = await self.model.query(self.messages, tools)
        extra = message.get("extra", {})
        self.model_cost += extra.get("cost", 0.0)
        self.prompt_tokens = extra.get("prompt_tokens", self.prompt_tokens)
        self.completion_tokens = extra.get("completion_tokens", self.completion_tokens)
        self.cached_tokens = extra.get("cached_tokens")
        self.cache_creation_tokens = extra.get("cache_creation_tokens")
        call_prompt = extra.get("prompt_tokens", 0) or 0
        call_cached = extra.get("cached_tokens")
        self.total_prompt_tokens += call_prompt
        if call_cached is not None:
            self.total_cached_tokens += call_cached
        self.add_messages(message)
        self._trajectory.record_step(input_snapshot, message)

        self.on_step_start()
        self.on_model_response(message)
        return message

    async def execute_actions(self, message: dict) -> list[dict]:
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
                outputs.append(await self.env.execute(action, **self.runtime_kwargs))
        for action, output in zip(actions, outputs):
            sc = output.get("search_cost", 0.0)
            if sc:
                self.search_cost += sc
                self.n_searches += 1
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
                "cached_tokens": self.cached_tokens,
                "cache_creation_tokens": self.cache_creation_tokens,
                "total_prompt_tokens": self.total_prompt_tokens,
                "total_cached_tokens": self.total_cached_tokens,
                "cache_hit_rate": (
                    self.total_cached_tokens / self.total_prompt_tokens
                    if self.total_prompt_tokens > 0 and self.total_cached_tokens > 0
                    else None
                ),
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
