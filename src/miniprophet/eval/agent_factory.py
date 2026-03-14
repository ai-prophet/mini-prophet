"""Agent factory for prophet eval."""

from __future__ import annotations

import importlib
import inspect
from typing import Any

from miniprophet import ContextManager, Environment, Model
from miniprophet.eval.agent_runtime import (
    EvalBatchAgentWrapper,
    RateLimitCoordinator,
)


class EvalAgentFactory:
    """Construct eval-capable agents from built-ins or import paths."""

    DEFAULT_AGENT_NAME = "default"

    @classmethod
    def _import_agent_class(cls, import_path: str) -> type:
        if ":" not in import_path:
            raise ValueError("Agent import path must be in format 'module.path:ClassName'")

        module_path, class_name = import_path.split(":", 1)
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            raise ValueError(f"Failed to import module '{module_path}': {exc}") from exc

        try:
            agent_cls = getattr(module, class_name)
        except AttributeError as exc:
            raise ValueError(f"Module '{module_path}' has no class '{class_name}'") from exc

        return agent_cls

    @classmethod
    def _resolve_agent_class(
        cls,
        *,
        agent_name: str | None,
        agent_class: type | None,
    ) -> type:
        """Resolve to a concrete agent class from the various input forms."""
        if agent_class is not None:
            return agent_class

        # Built-in default agent
        resolved_name = (agent_name or cls.DEFAULT_AGENT_NAME).strip().lower()
        if resolved_name != cls.DEFAULT_AGENT_NAME:
            raise ValueError(
                f"Unknown built-in eval agent '{resolved_name}'. "
                f"Only '{cls.DEFAULT_AGENT_NAME}' is currently supported."
            )

        from miniprophet.agent.default import DefaultForecastAgent

        return DefaultForecastAgent

    @classmethod
    def create(
        cls,
        *,
        model: Model,
        env: Environment,
        context_manager: ContextManager | None,
        agent_name: str | None = None,
        agent_class: type | None = None,
        agent_kwargs: dict[str, Any],
        task_id: str,
        coordinator: RateLimitCoordinator | None = None,
        progress_manager: Any | None = None,
    ) -> EvalBatchAgentWrapper:
        agent_cls = cls._resolve_agent_class(
            agent_name=agent_name,
            agent_class=agent_class,
        )

        init_kwargs = dict(agent_kwargs)
        signature = inspect.signature(agent_cls)
        if "context_manager" in signature.parameters:
            init_kwargs["context_manager"] = context_manager
        if "cancel_event" in signature.parameters:
            init_kwargs["cancel_event"] = cancel_event

        try:
            agent = agent_cls(model=model, env=env, **init_kwargs)
        except TypeError as exc:
            raise ValueError(
                "Agent constructor is incompatible. "
                "Expected at least (model=..., env=..., **kwargs)."
            ) from exc

        return EvalBatchAgentWrapper(
            agent=agent,
            task_id=task_id,
            coordinator=coordinator,
            progress_manager=progress_manager,
        )
