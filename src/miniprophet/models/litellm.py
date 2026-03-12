"""LiteLLM model implementation for mini-prophet."""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Literal

import litellm
from pydantic import BaseModel

from miniprophet.models import GLOBAL_MODEL_STATS
from miniprophet.models.retry import retry
from miniprophet.models.utils import (
    format_observation_messages,
    parse_single_action,
)

logger = logging.getLogger("miniprophet.models.litellm")


class LitellmModelConfig(BaseModel):
    model_name: str
    """Model name with provider prefix, e.g. ``openai/gpt-4o``, ``anthropic/claude-sonnet-4-5-20250929``."""
    model_kwargs: dict[str, Any] = {}
    format_error_template: str = "Error: {error}. Please make a valid tool call."
    observation_template: str = "{output}"
    cost_tracking: Literal["default", "ignore_errors"] = os.getenv(
        "MINIPROPHET_COST_TRACKING", "default"
    )  # type: ignore


class LitellmModel:
    logger = logger

    abort_exceptions: list[type[Exception]] = [
        litellm.exceptions.UnsupportedParamsError,
        litellm.exceptions.NotFoundError,
        litellm.exceptions.PermissionDeniedError,
        litellm.exceptions.ContextWindowExceededError,
        litellm.exceptions.AuthenticationError,
        KeyboardInterrupt,
    ]

    def __init__(self, **kwargs: Any) -> None:
        self.config = LitellmModelConfig(**kwargs)

    # ------------------------------------------------------------------
    # Core query
    # ------------------------------------------------------------------

    def query(self, messages: list[dict], tools: list[dict]) -> dict:
        for attempt in retry(logger=self.logger, abort_exceptions=self.abort_exceptions):
            with attempt:
                response = self._query(self._prepare_messages(messages), self._prepare_tools(tools))

        cost_info = self._calculate_cost(response)
        GLOBAL_MODEL_STATS.add(cost_info["cost"])
        usage_info = self._extract_usage(response)

        message = self._build_message(response)
        message["extra"] = {
            "actions": self._parse_actions(response),
            "response": self._dump_response(response),
            **cost_info,
            **usage_info,
            "timestamp": time.time(),
        }
        return message

    def _query(self, messages: list[dict], tools: list[dict]):
        return litellm.completion(
            model=self.config.model_name,
            messages=messages,
            tools=tools,
            **self.config.model_kwargs,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _prepare_messages(self, messages: list[dict]) -> list[dict]:
        """Strip 'extra' keys before sending to the API."""
        return [{k: v for k, v in msg.items() if k != "extra"} for msg in messages]

    def _prepare_tools(self, tools: list[dict]) -> list[dict]:
        return tools

    def _calculate_cost(self, response) -> dict[str, float]:
        try:
            cost = litellm.cost_calculator.completion_cost(response, model=self.config.model_name)
            if cost <= 0.0:
                raise ValueError(f"Cost must be > 0.0, got {cost}")
        except Exception as e:
            cost = 0.0
            if self.config.cost_tracking != "ignore_errors":
                msg = (
                    f"Error calculating cost for model {self.config.model_name}: {e}. "
                    "Set cost_tracking='ignore_errors' in config, or set MINIPROPHET_COST_TRACKING to 'ignore_errors' to suppress."
                )
                self.logger.critical(msg)
                raise RuntimeError(msg) from e
        return {"cost": cost}

    def _extract_usage(self, response) -> dict[str, int]:
        """Extract token usage from the LiteLLM response."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return {}
        return {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
            "total_tokens": getattr(usage, "total_tokens", 0) or 0,
        }

    def get_max_context_tokens(self) -> int | None:
        """Return the max input token limit for the configured model, or None if unknown."""
        try:
            info = litellm.get_model_info(self.config.model_name)
            return info.get("max_input_tokens")
        except Exception:
            self.logger.debug(
                f"Could not retrieve max context tokens for {self.config.model_name}"
            )
            return None

    def _build_message(self, response) -> dict:
        return response.choices[0].message.model_dump()

    def _dump_response(self, response) -> dict[str, Any]:
        if hasattr(response, "model_dump"):
            return response.model_dump()
        if isinstance(response, dict):
            return response
        return dict(response)

    def _parse_actions(self, response) -> list[dict]:
        """Parse tool calls from the response. Raises FormatError on problems."""
        tool_calls = response.choices[0].message.tool_calls or []
        return parse_single_action(
            tool_calls,
            self.config.format_error_template,
            lambda tc: {
                "name": tc.function.name or "",
                "arguments": tc.function.arguments or "{}",
                "tool_call_id": tc.id,
            },
        )

    # ------------------------------------------------------------------
    # Message formatting
    # ------------------------------------------------------------------

    def format_message(self, **kwargs: Any) -> dict:
        return dict(kwargs)

    def format_observation_messages(self, message: dict, outputs: list[dict]) -> list[dict]:
        return format_observation_messages(message, outputs)

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "model": self.config.model_dump(mode="json"),
                    "model_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
            }
        }
