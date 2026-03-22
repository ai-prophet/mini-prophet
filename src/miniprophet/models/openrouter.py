"""OpenRouter API model implementation for mini-prophet."""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Literal

import requests
from pydantic import BaseModel

from miniprophet.models import GLOBAL_MODEL_STATS
from miniprophet.models.retry import retry
from miniprophet.models.utils import (
    format_observation_messages,
    parse_single_action,
)

logger = logging.getLogger("miniprophet.models.openrouter")


class OpenRouterModelConfig(BaseModel):
    model_name: str
    model_kwargs: dict[str, Any] = {}
    format_error_template: str = "Error: {error}. Please make a valid tool call."
    observation_template: str = "{output}"
    cost_tracking: Literal["default", "ignore_errors"] = os.getenv(
        "MINIPROPHET_COST_TRACKING", "default"
    )  # type: ignore


class OpenRouterAPIError(Exception):
    pass


class OpenRouterAuthenticationError(Exception):
    pass


class OpenRouterRateLimitError(Exception):
    pass


class OpenRouterModel:
    abort_exceptions: list[type[Exception]] = [OpenRouterAuthenticationError, KeyboardInterrupt]  # type: ignore

    def __init__(self, **kwargs: Any) -> None:
        self.config = OpenRouterModelConfig(**kwargs)
        self._api_url = "https://openrouter.ai/api/v1/chat/completions"
        self._api_key = os.getenv("OPENROUTER_API_KEY", "")

    # ------------------------------------------------------------------
    # Async core query
    # ------------------------------------------------------------------

    async def query(self, messages: list[dict], tools: list[dict]) -> dict:
        prepared = [{k: v for k, v in msg.items() if k != "extra"} for msg in messages]
        async for attempt in retry(logger=logger, abort_exceptions=self.abort_exceptions):
            with attempt:
                response = await self._query(prepared, tools)

        cost_info = self._calculate_cost(response)
        GLOBAL_MODEL_STATS.add(cost_info["cost"])
        usage_info = self._extract_usage(response)
        GLOBAL_MODEL_STATS.add_tokens(
            usage_info.get("prompt_tokens", 0) or 0,
            usage_info.get("cached_tokens"),
        )

        message = dict(response["choices"][0]["message"])
        message["extra"] = {
            "actions": self._parse_actions(response),
            "response": response,
            **cost_info,
            **usage_info,
            "timestamp": time.time(),
        }
        return message

    async def _query(self, messages: list[dict], tools: list[dict]) -> dict:
        import httpx

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "tools": tools,
            "usage": {"include": True},
            **self.config.model_kwargs,
        }

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    self._api_url,
                    headers=headers,
                    json=payload,
                    timeout=120,
                )
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            if status_code == 401:
                raise OpenRouterAuthenticationError(
                    "Authentication failed. Check OPENROUTER_API_KEY."
                )
            if status_code == 429:
                raise OpenRouterRateLimitError("Rate limit exceeded")
            raise OpenRouterAPIError(f"HTTP {status_code}: {exc.response.text[:300]}")
        except httpx.RequestError as exc:
            raise OpenRouterAPIError(f"Request failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _prepare_messages(self, messages: list[dict]) -> list[dict]:
        """Strip 'extra' keys before sending to the API."""
        return [{k: v for k, v in msg.items() if k != "extra"} for msg in messages]

    def _calculate_cost(self, response: dict) -> dict[str, float]:
        usage = response.get("usage", {})
        cost = usage.get("cost", 0.0)
        if cost is None:
            cost = 0.0
        if (
            cost <= 0.0
            and "free" not in self.config.model_name
            and self.config.cost_tracking != "ignore_errors"
        ):
            raise RuntimeError(
                f"No valid cost info from OpenRouter for {self.config.model_name}. "
                f"Usage: {usage}. Set cost_tracking='ignore_errors' to suppress."
            )
        return {"cost": cost}

    def _extract_usage(self, response: dict) -> dict[str, int | None]:
        """Extract token usage from the OpenRouter response."""
        usage = response.get("usage", {})
        result: dict[str, int | None] = {
            "prompt_tokens": usage.get("prompt_tokens", 0) or 0,
            "completion_tokens": usage.get("completion_tokens", 0) or 0,
            "total_tokens": usage.get("total_tokens", 0) or 0,
        }
        details = usage.get("prompt_tokens_details") or {}
        result["cached_tokens"] = details.get("cached_tokens")
        # OpenRouter uses "cache_write_tokens"; normalize to "cache_creation_tokens"
        result["cache_creation_tokens"] = details.get("cache_write_tokens")
        return result

    def get_max_context_tokens(self) -> int | None:
        """Return the max input token limit via litellm model info, or via OpenRouter API."""
        # Try litellm first (works for many standard models)
        try:
            import litellm

            info = litellm.get_model_info(f"openrouter/{self.config.model_name}")
            max_input = info.get("max_input_tokens")
            if max_input:
                return max_input
        except Exception:
            pass
        # Fallback: query OpenRouter models API
        try:
            resp = requests.get(
                f"https://openrouter.ai/api/v1/models/{self.config.model_name}",
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=10,
            )
            if resp.ok:
                data = resp.json().get("data", resp.json())
                return data.get("context_length")
        except Exception:
            pass
        return None

    def _parse_actions(self, response: dict) -> list[dict]:
        """Parse tool calls from the API response. Raises FormatError on problems."""
        tool_calls = response["choices"][0]["message"].get("tool_calls") or []
        return parse_single_action(
            tool_calls,
            self.config.format_error_template,
            lambda tc: {
                "name": tc.get("function", {}).get("name", ""),
                "arguments": tc.get("function", {}).get("arguments", "{}"),
                "tool_call_id": tc.get("id"),
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
