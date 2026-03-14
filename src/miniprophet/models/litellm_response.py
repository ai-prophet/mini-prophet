"""LiteLLM Responses API model implementation for mini-prophet."""

from __future__ import annotations

import logging
from typing import Any

import litellm

from miniprophet.models.litellm import LitellmModel, LitellmModelConfig
from miniprophet.models.responses_utils import (
    action_from_response_function_call,
    build_chat_message_from_response,
    prepare_response_messages,
    prepare_response_tools,
    response_function_calls,
    response_output_items,
)
from miniprophet.models.utils import parse_single_action

logger = logging.getLogger("miniprophet.models.litellm_response")


class LitellmResponseModelConfig(LitellmModelConfig):
    """Responses API config, based on the standard LiteLLM model config."""


class LitellmResponseModel(LitellmModel):
    logger = logger

    def __init__(self, **kwargs: Any) -> None:
        self.config = LitellmResponseModelConfig(**kwargs)

    def _query(self, messages: list[dict], tools: list[dict]):
        return litellm.responses(
            model=self.config.model_name,
            input=messages,
            tools=tools,
            **self.config.model_kwargs,
        )

    async def _aquery(self, messages: list[dict], tools: list[dict]):
        return await litellm.aresponses(
            model=self.config.model_name,
            input=messages,
            tools=tools,
            **self.config.model_kwargs,
        )

    def _prepare_messages(self, messages: list[dict]) -> list[dict]:
        return prepare_response_messages(messages)

    def _prepare_tools(self, tools: list[dict]) -> list[dict]:
        return prepare_response_tools(tools)

    def _parse_actions(self, response) -> list[dict]:
        """Parse tool calls from a Responses API response. Raises FormatError on problems."""
        tool_calls = response_function_calls(response_output_items(response))
        return parse_single_action(
            tool_calls,
            self.config.format_error_template,
            action_from_response_function_call,
        )

    def _build_message(self, response) -> dict:
        """Build an assistant-like message from Responses API output items."""
        return build_chat_message_from_response(response_output_items(response))
