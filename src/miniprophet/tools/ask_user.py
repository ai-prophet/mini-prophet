"""AskUser tool: ask the user a clarifying question (blocking)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

AskUserCallback = Callable[[str], Awaitable[str]]

ASK_USER_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "ask_user",
        "description": (
            "Ask the user a clarifying question about the forecasting problem. "
            "Use this when the question is ambiguous or you need additional context "
            "that cannot be resolved by web search."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to ask the user.",
                },
            },
            "required": ["question"],
        },
    },
}


class AskUserTool:
    """Blocking tool that pauses the agent and waits for user input.

    In batch mode (``callback=None``), returns a canned "no user" response.
    """

    def __init__(self, *, callback: AskUserCallback | None = None) -> None:
        self._callback = callback

    @property
    def name(self) -> str:
        return "ask_user"

    def get_schema(self) -> dict:
        return ASK_USER_SCHEMA

    async def execute(self, args: dict) -> dict:
        question = (args.get("question") or "").strip()
        if not question:
            return {"output": "Error: 'question' must not be empty.", "error": True}

        if self._callback is None:
            return {"output": "(No user available — batch mode)"}

        response = await self._callback(question)
        return {"output": response or "(No response from user)"}

    def display(self, output: dict) -> None:
        from miniprophet.cli.components.observation import print_observation

        print_observation(output)
