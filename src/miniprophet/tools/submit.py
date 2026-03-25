"""Submit tool: final probabilistic forecast submission."""

from __future__ import annotations

from miniprophet.environment.source_board import SourceBoard
from miniprophet.exceptions import Submitted

SUBMIT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "submit",
        "description": (
            "Submit your final forecast as a single probability P(Yes) between 0 and 1. "
            "If the question is not a valid binary yes/no question, submit probability=0."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "probability": {
                    "type": "number",
                    "description": (
                        "A number between 0 and 1 representing the probability "
                        "that the answer is Yes."
                    ),
                },
            },
            "required": ["probability"],
        },
    },
}


class SubmitTool:
    """Validates and submits the final binary forecast."""

    def __init__(self, board: SourceBoard) -> None:
        self._board = board

    @property
    def name(self) -> str:
        return "submit"

    def get_schema(self) -> dict:
        return SUBMIT_SCHEMA

    async def execute(self, args: dict) -> dict:
        return self._execute_impl(args)

    def _execute_impl(self, args: dict) -> dict:
        probability = args.get("probability")
        if not isinstance(probability, int | float):
            return {
                "output": "Error: 'probability' must be a number between 0 and 1.",
                "error": True,
            }
        if not (0 <= probability <= 1):
            return {
                "output": f"Error: probability must be between 0 and 1, got {probability}.",
                "error": True,
            }

        submission = {"Yes": float(probability), "No": round(1 - float(probability), 10)}
        raise Submitted(
            {
                "role": "exit",
                "content": "Forecast submitted successfully.",
                "extra": {
                    "exit_status": "submitted",
                    "submission": submission,
                    "board": self._board.serialize(),
                },
            }
        )

    def display(self, output: dict) -> None:
        from miniprophet.cli.components.observation import print_observation

        print_observation(output)
