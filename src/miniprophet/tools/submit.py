"""Submit tool: final probabilistic forecast submission."""

from __future__ import annotations

from miniprophet.environment.source_board import SourceBoard
from miniprophet.exceptions import Submitted

SUBMIT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "submit",
        "description": (
            "Submit your final probabilistic forecast. Provide a probability "
            "(between 0 and 1) for EVERY listed outcome. The probabilities should "
            "reflect the balance of evidence gathered on your source board."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "probabilities": {
                    "type": "object",
                    "description": (
                        "A JSON object mapping each outcome name (exactly as listed) "
                        "to a probability value between 0 and 1."
                    ),
                },
            },
            "required": ["probabilities"],
        },
    },
}


class SubmitTool:
    """Validates and submits the final forecast."""

    def __init__(self, outcomes: list[str], board: SourceBoard) -> None:
        self._outcomes = outcomes
        self._board = board

    @property
    def name(self) -> str:
        return "submit"

    def get_schema(self) -> dict:
        return SUBMIT_SCHEMA

    async def execute(self, args: dict) -> dict:
        return self._execute_impl(args)

    def _execute_impl(self, args: dict) -> dict:
        probabilities = args.get("probabilities")
        if not isinstance(probabilities, dict):
            return {
                "output": "Error: 'probabilities' must be a JSON object mapping outcomes to values.",
                "error": True,
            }

        errors: list[str] = []
        for outcome in self._outcomes:
            if outcome not in probabilities:
                errors.append(f"Missing probability for outcome: '{outcome}'.")
        for key, val in probabilities.items():
            if key not in self._outcomes:
                errors.append(f"Unknown outcome: '{key}'.")
            elif not isinstance(val, int | float) or not (0 <= val <= 1):
                errors.append(
                    f"Probability for '{key}' must be a number between 0 and 1, got {val}."
                )

        if errors:
            return {"output": "Submission rejected:\n" + "\n".join(errors), "error": True}

        raise Submitted(
            {
                "role": "exit",
                "content": "Forecast submitted successfully.",
                "extra": {
                    "exit_status": "submitted",
                    "submission": probabilities,
                    "board": self._board.serialize(),
                },
            }
        )

    def display(self, output: dict) -> None:
        from miniprophet.cli.components.observation import print_observation

        print_observation(output)
