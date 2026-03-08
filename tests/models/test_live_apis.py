"""Alert: Real API integration tests that cost money.

################################################################################
#                                                                              #
#                         CRITICAL WARNING                                     #
#                                                                              #
#   THIS TEST FILE SHOULD NEVER BE RUN BY AN AI AGENT.                         #
#   IT REQUIRES EXPLICIT HUMAN REQUEST AND SUPERVISION.                        #
#                                                                              #
#   These tests make REAL API calls that:                                      #
#   - Cost real money (API usage fees)                                         #
#   - Require valid API keys for multiple providers                            #
#   - May have rate limits and quotas                                          #
#                                                                              #
#                                                                              #
################################################################################
"""

from __future__ import annotations

import os

import pytest

from miniprophet.models.openrouter import OpenRouterModel

pytestmark = pytest.mark.live_api


def _require_env(*keys: str) -> None:
    missing = [k for k in keys if not os.getenv(k)]
    if missing:
        pytest.skip(f"Missing env vars for live test: {', '.join(missing)}")


def test_openrouter_live_single_tool_call() -> None:
    _require_env("OPENROUTER_API_KEY", "MINIPROPHET_TEST_OPENROUTER_MODEL")

    model = OpenRouterModel(
        model_name=os.environ["MINIPROPHET_TEST_OPENROUTER_MODEL"],
        # Live smoke test should not fail if provider omits cost in rare cases.
        cost_tracking="ignore_errors",
    )
    msg = model.query(
        messages=[
            {
                "role": "user",
                "content": (
                    "Return exactly one tool call to search for query 'latest inflation data'. "
                    "Do not return plain text."
                ),
            }
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            }
        ],
    )

    assert msg["extra"]["actions"]
    assert msg["extra"]["actions"][0]["name"] == "search"
