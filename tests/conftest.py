from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

# Allow `pytest` without editable install.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from miniprophet.environment.source_board import Source  # noqa: E402
from miniprophet.tools.search import SearchResult  # noqa: E402


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-live-api",
        action="store_true",
        default=False,
        help="Run tests marked with 'live_api' (disabled by default).",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-live-api"):
        return
    skip_live = pytest.mark.skip(reason="live_api tests require --run-live-api")
    for item in items:
        if "live_api" in item.keywords:
            item.add_marker(skip_live)


class DummyModel:
    """Deterministic model stub for agent-loop tests."""

    @dataclass
    class Config:
        model_name: str = "dummy/model"

    def __init__(self, scripted_messages: list[dict] | None = None) -> None:
        self.config = self.Config()
        self._messages = list(scripted_messages or [])
        self.n_queries = 0

    def query(self, messages: list[dict], tools: list[dict]) -> dict:
        self.n_queries += 1
        if not self._messages:
            return {
                "role": "assistant",
                "content": "",
                "extra": {"actions": [], "cost": 0.0, "timestamp": time.time()},
            }
        return self._messages.pop(0)

    def format_message(self, **kwargs) -> dict:
        return dict(kwargs)

    def format_observation_messages(self, message: dict, outputs: list[dict]) -> list[dict]:
        actions = message.get("extra", {}).get("actions", [])
        out: list[dict] = []
        for action, output in zip(actions, outputs):
            text = output.get("output", "")
            if output.get("error"):
                content = f"<error>\n{text}\n</error>"
            else:
                content = f"<output>\n{text}\n</output>"
            role = "tool" if action.get("tool_call_id") else "user"
            msg = {
                "role": role,
                "content": content,
                "extra": {
                    "error": output.get("error", False),
                    "search_cost": output.get("search_cost", 0.0),
                    "timestamp": time.time(),
                },
            }
            if action.get("tool_call_id"):
                msg["tool_call_id"] = action["tool_call_id"]
            out.append(msg)
        return out

    def serialize(self) -> dict:
        return {"info": {"config": {"model": {"model_name": self.config.model_name}}}}


class DummyBoard:
    def __init__(self) -> None:
        self._serialized: list[dict] = []

    def render(self) -> str:
        return "<source_board>(dummy)</source_board>"

    def serialize(self) -> list[dict]:
        return list(self._serialized)


class DummyEnvironment:
    """Environment stub that returns scripted outputs."""

    def __init__(self, outputs: list[dict] | None = None, with_board: bool = True) -> None:
        self.outputs = list(outputs or [])
        self.executed: list[tuple[dict, dict]] = []
        self._tools = {
            "search": {"type": "function", "function": {"name": "search", "parameters": {}}}
        }
        if with_board:
            self.board = DummyBoard()

    def execute(self, action: dict, **kwargs) -> dict:
        self.executed.append((action, kwargs))
        if self.outputs:
            return self.outputs.pop(0)
        return {"output": "ok"}

    def get_tool_schemas(self) -> list[dict]:
        return list(self._tools.values())

    def get_tool(self, name: str):
        return self._tools.get(name)

    def serialize_sources_state(self) -> dict:
        return {"sources": {}, "source_board": []}

    def serialize(self) -> dict:
        return {"info": {"config": {"environment": "dummy"}}}


class DummySearchTool:
    search_parameters_schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }

    def __init__(
        self,
        sources: list[Source] | None = None,
        *,
        cost: float = 0.0,
        error: Exception | None = None,
    ) -> None:
        self._sources = list(sources or [])
        self._cost = cost
        self._error = error
        self.calls: list[dict[str, Any]] = []

    def search(self, query: str, limit: int = 5, **kwargs) -> SearchResult:
        self.calls.append({"query": query, "limit": limit, "kwargs": kwargs})
        if self._error is not None:
            raise self._error
        return SearchResult(sources=self._sources[:limit], cost=self._cost)

    def serialize(self) -> dict:
        return {"info": {"config": {"search": {"search_class": "dummy"}}}}


@pytest.fixture
def dummy_source() -> Source:
    return Source(url="https://example.com", title="Example", snippet="snippet", date="2026-01-01")


@pytest.fixture
def two_sources() -> list[Source]:
    return [
        Source(url="https://a.example", title="A", snippet="first", date="2026-01-02"),
        Source(url="https://b.example", title="B", snippet="second", date="2026-01-03"),
    ]


@pytest.fixture
def assistant_action_message() -> Any:
    def _make(
        *,
        name: str,
        arguments: str = "{}",
        tool_call_id: str | None = "tc_1",
        cost: float = 0.0,
        content: str = "",
    ) -> dict:
        return {
            "role": "assistant",
            "content": content,
            "extra": {
                "actions": [
                    {
                        "name": name,
                        "arguments": arguments,
                        "tool_call_id": tool_call_id,
                    }
                ],
                "cost": cost,
                "timestamp": time.time(),
            },
        }

    return _make
