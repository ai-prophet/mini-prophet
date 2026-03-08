from __future__ import annotations

import json

from conftest import DummySearchTool

from miniprophet.environment.forecast_env import ForecastEnvironment, create_default_tools
from miniprophet.environment.source_board import Source, SourceBoard
from miniprophet.tools.search_tool import SearchForecastTool


class _EchoTool:
    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def get_schema(self) -> dict:
        return {"type": "function", "function": {"name": self._name, "parameters": {}}}

    def execute(self, args: dict) -> dict:
        return {"output": json.dumps(args, sort_keys=True)}

    def display(self, output: dict) -> None:
        return None


def test_forecast_environment_execute_passes_runtime_kwargs() -> None:
    env = ForecastEnvironment([_EchoTool("echo")])

    output = env.execute(
        {"name": "echo", "arguments": '{"query": "x"}'}, search_date_before="01/02/2026"
    )

    payload = json.loads(output["output"])
    assert payload["query"] == "x"
    assert payload["search_date_before"] == "01/02/2026"


def test_forecast_environment_execute_handles_invalid_json() -> None:
    env = ForecastEnvironment([_EchoTool("echo")])
    output = env.execute({"name": "echo", "arguments": "{"})
    assert output["error"] is True
    assert "Invalid JSON" in output["output"]


def test_forecast_environment_execute_unknown_tool() -> None:
    env = ForecastEnvironment([_EchoTool("echo")])
    output = env.execute({"name": "missing", "arguments": "{}"})
    assert output["error"] is True
    assert "Unknown tool" in output["output"]


def test_create_default_tools_and_serialize_sources_state(two_sources: list[Source]) -> None:
    board = SourceBoard()
    backend = DummySearchTool(sources=two_sources)
    tools = create_default_tools(search_tool=backend, outcomes=["A", "B"], board=board)
    env = ForecastEnvironment(tools, board=board)

    search_tool = env.get_tool("search")
    assert isinstance(search_tool, SearchForecastTool)
    search_tool.execute({"query": "q"})

    add_result = env.execute(
        {
            "name": "add_source",
            "arguments": '{"source_id": "S1", "note": "n", "reaction": {"A": "neutral"}}',
        }
    )
    assert "added to board" in add_result["output"]

    payload = env.serialize_sources_state()
    assert set(payload["sources"].keys()) == {"S1", "S2"}
    assert payload["source_board"][0]["source_id"] == "S1"


def test_forecast_environment_get_tool_schemas_returns_all_tools() -> None:
    env = ForecastEnvironment([_EchoTool("a"), _EchoTool("b")])
    schemas = env.get_tool_schemas()
    names = {s["function"]["name"] for s in schemas}
    assert names == {"a", "b"}
