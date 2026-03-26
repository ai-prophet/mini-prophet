"""ForecastEnvironment: thin dispatcher that delegates to modular Tool instances."""

from __future__ import annotations

import json
import logging
from typing import Any

from miniprophet import Tool
from miniprophet.environment.source_registry import SourceRegistry
from miniprophet.tools.search import SearchBackend

logger = logging.getLogger("miniprophet.environment")


def create_default_tools(
    search_tool: SearchBackend,
    registry: SourceRegistry,
    *,
    search_limit: int = 10,
    search_results_limit: int = 5,
) -> list[Tool]:
    """Build the standard set of forecast tools sharing a common SourceRegistry."""
    from miniprophet.tools.list_sources_tool import ListSourcesTool
    from miniprophet.tools.read_source_tool import ReadSourceTool
    from miniprophet.tools.search_tool import SearchForecastTool, SearchToolConfig
    from miniprophet.tools.submit import SubmitTool

    search_config = SearchToolConfig(
        search_results_limit=search_results_limit,
    )

    return [
        SearchForecastTool(
            search_backend=search_tool,
            registry=registry,
            search_limit=search_limit,
            config=search_config,
        ),
        ReadSourceTool(registry=registry),
        ListSourcesTool(registry=registry),
        SubmitTool(registry=registry),
    ]


class ForecastEnvironment:
    """Dispatches tool-call actions to registered Tool instances."""

    def __init__(
        self,
        tools: list[Tool],
        *,
        registry: SourceRegistry | None = None,
        **kwargs: Any,
    ) -> None:
        if registry is None:
            registry = SourceRegistry()
        self.registry = registry
        self._tools: dict[str, Tool] = {t.name: t for t in tools}

    async def execute(self, action: dict, **kwargs) -> dict:
        tool_name = action.get("name", "")
        try:
            raw_args = action.get("arguments", "{}")
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError as exc:
            return {"output": f"Invalid JSON in tool arguments: {exc}", "error": True}

        tool = self._tools.get(tool_name)
        if tool is None:
            return {"output": f"Unknown tool: {tool_name}", "error": True}
        args.update(kwargs)
        return await tool.execute(args)

    def get_tool_schemas(self) -> list[dict]:
        return [t.get_schema() for t in self._tools.values()]

    def get_tool(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def serialize_sources_state(self) -> dict:
        """Serialize all sources from the registry."""
        return {"sources": self.registry.serialize()}

    def serialize(self) -> dict:
        return {}
