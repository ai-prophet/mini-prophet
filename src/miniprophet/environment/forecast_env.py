"""ForecastEnvironment: thin dispatcher that delegates to modular Tool instances."""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel

from miniprophet import Tool
from miniprophet.environment.source_board import Source, SourceBoard
from miniprophet.tools.search import SearchBackend

logger = logging.getLogger("miniprophet.environment")


def create_default_tools(
    search_tool: SearchBackend,
    outcomes: list[str],
    board: SourceBoard,
    *,
    search_limit: int = 10,
    search_results_limit: int = 5,
    max_source_display_chars: int = 2000,
) -> list[Tool]:
    """Build the standard set of forecast tools sharing a common board and source registry."""
    from miniprophet.tools.search_tool import SearchForecastTool, SearchToolConfig
    from miniprophet.tools.source_board_tools import AddSourceTool, EditNoteTool
    from miniprophet.tools.submit import SubmitTool

    source_registry: dict[str, Source] = {}
    search_config = SearchToolConfig(
        search_results_limit=search_results_limit,
        max_source_display_chars=max_source_display_chars,
    )

    return [
        SearchForecastTool(
            search_backend=search_tool,
            source_registry=source_registry,
            search_limit=search_limit,
            config=search_config,
        ),
        AddSourceTool(source_registry=source_registry, board=board, outcomes=outcomes),
        EditNoteTool(board=board, outcomes=outcomes),
        SubmitTool(outcomes=outcomes, board=board),
    ]


class ForecastEnvConfig(BaseModel):
    search_results_limit: int = 5
    max_source_display_chars: int = 2000


class ForecastEnvironment:
    """Dispatches tool-call actions to registered Tool instances."""

    def __init__(
        self,
        tools: list[Tool],
        *,
        board: SourceBoard | None = None,
        **kwargs: Any,
    ) -> None:
        if board is None:
            board = SourceBoard()
        self.board = board
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
        """Serialize raw searched sources and compact board references."""
        sources: dict[str, dict] = {}
        search_tool = self.get_tool("search")
        if search_tool is not None and hasattr(search_tool, "serialize_sources"):
            payload = search_tool.serialize_sources()  # type: ignore[attr-defined]
            if isinstance(payload, dict):
                sources = payload

        source_board: list[dict] = []
        for entry in self.board.serialize():
            board_entry = {
                "source_id": entry.get("source_id"),
                "note": entry.get("note", ""),
                "reaction": entry.get("reaction", {}),
            }
            if not board_entry["source_id"]:
                source = entry.get("source", {})
                if isinstance(source, dict):
                    board_entry["source"] = {
                        "url": source.get("url", ""),
                        "title": source.get("title", ""),
                        "date": source.get("date"),
                    }
            source_board.append(board_entry)

        return {
            "sources": sources,
            "source_board": source_board,
        }

    def serialize(self) -> dict:
        return {}
