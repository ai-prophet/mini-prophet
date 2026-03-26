"""Search tool: web search with source ID assignment via SourceRegistry."""

from __future__ import annotations

import copy
import logging
from typing import Any

from pydantic import BaseModel

from miniprophet.environment.source_board import Source
from miniprophet.environment.source_registry import SourceRegistry, render_source_preview
from miniprophet.exceptions import SearchAuthError, SearchError
from miniprophet.tools.search import SearchBackend, SearchResult

logger = logging.getLogger("miniprophet.tools.search")

BASE_SEARCH_PARAMETERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "The search query to find relevant information.",
        },
    },
    "required": ["query"],
}

SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search",
        "description": (
            "Search the web for information relevant to the forecasting problem. "
            "Returns a list of sources with titles and gist summaries. "
            "Each source is assigned a global ID (S1, S2, ...) that persists across searches."
        ),
        "parameters": BASE_SEARCH_PARAMETERS_SCHEMA,
    },
}


class SearchToolConfig(BaseModel):
    search_results_limit: int = 5


class SearchForecastTool:
    """Wraps a SearchBackend into a forecast Tool with SourceRegistry-based ID tracking."""

    def __init__(
        self,
        search_backend: SearchBackend,
        registry: SourceRegistry,
        *,
        search_limit: int = 10,
        problem_id: str = "main",
        config: SearchToolConfig | None = None,
    ) -> None:
        self._backend = search_backend
        self._registry = registry
        self._search_limit = search_limit
        self._problem_id = problem_id
        self._config = config or SearchToolConfig()
        self.n_searches: int = 0

    @property
    def name(self) -> str:
        return "search"

    def get_schema(self) -> dict:
        schema = copy.deepcopy(SEARCH_SCHEMA)
        backend_parameters = getattr(self._backend, "search_parameters_schema", None)
        if isinstance(backend_parameters, dict):
            schema["function"]["parameters"] = backend_parameters  # type: ignore
        return schema

    async def _assign_source_id(self, source: Source) -> str:
        return await self._registry.add(source, problem_id=self._problem_id)

    async def execute(self, args: dict) -> dict:
        query = args.get("query", "").strip()
        if not query:
            return {"output": "Error: 'query' is required for the search tool.", "error": True}

        if self.n_searches >= self._search_limit:
            return {
                "output": (
                    f"Search limit reached ({self._search_limit} queries). "
                    "You can still use `read_source` to read the full text of sources "
                    "you have already found, or `list_sources` to review them. "
                    "Then submit your forecast."
                ),
                "error": True,
            }

        try:
            search_kwargs = {k: v for k, v in args.items() if k != "query"}
            result: SearchResult = await self._backend.search(
                query,
                limit=self._config.search_results_limit,
                **search_kwargs,
            )
        except SearchAuthError:
            raise
        except SearchError as exc:
            return {
                "output": f"Search failed: {exc}. Try again or use existing sources.",
                "error": True,
            }

        self.n_searches += 1

        # Assign IDs and collect results
        search_results: list[tuple[str, Source]] = []
        for src in result.sources:
            sid = await self._assign_source_id(src)
            search_results.append((sid, src))

        if not search_results:
            body = "No sources found for this query."
        else:
            lines: list[str] = [f'<search_results count="{len(search_results)}">']
            for sid, src in search_results:
                entry = await self._registry.get(sid)
                lines.append(
                    render_source_preview(
                        source_id=sid,
                        title=src.title,
                        url=src.url,
                        date=src.date,
                        gist=entry.gist,
                        full_length=len(src.snippet),
                    )
                )
            lines.append("</search_results>")
            body = "\n".join(lines)

        return {
            "output": body,
            "search_cost": result.cost,
            "search_results": search_results,
        }

    def display(self, output: dict) -> None:
        from miniprophet.cli.components.observation import print_observation
        from miniprophet.cli.components.search_results import print_search_observation

        search_results = output.get("search_results", [])
        if not output.get("error") and search_results:
            print_search_observation(search_results, self._registry.max_gist_chars)
        else:
            print_observation(output)
