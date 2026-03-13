"""Search tool: web search with source ID assignment."""

from __future__ import annotations

import copy
import logging
from typing import Any

from pydantic import BaseModel

from miniprophet.environment.source_board import Source
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
            "Returns a list of sources with titles, snippets, and article content. "
            "Each source is assigned a global ID (S1, S2, ...) that persists across searches."
        ),
        "parameters": BASE_SEARCH_PARAMETERS_SCHEMA,
    },
}


class SearchToolConfig(BaseModel):
    search_results_limit: int = 5
    max_source_display_chars: int = 2000


class SearchForecastTool:
    """Wraps a SearchBackend into a forecast Tool with source ID tracking."""

    def __init__(
        self,
        search_backend: SearchBackend,
        source_registry: dict[str, Source],
        *,
        search_limit: int = 10,
        config: SearchToolConfig | None = None,
    ) -> None:
        self._backend = search_backend
        self._source_registry = source_registry
        self._search_limit = search_limit
        self._config = config or SearchToolConfig()
        self._next_source_id: int = 1
        self.n_searches: int = 0
        self.last_search_results: list[tuple[str, Source]] = []

    @property
    def name(self) -> str:
        return "search"

    def get_schema(self) -> dict:
        schema = copy.deepcopy(SEARCH_SCHEMA)
        backend_parameters = getattr(self._backend, "search_parameters_schema", None)
        if isinstance(backend_parameters, dict):
            schema["function"]["parameters"] = backend_parameters  # type: ignore
        return schema

    def _assign_source_id(self, source: Source) -> str:
        sid = f"S{self._next_source_id}"
        self._next_source_id += 1
        self._source_registry[sid] = source
        return sid

    def serialize_sources(self) -> dict[str, dict[str, Any]]:
        """Return all discovered sources keyed by stable source_id (S1, S2, ...)."""
        items = sorted(
            self._source_registry.items(),
            key=lambda kv: (
                int(kv[0][1:]) if kv[0].startswith("S") and kv[0][1:].isdigit() else 10**9
            ),
        )
        return {
            sid: {
                "url": src.url,
                "title": src.title,
                "snippet": src.snippet,
                "date": src.date,
            }
            for sid, src in items
        }

    def execute(self, args: dict) -> dict:
        query = args.get("query", "").strip()
        if not query:
            return {"output": "Error: 'query' is required for the search tool.", "error": True}

        if self.n_searches >= self._search_limit:
            return {
                "output": (
                    f"Search limit reached ({self._search_limit} queries). "
                    "Use your existing sources to submit a forecast."
                ),
                "error": True,
            }

        try:
            search_kwargs = {k: v for k, v in args.items() if k != "query"}
            result: SearchResult = self._backend.search(
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
        self.last_search_results = [(self._assign_source_id(src), src) for src in result.sources]

        if not self.last_search_results:
            body = "No sources found for this query."
        else:
            lines: list[str] = [f'<search_results count="{len(self.last_search_results)}">']
            for sid, src in self.last_search_results:
                date_line = f"Date: {src.date or 'No date info'}\n"
                lines.append(
                    f'<result id="{sid}" title="{src.title}" url="{src.url}">\n'
                    f"{date_line}"
                    f"Snippet: {src.snippet}\n"
                    f"</result>"
                )
            lines.append("</search_results>")
            body = "\n".join(lines)

        return {
            "output": body,
            "search_cost": result.cost,
            "search_results": self.last_search_results,
        }

    def display(self, output: dict) -> None:
        from miniprophet.cli.components.observation import print_observation
        from miniprophet.cli.components.search_results import print_search_observation

        search_results = output.get("search_results", [])
        if not output.get("error") and search_results:
            print_search_observation(search_results, self._config.max_source_display_chars)
        else:
            print_observation(output)
