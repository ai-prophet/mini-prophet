"""Tavily Search API integration for mini-prophet.

Uses the official Tavily Python SDK (`tavily-python`). Tavily returns
structured search results with pre-extracted content snippets, so no
additional scraping is needed.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Literal

from tavily import TavilyClient
from tavily.errors import InvalidAPIKeyError, MissingAPIKeyError, UsageLimitExceededError

from miniprophet.environment.source_board import Source
from miniprophet.exceptions import SearchAuthError, SearchNetworkError, SearchRateLimitError
from miniprophet.tools.search import SearchResult

logger = logging.getLogger("miniprophet.tools.search.tavily")

TAVILY_SEARCH_PARAMETERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "The search query to find relevant information.",
        },
        "start_date": {
            "type": "string",
            "description": (
                "Optional start date filter (YYYY-MM-DD). "
                "Only return results published on or after this date."
            ),
        },
    },
    "required": ["query"],
}


class TavilySearchBackend:
    """Search backend backed by the Tavily Python SDK.

    Tavily returns ranked web results with extracted content already included
    in each result's ``content`` field, so no additional scraping is needed.

    Supports time-based filtering via ``time_range`` (day/week/month/year)
    and ``start_date``/``end_date`` (YYYY-MM-DD format) as well as the
    runtime ``search_date_after`` / ``search_date_before`` kwargs (MM/DD/YYYY).
    """

    search_parameters_schema = TAVILY_SEARCH_PARAMETERS_SCHEMA

    def __init__(
        self,
        search_depth: Literal["basic", "advanced"] = "basic",
        topic: Literal["general", "news", "finance"] = "general",
        max_characters: int = 3000,
        country: str = "us",
        timeout: int = 60,
    ) -> None:
        self._api_key = os.getenv("TAVILY_API_KEY", "")
        if not self._api_key:
            raise SearchAuthError("TAVILY_API_KEY environment variable is not set")

        self._search_depth = search_depth
        self._topic = topic
        self._max_characters = max_characters
        self._country = country
        self._timeout = timeout
        self._client = TavilyClient(api_key=self._api_key)

    def search(self, query: str, limit: int = 5, **kwargs: Any) -> SearchResult:
        payload: dict[str, Any] = {
            "query": query,
            "max_results": min(limit, 20),
            "search_depth": self._search_depth,
            "topic": self._topic,
            "country": self._country,
            "include_raw_content": False,
            "include_answer": False,
            "include_images": False,
            "include_usage": True,
            "timeout": self._timeout,
        }

        # Handle runtime date filters (MM/DD/YYYY -> YYYY-MM-DD)
        search_date_after = kwargs.pop("search_date_after", None)
        search_date_before = kwargs.pop("search_date_before", None)
        payload.update(kwargs)

        if search_date_after:
            payload["start_date"] = self._date_mmddyyyy_to_iso(search_date_after)
        if search_date_before:
            payload["end_date"] = self._date_mmddyyyy_to_iso(search_date_before)

        payload = {k: v for k, v in payload.items() if v is not None}

        try:
            resp = self._client.search(**payload)
        except (InvalidAPIKeyError, MissingAPIKeyError) as exc:
            raise SearchAuthError(
                "Tavily API authentication failed. Check TAVILY_API_KEY."
            ) from exc
        except UsageLimitExceededError as exc:
            raise SearchRateLimitError("Tavily API usage limit exceeded") from exc
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            if status_code is None:
                status_code = getattr(getattr(exc, "response", None), "status_code", None)

            if status_code == 401:
                raise SearchAuthError(
                    "Tavily API authentication failed. Check TAVILY_API_KEY."
                ) from exc
            if status_code == 429:
                raise SearchRateLimitError("Tavily API rate limit exceeded") from exc
            raise SearchNetworkError(f"Tavily SDK request failed: {exc}") from exc

        sources: list[Source] = []
        for item in resp.get("results", []):
            url = item.get("url", "")
            if not url:
                continue

            title = item.get("title", "") or url
            content = item.get("content", "")
            snippet = content[: self._max_characters] if content else ""
            date = item.get("published_date") or None

            sources.append(Source(url=url, title=title, snippet=snippet, date=date))

        cost = self._extract_cost(resp)
        logger.info(f"Tavily search '{query}': {len(sources)} source(s), cost={cost}")
        return SearchResult(sources=sources, cost=cost)

    @staticmethod
    def _extract_cost(response: dict) -> float:
        """Extract credit cost from the response usage field."""
        usage = response.get("usage")
        if not isinstance(usage, dict):
            return 0.0
        try:
            return float(usage.get("credits", 0))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _date_mmddyyyy_to_iso(value: str) -> str:
        """Convert MM/DD/YYYY runtime date filter to YYYY-MM-DD for Tavily."""
        from datetime import datetime

        try:
            parsed = datetime.strptime(str(value).strip(), "%m/%d/%Y")
        except ValueError as exc:
            raise SearchNetworkError(
                f"Invalid date '{value}'. Expected MM/DD/YYYY for runtime search date filters."
            ) from exc
        return parsed.strftime("%Y-%m-%d")

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "search": {
                        "search_class": "tavily",
                        "search_depth": self._search_depth,
                        "topic": self._topic,
                        "max_characters": self._max_characters,
                        "country": self._country,
                        "timeout": self._timeout,
                    }
                }
            }
        }
