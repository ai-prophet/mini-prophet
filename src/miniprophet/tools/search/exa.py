"""Exa Search API integration for mini-prophet.

Uses the official Exa Python SDK (`exa-py`) and requests page contents
directly in the search call via the `contents` payload.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any

from exa_py import Exa

from miniprophet.environment.source_board import Source
from miniprophet.exceptions import SearchAuthError, SearchNetworkError, SearchRateLimitError
from miniprophet.tools.search import SearchResult

logger = logging.getLogger("miniprophet.tools.search.exa")

CONTENT_NOT_AVAILABLE = "content not available"
EXA_SEARCH_PARAMETERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "The search query to find relevant information.",
        },
        "start_published_date": {
            "type": "string",
            "description": (
                "Optional lower published-date bound in ISO format (e.g. '2025-01-01T00:00:00Z')."
            ),
        },
    },
    "required": ["query"],
}


class ExaSearchBackend:
    """Search backend backed by the Exa Python SDK."""

    search_parameters_schema = EXA_SEARCH_PARAMETERS_SCHEMA

    def __init__(
        self,
        content_mode: str = "text",
        text_max_characters: int = 3000,
        highlights_max_characters: int = 1200,
        search_type: str = "auto",
        category: str | None = None,
    ) -> None:
        self._api_key = os.getenv("EXA_API_KEY", "")
        if not self._api_key:
            raise SearchAuthError("EXA_API_KEY environment variable is not set")

        normalized_mode = content_mode.strip().lower()
        if normalized_mode not in {"text", "highlights"}:
            raise ValueError("search.exa.content_mode must be one of: text, highlights")

        self._content_mode = normalized_mode
        self._text_max_characters = max(1, int(text_max_characters))
        self._highlights_max_characters = max(1, int(highlights_max_characters))
        self._search_type = search_type
        self._category = category
        self._client = Exa(api_key=self._api_key)
        self._async_client = None

    async def search(self, query: str, limit: int = 5, **kwargs: Any) -> SearchResult:
        if self._async_client is None:
            from exa_py import AsyncExa

            self._async_client = AsyncExa(api_key=self._api_key)

        payload: dict[str, Any] = {
            "query": query,
            "num_results": min(limit, 100),
            "contents": self._build_contents_payload(),
            "type": self._search_type,
        }
        if self._category:
            payload["category"] = self._category

        search_date_before = kwargs.pop("search_date_before", None)
        search_date_after = kwargs.pop("search_date_after", None)
        payload.update(kwargs)
        if search_date_after:
            payload["start_published_date"] = self._date_mmddyyyy_to_iso(
                search_date_after, end_of_day=False
            )
        if search_date_before:
            payload["end_published_date"] = self._date_mmddyyyy_to_iso(
                search_date_before, end_of_day=True
            )

        payload = {k: v for k, v in payload.items() if v is not None}

        try:
            resp = await self._async_client.search(**payload)
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            if status_code is None:
                status_code = getattr(getattr(exc, "response", None), "status_code", None)

            if status_code == 401:
                raise SearchAuthError("Exa API authentication failed. Check EXA_API_KEY.")
            if status_code == 429:
                raise SearchRateLimitError("Exa API rate limit exceeded")
            raise SearchNetworkError(f"Exa SDK request failed: {exc}") from exc

        sources: list[Source] = []
        for item in getattr(resp, "results", []) or []:
            url = self._as_str(self._get_field(item, "url"))
            if not url:
                continue

            title = self._as_str(self._get_field(item, "title")) or url
            date = self._as_str(self._get_field(item, "published_date")) or None
            snippet = self._extract_snippet(item)

            sources.append(Source(url=url, title=title, snippet=snippet, date=date))

        logger.info(f"Exa async search '{query}': {len(sources)} source(s)")
        return SearchResult(sources=sources, cost=self._extract_cost(resp))

    def _build_contents_payload(self) -> dict[str, Any]:
        if self._content_mode == "highlights":
            return {"highlights": {"max_characters": self._highlights_max_characters}}
        return {"text": {"max_characters": self._text_max_characters}}

    def _extract_snippet(self, item: Any) -> str:
        text = self._as_str(self._get_field(item, "text")).strip()
        summary = self._as_str(self._get_field(item, "summary")).strip()
        highlights = self._get_field(item, "highlights")
        highlights_text = ""
        if isinstance(highlights, list):
            highlights_text = "\n".join(
                part.strip() for part in highlights if isinstance(part, str) and part.strip()
            ).strip()

        if self._content_mode == "highlights":
            return highlights_text or text or summary or CONTENT_NOT_AVAILABLE
        return text or highlights_text or summary or CONTENT_NOT_AVAILABLE

    def _extract_cost(self, response: Any) -> float:
        cost_dollars = self._get_field(response, "cost_dollars")
        if cost_dollars is None:
            return 0.0
        total = self._get_field(cost_dollars, "total")
        try:
            return float(total or 0.0)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _date_mmddyyyy_to_iso(value: str, *, end_of_day: bool) -> str:
        try:
            parsed = datetime.strptime(str(value).strip(), "%m/%d/%Y")
        except ValueError as exc:
            raise SearchNetworkError(
                f"Invalid date '{value}'. Expected MM/DD/YYYY for runtime search date filters."
            ) from exc
        if end_of_day:
            parsed = parsed.replace(hour=23, minute=59, second=59)
        else:
            parsed = parsed.replace(hour=0, minute=0, second=0)
        return parsed.strftime("%Y-%m-%dT%H:%M:%SZ")

    @staticmethod
    def _get_field(obj: Any, key: str) -> Any:
        if obj is None:
            return None
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)

    @staticmethod
    def _as_str(value: Any) -> str:
        return value if isinstance(value, str) else ""

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "search": {
                        "search_class": "exa",
                        "content_mode": self._content_mode,
                        "text_max_characters": self._text_max_characters,
                        "highlights_max_characters": self._highlights_max_characters,
                        "search_type": self._search_type,
                        "category": self._category,
                    }
                }
            }
        }
