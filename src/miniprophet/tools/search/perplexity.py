"""Perplexity Search API integration for mini-prophet.

Uses the Perplexity Search API (https://docs.perplexity.ai/docs/search/quickstart)
which returns structured web results with pre-extracted content in the snippet field.
No separate content extraction step (like trafilatura) is needed.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any

from perplexity import Perplexity

from miniprophet.environment.source_board import Source
from miniprophet.exceptions import (
    SearchAuthError,
    SearchNetworkError,
    SearchRateLimitError,
)
from miniprophet.tools.search import SearchResult

logger = logging.getLogger("miniprophet.tools.search.perplexity")

# pricing from: https://docs.perplexity.ai/docs/getting-started/pricing
PERPLEXITY_PER_SEARCH_COST = 5 / 1000
PERPLEXITY_SEARCH_PARAMETERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "The search query to find relevant information.",
        },
        "search_after_date_filter": {
            "type": "string",
            "description": (
                "Optional publication-time lower bound in MM/DD/YYYY format (e.g. '01/01/2025')."
            ),
        },
        "last_updated_after_filter": {
            "type": "string",
            "description": (
                "Optional last-update lower bound in MM/DD/YYYY format (e.g. '01/01/2025')."
            ),
        },
    },
    "required": ["query"],
}


class PerplexitySearchBackend:
    """Search backend backed by the Perplexity Search API.

    Perplexity returns ranked web results with substantial extracted content
    already included in the snippet field, so no additional scraping is needed.
    """

    search_parameters_schema = PERPLEXITY_SEARCH_PARAMETERS_SCHEMA

    def __init__(
        self,
        timeout: int = 30,
        max_tokens_per_page: int = 4096,
        max_tokens: int = 10000,
        country: str = "US",
    ) -> None:
        self._api_key = os.getenv("PERPLEXITY_API_KEY", "")
        if not self._api_key:
            raise SearchAuthError("PERPLEXITY_API_KEY environment variable is not set")

        self._timeout = timeout
        self._max_tokens_per_page = max_tokens_per_page
        self._max_tokens = max_tokens
        self._country = country
        self._client = Perplexity(api_key=self._api_key, timeout=float(self._timeout))
        self._async_client = None

    def search(self, query: str, limit: int = 5, **kwargs: Any) -> SearchResult:
        payload: dict[str, Any] = {
            "query": query,
            "max_results": min(limit, 20),
            "max_tokens_per_page": self._max_tokens_per_page,
            "max_tokens": self._max_tokens,
        }
        if self._country:
            payload["country"] = self._country

        search_date_before = kwargs.pop("search_date_before", None)
        search_date_after = kwargs.pop("search_date_after", None)
        payload.update(kwargs)
        # This would effectively override any "after date" set by the model itself
        if search_date_after:
            payload["search_after_date_filter"] = search_date_after
            payload["last_updated_after_filter"] = search_date_after
        if search_date_before:
            payload["search_before_date_filter"] = search_date_before
            payload["last_updated_before_filter"] = search_date_before

        try:
            resp = self._client.search.create(**payload)
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            if status_code is None:
                status_code = getattr(getattr(exc, "response", None), "status_code", None)

            if status_code == 401:
                raise SearchAuthError(
                    "Perplexity API authentication failed. Check PERPLEXITY_API_KEY."
                )
            if status_code == 429:
                raise SearchRateLimitError("Perplexity API rate limit exceeded")
            raise SearchNetworkError(f"Perplexity SDK request failed: {exc}") from exc

        sources: list[Source] = []
        for item in resp.results:
            url = item.url or ""
            snippet = item.snippet or ""
            title = item.title or ""
            date, updated_date = item.date, item.last_updated
            # we take the latest date between the publication date and the last updated date
            if date is not None and updated_date is not None:
                d1, d2 = (
                    datetime.strptime(date, "%Y-%m-%d"),
                    datetime.strptime(updated_date, "%Y-%m-%d"),
                )
                date = max(d1, d2).strftime("%Y-%m-%d")
            sources.append(Source(url=url, title=title, snippet=snippet, date=date))

        logger.info(f"Perplexity search '{query}': {len(sources)} source(s)")
        # For perplexity, the cost is fixed for each request, regardless of the number of sources returned
        return SearchResult(sources=sources, cost=PERPLEXITY_PER_SEARCH_COST)

    async def asearch(self, query: str, limit: int = 5, **kwargs: Any) -> SearchResult:
        if self._async_client is None:
            from perplexity import AsyncPerplexity

            self._async_client = AsyncPerplexity(
                api_key=self._api_key, timeout=float(self._timeout)
            )

        payload: dict[str, Any] = {
            "query": query,
            "max_results": min(limit, 20),
            "max_tokens_per_page": self._max_tokens_per_page,
            "max_tokens": self._max_tokens,
        }
        if self._country:
            payload["country"] = self._country

        search_date_before = kwargs.pop("search_date_before", None)
        search_date_after = kwargs.pop("search_date_after", None)
        payload.update(kwargs)
        if search_date_after:
            payload["search_after_date_filter"] = search_date_after
            payload["last_updated_after_filter"] = search_date_after
        if search_date_before:
            payload["search_before_date_filter"] = search_date_before
            payload["last_updated_before_filter"] = search_date_before

        try:
            resp = await self._async_client.search.create(**payload)
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            if status_code is None:
                status_code = getattr(getattr(exc, "response", None), "status_code", None)

            if status_code == 401:
                raise SearchAuthError(
                    "Perplexity API authentication failed. Check PERPLEXITY_API_KEY."
                )
            if status_code == 429:
                raise SearchRateLimitError("Perplexity API rate limit exceeded")
            raise SearchNetworkError(f"Perplexity SDK request failed: {exc}") from exc

        sources: list[Source] = []
        for item in resp.results:
            url = item.url or ""
            snippet = item.snippet or ""
            title = item.title or ""
            date, updated_date = item.date, item.last_updated
            if date is not None and updated_date is not None:
                d1, d2 = (
                    datetime.strptime(date, "%Y-%m-%d"),
                    datetime.strptime(updated_date, "%Y-%m-%d"),
                )
                date = max(d1, d2).strftime("%Y-%m-%d")
            sources.append(Source(url=url, title=title, snippet=snippet, date=date))

        logger.info(f"Perplexity async search '{query}': {len(sources)} source(s)")
        return SearchResult(sources=sources, cost=PERPLEXITY_PER_SEARCH_COST)

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "search": {
                        "search_class": "perplexity",
                        "timeout": self._timeout,
                        "max_tokens_per_page": self._max_tokens_per_page,
                        "max_tokens": self._max_tokens,
                        "country": self._country,
                    }
                }
            }
        }
