"""Brave Search API integration for mini-prophet.

Simplified synchronous implementation inspired by the reference snippet
in snippets/brave_search.py.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import requests
import trafilatura

from miniprophet.environment.source_board import Source
from miniprophet.exceptions import SearchAuthError, SearchNetworkError, SearchRateLimitError
from miniprophet.tools.search import SearchResult

logger = logging.getLogger("miniprophet.tools.search.brave")

BRAVE_API_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
BRAVE_SEARCH_PARAMETERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "The search query to find relevant information.",
        },
        "freshness": {
            "type": "string",
            "description": (
                "Optional date filter for recent results. Use Brave freshness values "
                "'pd', 'pw', 'pm', 'py', or a custom range like "
                "'2025-01-01to2025-12-31'."
            ),
        },
    },
    "required": ["query"],
}


class BraveSearchBackend:
    """Search backend backed by the Brave Search API + trafilatura content extraction."""

    search_parameters_schema = BRAVE_SEARCH_PARAMETERS_SCHEMA

    def __init__(
        self,
        connect_timeout: int = 10,
        total_timeout: int = 30,
        max_retries: int = 3,
        max_extract_chars: int = 3000,
    ) -> None:
        self._api_key = os.getenv("BRAVE_API_KEY", "")
        self._connect_timeout = connect_timeout
        self._total_timeout = total_timeout
        self._max_retries = max_retries
        self._max_extract_chars = max_extract_chars

    def search(self, query: str, limit: int = 5, **kwargs: Any) -> SearchResult:
        if "search_date_before" in kwargs or "search_date_after" in kwargs:
            kwargs.pop("search_date_before", None)
            kwargs.pop("search_date_after", None)
            logger.warning("Brave has not supported date filtering yet")

        freshness = kwargs.get("freshness")
        links = self._get_links(query, limit, freshness=freshness)
        sources: list[Source] = []
        for link in links:
            text = self._fetch_article_text(link["url"])
            if text:
                sources.append(
                    Source(
                        url=link["url"],
                        title=link["title"],
                        snippet=link["snippet"],
                        date=link.get("date"),
                    )
                )
        logger.info(f"Search '{query}': {len(sources)}/{len(links)} sources extracted")
        return SearchResult(sources=sources, cost=0.0)

    def _get_links(
        self, query: str, limit: int, freshness: Any | None = None
    ) -> list[dict[str, str | None]]:
        if not self._api_key:
            raise SearchAuthError("BRAVE_API_KEY environment variable is not set")

        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self._api_key,
        }
        params = {
            "q": query,
            "count": limit,
            "country": "US",
            "search_lang": "en",
            "result_filter": "web",
        }
        if isinstance(freshness, str) and freshness.strip():
            params["freshness"] = freshness.strip()

        try:
            resp = requests.get(
                BRAVE_API_ENDPOINT,
                headers=headers,
                params=params,
                timeout=(self._connect_timeout, self._total_timeout),
            )
            resp.raise_for_status()
        except requests.exceptions.HTTPError:
            if resp.status_code == 401:
                raise SearchAuthError("Brave API authentication failed. Check BRAVE_API_KEY.")
            if resp.status_code == 429:
                raise SearchRateLimitError("Brave API rate limit exceeded")
            raise SearchNetworkError(f"Brave API HTTP {resp.status_code}: {resp.text[:200]}")
        except requests.exceptions.RequestException as exc:
            raise SearchNetworkError(f"Brave API request failed: {exc}") from exc

        data = resp.json()
        results: list[dict[str, str | None]] = []
        for item in data.get("web", {}).get("results", []):
            url = item.get("url")
            if url:
                results.append(
                    {
                        "url": url,
                        "title": item.get("title", ""),
                        "snippet": item.get("description", ""),
                        "date": item.get("age") or item.get("page_age"),
                    }
                )
        return results[:limit]

    def _fetch_article_text(self, url: str) -> str | None:
        for attempt in range(self._max_retries):
            try:
                resp = requests.get(
                    url,
                    timeout=(self._connect_timeout, self._total_timeout),
                    headers={"User-Agent": "miniprophet/0.1"},
                    allow_redirects=True,
                )
                resp.raise_for_status()
                text = trafilatura.extract(
                    resp.text,
                    include_comments=False,
                    no_fallback=True,
                    url=url,
                )
                if text:
                    return text
                return None
            except Exception as exc:
                if attempt == self._max_retries - 1:
                    logger.debug(f"Failed to fetch {url[:60]}: {exc}")
                    return None

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "search": {
                        "search_class": "brave",
                        "connect_timeout": self._connect_timeout,
                        "total_timeout": self._total_timeout,
                        "max_retries": self._max_retries,
                        "max_extract_chars": self._max_extract_chars,
                    }
                }
            }
        }
