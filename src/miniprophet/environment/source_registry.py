"""SourceRegistry: async-safe, harness-level source store."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from miniprophet.environment.source_board import Source


@dataclass
class SourceSummary:
    """Summary metadata for a registered source (saves context space)."""

    source: Source
    problem_id: str  # "main" or a subproblem ID — tracks invocation context
    gist: str  # snippet truncated to max_gist_chars


class SourceRegistry:
    """Async-safe, harness-level source store shared across main agent and subagents.

    Each forecasting problem (in batch mode) gets its own independent registry.
    Uses asyncio.Lock to protect concurrent writes from parallel subagents.
    """

    def __init__(self, *, max_gist_chars: int = 200) -> None:
        self._entries: dict[str, SourceSummary] = {}
        self._next_id: int = 1
        self._max_gist_chars = max_gist_chars
        self._lock = asyncio.Lock()

    @property
    def max_gist_chars(self) -> int:
        return self._max_gist_chars

    def __len__(self) -> int:
        return len(self._entries)

    async def add(self, source: Source, *, problem_id: str = "main") -> str:
        """Add a source under lock. Return its ID (e.g. 'S1')."""
        async with self._lock:
            sid = f"S{self._next_id}"
            self._next_id += 1
            gist = source.snippet[: self._max_gist_chars]
            self._entries[sid] = SourceSummary(source=source, problem_id=problem_id, gist=gist)
            return sid

    async def get(self, source_id: str) -> SourceSummary:
        """Get full entry. Lock-free read. Raises KeyError if missing."""
        return self._entries[source_id]

    async def get_full_content(self, source_id: str) -> Source:
        """Get the raw Source object. Lock-free read. Raises KeyError if missing."""
        return self._entries[source_id].source

    async def list_sources(self, problem_id: str | None = None) -> list[dict[str, Any]]:
        """Return metadata rows. Filter by problem_id if given. Lock-free read."""
        results: list[dict[str, Any]] = []
        items = sorted(
            self._entries.items(),
            key=lambda kv: (
                int(kv[0][1:]) if kv[0].startswith("S") and kv[0][1:].isdigit() else 10**9
            ),
        )
        for sid, entry in items:
            if problem_id is not None and entry.problem_id != problem_id:
                continue
            results.append(
                {
                    "source_id": sid,
                    "title": entry.source.title,
                    "url": entry.source.url,
                    "date": entry.source.date,
                    "gist": entry.gist,
                    "problem_id": entry.problem_id,
                }
            )
        return results

    def serialize(self) -> dict[str, dict[str, Any]]:
        """Full serialization for sources.json artifact. Synchronous — called after agent completes."""
        items = sorted(
            self._entries.items(),
            key=lambda kv: (
                int(kv[0][1:]) if kv[0].startswith("S") and kv[0][1:].isdigit() else 10**9
            ),
        )
        return {
            sid: {
                "url": entry.source.url,
                "title": entry.source.title,
                "snippet": entry.source.snippet,
                "date": entry.source.date,
                "problem_id": entry.problem_id,
            }
            for sid, entry in items
        }
