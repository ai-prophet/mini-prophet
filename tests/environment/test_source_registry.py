from __future__ import annotations

import asyncio

import pytest

from miniprophet.environment.source_board import Source
from miniprophet.environment.source_registry import SourceRegistry


def _make_source(title: str = "T", snippet: str = "short snippet") -> Source:
    return Source(url="https://example.com", title=title, snippet=snippet, date="2026-01-01")


def test_add_and_get() -> None:
    registry = SourceRegistry()
    src = _make_source()
    sid = asyncio.run(registry.add(src))
    assert sid == "S1"
    entry = asyncio.run(registry.get("S1"))
    assert entry.source is src
    assert entry.problem_id == "main"


def test_sequential_ids() -> None:
    registry = SourceRegistry()
    s1 = asyncio.run(registry.add(_make_source("A")))
    s2 = asyncio.run(registry.add(_make_source("B")))
    assert s1 == "S1"
    assert s2 == "S2"
    assert len(registry) == 2


def test_get_unknown_raises() -> None:
    registry = SourceRegistry()
    with pytest.raises(KeyError):
        asyncio.run(registry.get("S99"))


def test_get_full_content() -> None:
    registry = SourceRegistry()
    src = _make_source()
    asyncio.run(registry.add(src))
    content = asyncio.run(registry.get_full_content("S1"))
    assert content is src


def test_list_all() -> None:
    registry = SourceRegistry()
    asyncio.run(registry.add(_make_source("A"), problem_id="main"))
    asyncio.run(registry.add(_make_source("B"), problem_id="sub1"))
    sources = asyncio.run(registry.list_sources())
    assert len(sources) == 2
    assert sources[0]["source_id"] == "S1"
    assert sources[1]["source_id"] == "S2"


def test_list_filter_by_problem_id() -> None:
    registry = SourceRegistry()
    asyncio.run(registry.add(_make_source("A"), problem_id="main"))
    asyncio.run(registry.add(_make_source("B"), problem_id="sub1"))
    asyncio.run(registry.add(_make_source("C"), problem_id="main"))

    main_sources = asyncio.run(registry.list_sources(problem_id="main"))
    assert len(main_sources) == 2
    assert {s["source_id"] for s in main_sources} == {"S1", "S3"}

    sub_sources = asyncio.run(registry.list_sources(problem_id="sub1"))
    assert len(sub_sources) == 1
    assert sub_sources[0]["source_id"] == "S2"


def test_max_gist_chars_truncation() -> None:
    registry = SourceRegistry(max_gist_chars=5)
    src = _make_source(snippet="abcdefghij")
    asyncio.run(registry.add(src))
    entry = asyncio.run(registry.get("S1"))
    assert entry.gist == "abcde"
    assert entry.source.snippet == "abcdefghij"  # full content preserved


def test_serialize() -> None:
    registry = SourceRegistry()
    asyncio.run(registry.add(_make_source("A"), problem_id="main"))
    asyncio.run(registry.add(_make_source("B"), problem_id="sub1"))
    data = registry.serialize()
    assert set(data.keys()) == {"S1", "S2"}
    assert data["S1"]["problem_id"] == "main"
    assert data["S2"]["problem_id"] == "sub1"
