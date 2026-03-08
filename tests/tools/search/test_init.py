from __future__ import annotations

import types

import pytest

from miniprophet.tools.search import get_search_backend


class _KwOnlySearch:
    def __init__(self, keep: int = 0) -> None:
        self.keep = keep

    def search(self, query: str, limit: int = 5, **kwargs):
        raise NotImplementedError

    def serialize(self) -> dict:
        return {}


class _VarKwSearch:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def search(self, query: str, limit: int = 5, **kwargs):
        raise NotImplementedError

    def serialize(self) -> dict:
        return {}


@pytest.fixture
def patch_search_module(monkeypatch: pytest.MonkeyPatch):
    mod = types.ModuleType("_tests_dummy_search_module")
    mod.KwOnlySearch = _KwOnlySearch
    mod.VarKwSearch = _VarKwSearch

    import sys

    sys.modules[mod.__name__] = mod
    yield mod


def test_get_search_tool_filters_kwargs_without_varkw(patch_search_module) -> None:
    class_path = f"{patch_search_module.__name__}.KwOnlySearch"
    tool = get_search_backend(
        {
            "search_class": class_path,
            class_path: {"keep": 3, "drop": 9},
        }
    )
    assert isinstance(tool, _KwOnlySearch)
    assert tool.keep == 3


def test_get_search_tool_passes_all_kwargs_with_varkw(patch_search_module) -> None:
    class_path = f"{patch_search_module.__name__}.VarKwSearch"
    tool = get_search_backend(
        {
            "search_class": class_path,
            class_path: {"a": 1, "b": 2},
        }
    )
    assert isinstance(tool, _VarKwSearch)
    assert tool.kwargs == {"a": 1, "b": 2}


def test_get_search_tool_raises_on_unknown_class() -> None:
    with pytest.raises(ValueError, match="Unknown search class"):
        get_search_backend({"search_class": "missing.module.Class"})
