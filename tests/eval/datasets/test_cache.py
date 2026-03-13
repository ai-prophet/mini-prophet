"""Tests for miniprophet.eval.datasets.cache module."""

from __future__ import annotations

from miniprophet.eval.datasets.cache import (
    get_dataset_cache_root,
    get_hf_cache_path,
    get_registry_cache_path,
)


def test_get_dataset_cache_root_creates_dir(monkeypatch) -> None:
    root = get_dataset_cache_root()
    assert root.exists()
    assert root.name == "datasets"


def test_get_registry_cache_path(monkeypatch) -> None:
    path = get_registry_cache_path("weekly-nba", "2026-03-01")
    assert path.name == "dataset.jsonl"
    assert "registry" in str(path)
    assert "weekly-nba" in str(path)
    assert "2026-03-01" in str(path)
    assert path.parent.exists()


def test_get_hf_cache_path_replaces_slashes() -> None:
    path = get_hf_cache_path("user/dataset", "feat/branch", "train/subset")
    assert "user__dataset" in str(path)
    assert "feat__branch" in str(path)
    assert "train__subset" in str(path)
    assert path.parent.exists()


def test_get_hf_cache_path_none_revision() -> None:
    path = get_hf_cache_path("user/dataset", None, "train")
    assert "default" in str(path)
