"""Tests for miniprophet.eval.datasets.hf_loader module."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from miniprophet.eval.datasets.hf_loader import parse_hf_ref, download_hf_dataset


class TestParseHfRef:
    def test_valid_ref(self) -> None:
        repo, rev = parse_hf_ref("alice/dataset@main")
        assert repo == "alice/dataset"
        assert rev == "main"

    def test_no_revision(self) -> None:
        repo, rev = parse_hf_ref("alice/dataset")
        assert repo == "alice/dataset"
        assert rev is None

    def test_empty_revision_becomes_none(self) -> None:
        repo, rev = parse_hf_ref("alice/dataset@")
        assert repo == "alice/dataset"
        assert rev is None

    def test_no_slash_raises(self) -> None:
        with pytest.raises(ValueError, match="username/dataset"):
            parse_hf_ref("just-a-name")

    def test_leading_slash_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid"):
            parse_hf_ref("/dataset@main")

    def test_trailing_slash_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid"):
            parse_hf_ref("user/@main")


class TestDownloadHfDataset:
    def test_cache_hit_skips_download(self, monkeypatch, tmp_path: Path) -> None:
        cache_file = tmp_path / "cached.jsonl"
        cache_file.write_text('{"task_id": "t1"}\n')

        monkeypatch.setattr(
            "miniprophet.eval.datasets.hf_loader.get_hf_cache_path",
            lambda repo, rev, split: cache_file,
        )

        result = download_hf_dataset("user/ds", split="train")
        assert result.cached_path == cache_file
        assert result.repo == "user/ds"

    def test_missing_datasets_raises(self, monkeypatch, tmp_path: Path) -> None:
        cache_file = tmp_path / "nonexistent.jsonl"

        monkeypatch.setattr(
            "miniprophet.eval.datasets.hf_loader.get_hf_cache_path",
            lambda repo, rev, split: cache_file,
        )
        # Make sure the 'datasets' import fails
        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "datasets":
                raise ImportError("No module named 'datasets'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        with pytest.raises(RuntimeError, match="datasets"):
            download_hf_dataset("user/ds")
