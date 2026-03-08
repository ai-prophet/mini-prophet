from __future__ import annotations

import json
from pathlib import Path

import pytest

from miniprophet.eval.datasets.registry import (
    RegistryDataset,
    RegistryDatasetSpec,
    load_registry,
    resolve_latest_version,
    resolve_registry_dataset,
)


def test_resolve_latest_version_prefers_latest() -> None:
    assert resolve_latest_version(["2026-02-20", "latest", "1.2.0"]) == "latest"


def test_resolve_latest_version_prefers_date_then_semver() -> None:
    assert resolve_latest_version(["2026-02-28", "2026-03-01", "1.9.0"]) == "2026-03-01"
    assert resolve_latest_version(["1.2.0", "v2.0.0", "alpha"]) == "v2.0.0"


def test_registry_dataset_spec_name_disallows_slash() -> None:
    with pytest.raises(ValueError, match="cannot contain '/'"):
        RegistryDatasetSpec(
            name="alice/dataset",
            version="latest",
            description="",
            git_url="https://example.com/repo.git",
            git_ref="main",
            path="dataset.jsonl",
        )


def test_registry_dataset_group_name_disallows_slash() -> None:
    with pytest.raises(ValueError, match="cannot contain '/'"):
        RegistryDataset.model_validate(
            {
                "name": "alice/dataset",
                "description": "x",
                "latest": "2026-03-01",
                "versions": [
                    {
                        "version": "2026-03-01",
                        "git_url": "https://example.com/repo.git",
                        "git_ref": "main",
                        "path": "datasets/alice/2026-03-01/tasks.jsonl",
                    }
                ],
            }
        )


def test_load_grouped_registry_and_resolve_latest(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    path.write_text(
        json.dumps(
            {
                "datasets": [
                    {
                        "name": "dummy",
                        "description": "Dummy dataset",
                        "latest": "2026-03-02",
                        "versions": [
                            {
                                "version": "2026-03-01",
                                "git_url": "https://github.com/ai-prophet/ai-prophet-datasets.git",
                                "git_ref": "sha1",
                                "path": "datasets/dummy/2026-03-01/tasks.jsonl",
                            },
                            {
                                "version": "2026-03-02",
                                "git_url": "https://github.com/ai-prophet/ai-prophet-datasets.git",
                                "git_ref": "sha2",
                                "path": "datasets/dummy/2026-03-02/tasks.jsonl",
                            },
                        ],
                    }
                ]
            }
        )
    )

    registry = load_registry(registry_path=path)
    spec = resolve_registry_dataset(registry, name="dummy", version=None)
    assert spec.version == "2026-03-02"
    assert spec.path == "datasets/dummy/2026-03-02/tasks.jsonl"
    assert spec.description == "Dummy dataset"


def test_load_legacy_registry_and_resolve_latest_alias(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    path.write_text(
        json.dumps(
            {
                "datasets": [
                    {
                        "name": "dummy",
                        "version": "2026-03-02",
                        "description": "Dummy dataset",
                        "git_url": "https://github.com/ai-prophet/ai-prophet-datasets.git",
                        "git_ref": "sha2",
                        "path": "datasets/dummy/2026-03-02/tasks.jsonl",
                    },
                    {
                        "name": "dummy",
                        "version": "latest",
                        "description": "Dummy latest alias",
                        "git_url": "https://github.com/ai-prophet/ai-prophet-datasets.git",
                        "git_ref": "sha2",
                        "path": "datasets/dummy/2026-03-02/tasks.jsonl",
                    },
                ]
            }
        )
    )

    registry = load_registry(registry_path=path)
    spec = resolve_registry_dataset(registry, name="dummy", version="latest")
    assert spec.version == "latest"
    assert spec.description == "Dummy latest alias"
