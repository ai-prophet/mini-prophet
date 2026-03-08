from __future__ import annotations

from rich.console import Console

from miniprophet.eval import datasets_cli as cli
from miniprophet.eval.datasets.registry import RegistryCatalog, RegistryDataset


def _catalog() -> RegistryCatalog:
    return RegistryCatalog(
        datasets=[
            RegistryDataset.model_validate(
                {
                    "name": "dummy",
                    "description": "Dummy benchmark dataset",
                    "latest": "1.2.0",
                    "versions": [
                        {
                            "version": "1.0.0",
                            "git_url": "https://example.com/repo.git",
                            "git_ref": "sha-a",
                            "path": "datasets/dummy/1.0.0/tasks.jsonl",
                        },
                        {
                            "version": "1.1.0",
                            "git_url": "https://example.com/repo.git",
                            "git_ref": "sha-b",
                            "path": "datasets/dummy/1.1.0/tasks.jsonl",
                        },
                        {
                            "version": "1.2.0",
                            "git_url": "https://example.com/repo.git",
                            "git_ref": "sha-c",
                            "path": "datasets/dummy/1.2.0/tasks.jsonl",
                        },
                        {
                            "version": "1.3.0",
                            "git_url": "https://example.com/repo.git",
                            "git_ref": "sha-d",
                            "path": "datasets/dummy/1.3.0/tasks.jsonl",
                        },
                    ],
                }
            )
        ]
    )


def test_format_versions_preview_latest_then_top_two_and_overflow() -> None:
    dataset = _catalog().datasets[0]
    latest = cli._resolve_dataset_latest(dataset)

    preview = cli._format_versions_preview(dataset, latest)
    assert preview.splitlines() == [
        "1.2.0 (latest)",
        "1.3.0",
        "1.1.0",
        "(...1 more versions)",
    ]


def test_list_datasets_index_shows_preview_versions(monkeypatch) -> None:
    monkeypatch.setattr(cli, "load_registry", lambda **_: _catalog())
    capture_console = Console(record=True, width=160)
    monkeypatch.setattr(cli, "console", capture_console)

    cli.list_datasets(dataset_name=None, registry_path=None, registry_url=None)
    output = capture_console.export_text()

    assert "Available Datasets" in output
    assert "1.2.0 (latest)" in output
    assert "(...1 more versions)" in output


def test_list_datasets_with_name_shows_all_versions(monkeypatch) -> None:
    monkeypatch.setattr(cli, "load_registry", lambda **_: _catalog())
    capture_console = Console(record=True, width=160)
    monkeypatch.setattr(cli, "console", capture_console)

    cli.list_datasets(dataset_name="dummy", registry_path=None, registry_url=None)
    output = capture_console.export_text()

    assert "Dataset Versions:" in output
    assert "dummy" in output
    assert "Latest: 1.2.0" in output
    assert "Total versions: 4" in output
    assert output.find("1.3.0") < output.find("1.2.0") < output.find("1.1.0") < output.find("1.0.0")
