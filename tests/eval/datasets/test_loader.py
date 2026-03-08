from __future__ import annotations

from miniprophet.eval.datasets.loader import DatasetSourceKind, parse_dataset_ref


def test_parse_dataset_ref_registry() -> None:
    parsed = parse_dataset_ref("weekly-nba@2026-03-01")
    assert parsed.kind == DatasetSourceKind.REGISTRY
    assert parsed.name == "weekly-nba"
    assert parsed.version == "2026-03-01"


def test_parse_dataset_ref_hf() -> None:
    parsed = parse_dataset_ref("alice/weekly-forecasts@main")
    assert parsed.kind == DatasetSourceKind.HF
    assert parsed.hf_repo == "alice/weekly-forecasts"
    assert parsed.hf_revision == "main"
