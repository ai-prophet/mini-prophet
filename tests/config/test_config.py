from __future__ import annotations

from pathlib import Path

import pytest

from miniprophet.config import get_config_from_spec, get_config_path


def test_get_config_from_spec_parses_key_value_json() -> None:
    cfg = get_config_from_spec("model.model_kwargs.temperature=0.2")
    assert cfg == {"model": {"model_kwargs": {"temperature": 0.2}}}


def test_get_config_path_uses_env_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    cfg = cfg_dir / "custom.yaml"
    cfg.write_text("agent:\n  step_limit: 3\n")

    monkeypatch.setenv("MINIPROPHET_CONFIG_DIR", str(cfg_dir))
    assert get_config_path("custom") == cfg


def test_get_config_path_raises_for_missing_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINIPROPHET_CONFIG_DIR", "/tmp/nonexistent-miniprophet-config-dir")
    with pytest.raises(FileNotFoundError):
        get_config_path("does-not-exist")
