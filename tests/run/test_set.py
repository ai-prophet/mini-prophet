"""Tests for miniprophet.run.set CLI command."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import typer

from miniprophet.run.set import main


def test_set_key_value_saves_new(monkeypatch, tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    monkeypatch.setattr("miniprophet.run.set.global_config_file", env_file)

    with patch("miniprophet.run.set.save_env_var") as mock_save, \
         patch("miniprophet.run.set.read_env_vars", return_value={}):
        main(key="MY_KEY", value="my_value", interactive=False)
        mock_save.assert_called_once_with(env_file, "MY_KEY", "my_value")


def test_set_key_value_updates_existing(monkeypatch, tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    monkeypatch.setattr("miniprophet.run.set.global_config_file", env_file)

    with patch("miniprophet.run.set.save_env_var"), \
         patch("miniprophet.run.set.read_env_vars", return_value={"MY_KEY": "old"}):
        main(key="MY_KEY", value="new_value", interactive=False)


def test_set_missing_key_or_value_raises(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("miniprophet.run.set.global_config_file", tmp_path / ".env")
    with pytest.raises(typer.Exit):
        main(key=None, value=None, interactive=False)


def test_set_invalid_key_raises(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("miniprophet.run.set.global_config_file", tmp_path / ".env")
    with pytest.raises(typer.Exit):
        main(key="123invalid", value="v", interactive=False)


def test_set_interactive_with_key_raises(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("miniprophet.run.set.global_config_file", tmp_path / ".env")
    with pytest.raises(typer.Exit):
        main(key="KEY", value=None, interactive=True)


def test_set_interactive_no_changes(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("miniprophet.run.set.global_config_file", tmp_path / ".env")
    with patch("miniprophet.run.set.prompt_and_save_env_vars", return_value={}):
        main(key=None, value=None, interactive=True)


def test_set_interactive_with_changes(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("miniprophet.run.set.global_config_file", tmp_path / ".env")
    with patch("miniprophet.run.set.prompt_and_save_env_vars", return_value={"A": "1"}):
        main(key=None, value=None, interactive=True)
