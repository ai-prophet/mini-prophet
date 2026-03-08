from __future__ import annotations

from miniprophet.cli.components.env_editor import is_valid_env_key


def test_is_valid_env_key_accepts_and_rejects() -> None:
    assert is_valid_env_key("OPENAI_API_KEY") is True
    assert is_valid_env_key("1BAD") is False
