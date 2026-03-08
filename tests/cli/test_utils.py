from __future__ import annotations

from miniprophet.cli.utils import get_console


def test_get_console_is_singleton() -> None:
    assert get_console() is get_console()
