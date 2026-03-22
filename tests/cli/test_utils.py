from __future__ import annotations

from miniprophet.cli.utils import format_token_summary, get_console


def test_get_console_is_singleton() -> None:
    assert get_console() is get_console()


def test_format_token_summary_with_cache_rate() -> None:
    result = format_token_summary(
        1000,
        50,
        cached_tokens=800,
        total_prompt_tokens=5000,
        total_cached_tokens=3500,
    )
    assert "cached=800" in result
    assert "cache_rate=70%" in result


def test_format_token_summary_no_cache() -> None:
    result = format_token_summary(1000, 50)
    assert "cached" not in result
    assert "cache_rate" not in result


def test_format_token_summary_cached_none_skipped() -> None:
    result = format_token_summary(1000, 50, cached_tokens=None)
    assert "cached" not in result
