"""Utility functions for CLI components."""

from __future__ import annotations

from rich.console import Console

_console: Console | None = None


# create a console singleton for all CLI components to share
def get_console() -> Console:
    global _console
    if _console is None:
        _console = Console()
    return _console


def format_token_count(n: int) -> str:
    """Format large token counts with k/M suffix for readability."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def format_token_summary(
    prompt_tokens: int,
    completion_tokens: int = 0,
    max_context_tokens: int | None = None,
) -> str:
    """Format a token usage summary string, e.g. ``context=45.3k/200.0k (23%)  completion=1.2k``."""
    ctx_used = format_token_count(prompt_tokens)
    if max_context_tokens:
        ctx_max = format_token_count(max_context_tokens)
        pct = prompt_tokens / max_context_tokens * 100
        token_str = f"context={ctx_used}/{ctx_max} ({pct:.0f}%)"
    else:
        token_str = f"context={ctx_used}"
    if completion_tokens > 0:
        token_str += f"  completion={format_token_count(completion_tokens)}"
    return token_str
