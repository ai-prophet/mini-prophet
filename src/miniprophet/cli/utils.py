"""Utility functions for CLI components."""

from __future__ import annotations

from rich.console import Console


class ConsoleProxy:
    """Delegating wrapper around a Rich Console.

    All attribute access is forwarded to an inner ``_target`` which defaults
    to a normal :class:`Console`.  Call :func:`set_console_target` to swap the
    target (e.g. to a :class:`TuiConsole` that writes into a Textual widget).

    Python looks up dunder/special methods on the *type*, not the instance,
    so ``__getattr__`` cannot intercept them.  We explicitly delegate the ones
    that Rich's ``Console`` (and ``Live``) rely on.
    """

    def __init__(self) -> None:
        object.__setattr__(self, "_target", Console())

    def __getattr__(self, name: str):
        return getattr(self._target, name)

    def __setattr__(self, name: str, value):
        if name == "_target":
            object.__setattr__(self, name, value)
        else:
            setattr(self._target, name, value)

    # -- dunder methods that must be forwarded explicitly --

    def __enter__(self):
        return self._target.__enter__()

    def __exit__(self, *args):
        return self._target.__exit__(*args)

    def __repr__(self) -> str:
        return repr(self._target)


_console: ConsoleProxy | None = None


def get_console() -> ConsoleProxy:
    """Return the shared console singleton (a proxy with a swappable target)."""
    global _console
    if _console is None:
        _console = ConsoleProxy()
    return _console


def set_console_target(target) -> None:
    """Swap the inner target of the global console proxy."""
    proxy = get_console()
    object.__setattr__(proxy, "_target", target)


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
    *,
    cached_tokens: int | None = None,
    total_prompt_tokens: int = 0,
    total_cached_tokens: int = 0,
) -> str:
    """Format a token usage summary string, e.g. ``context=45.3k/200.0k (23%)  completion=1.2k  cached=3.0k  cache_rate=70%``."""
    ctx_used = format_token_count(prompt_tokens)
    if max_context_tokens:
        ctx_max = format_token_count(max_context_tokens)
        pct = prompt_tokens / max_context_tokens * 100
        token_str = f"context={ctx_used}/{ctx_max} ({pct:.0f}%)"
    else:
        token_str = f"context={ctx_used}"
    if completion_tokens > 0:
        token_str += f"  completion={format_token_count(completion_tokens)}"
    if cached_tokens is not None:
        token_str += f"  cached={format_token_count(cached_tokens)}"
    if total_prompt_tokens > 0 and total_cached_tokens > 0:
        rate = total_cached_tokens / total_prompt_tokens * 100
        token_str += f"  cache_rate={rate:.0f}%"
    return token_str
