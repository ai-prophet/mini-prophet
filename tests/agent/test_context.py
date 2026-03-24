from __future__ import annotations

import pytest

from miniprophet.agent.context import SlidingWindowContextManager, get_context_manager


def test_sliding_window_context_truncates_and_includes_query_history() -> None:
    mgr = SlidingWindowContextManager(window_size=2)
    mgr.record_query("q1")
    mgr.record_query("q2")

    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a1"},
        {"role": "tool", "content": "t1"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "u2"},
    ]

    out = mgr.manage(messages, step=2)
    assert out[2]["extra"]["is_truncation_notice"] is True
    assert "Search Queries So Far" in out[2]["content"]
    assert "1. q1" in out[2]["content"]
    assert mgr.total_truncated == 2


def test_sliding_window_expands_for_tool_messages() -> None:
    """Window expands when the cut point would orphan a tool message."""
    mgr = SlidingWindowContextManager(window_size=2)
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a1"},
        {"role": "tool", "content": "t1"},
        {"role": "tool", "content": "t2"},
        {"role": "assistant", "content": "a2"},
    ]
    out = mgr.manage(messages, step=2)
    # Should keep at least 3 body messages to not orphan tool messages
    body = [m for m in out[2:] if not m.get("extra", {}).get("is_truncation_notice")]
    roles = [m["role"] for m in body]
    assert roles[0] != "tool"  # First kept message should not be an orphaned tool


def test_get_context_manager_invalid_class() -> None:
    with pytest.raises(ValueError, match="Unknown context manager"):
        get_context_manager({"context_manager_class": "nonexistent.module.Class"})
