from __future__ import annotations

from miniprophet.agent.context import SlidingWindowContextManager


def test_sliding_window_context_no_truncation() -> None:
    mgr = SlidingWindowContextManager(window_size=4)
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
    ]

    out = mgr.manage(messages, step=1)
    assert out == messages


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
