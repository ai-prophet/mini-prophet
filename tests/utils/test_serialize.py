from __future__ import annotations

from miniprophet.utils.serialize import UNSET, recursive_merge


def test_recursive_merge_merges_nested_and_skips_unset() -> None:
    merged = recursive_merge(
        {"a": 1, "nested": {"x": 1, "y": UNSET}},
        {"nested": {"y": 2}, "b": 3},
    )
    assert merged == {"a": 1, "b": 3, "nested": {"x": 1, "y": 2}}
