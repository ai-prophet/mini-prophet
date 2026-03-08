from __future__ import annotations

import pytest

from miniprophet.environment.source_board import Source, SourceBoard


def test_source_board_add_and_render_with_reaction() -> None:
    board = SourceBoard()
    src = Source(url="https://x", title="T", snippet="S", date="2026-02-01")

    entry = board.add(src, "important", reaction={"Yes": "positive"}, source_id="S1")

    assert entry.id == 1
    rendered = board.render()
    assert '<source board_id="1" title="T" url="https://x" date="2026-02-01">' in rendered
    assert "Reactions: Yes [+]" in rendered


def test_source_board_edit_note_updates_existing_entry() -> None:
    board = SourceBoard()
    board.add(Source(url="u", title="t", snippet="s"), "old")

    updated = board.edit_note(1, "new", reaction={"No": "negative"})

    assert updated.note == "new"
    assert board.get(1).reaction == {"No": "negative"}


def test_source_board_get_raises_for_missing_id() -> None:
    board = SourceBoard()
    with pytest.raises(KeyError, match="No board entry"):
        board.get(999)
