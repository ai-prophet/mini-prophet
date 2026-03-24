"""Tests for miniprophet.utils.log module."""

from __future__ import annotations

import logging
from pathlib import Path


def test_add_file_handler_creates_file_handler(tmp_path: Path) -> None:
    from miniprophet.utils.log import add_file_handler

    log_file = tmp_path / "test.log"
    add_file_handler(log_file, print_path=False)

    logger = logging.getLogger("miniprophet")
    # Find the handler we just added
    file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
    assert any(str(log_file) in str(getattr(h, "baseFilename", "")) for h in file_handlers)

    # Clean up
    for h in file_handlers:
        if str(log_file) in str(getattr(h, "baseFilename", "")):
            logger.removeHandler(h)
            h.close()


def test_add_file_handler_prints_path(tmp_path: Path, capsys) -> None:
    from miniprophet.utils.log import add_file_handler

    log_file = tmp_path / "print_test.log"
    add_file_handler(log_file, print_path=True)

    captured = capsys.readouterr()
    assert str(log_file) in captured.out

    # Clean up
    logger = logging.getLogger("miniprophet")
    for h in list(logger.handlers):
        if isinstance(h, logging.FileHandler) and str(log_file) in str(
            getattr(h, "baseFilename", "")
        ):
            logger.removeHandler(h)
            h.close()
