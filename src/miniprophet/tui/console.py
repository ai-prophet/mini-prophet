"""TuiConsole: Rich-compatible writer that redirects output to a Textual RichLog."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from rich.console import RenderableType
from rich.rule import Rule
from rich.text import Text

if TYPE_CHECKING:
    from textual.app import App
    from textual.widgets import RichLog


class TuiConsole:
    """Drop-in replacement for the subset of :class:`rich.console.Console`
    used by mini-prophet display components.

    Calls to :meth:`print` and :meth:`rule` are forwarded to a Textual
    :class:`RichLog` widget via its :meth:`~RichLog.write` method.

    Thread-safe: if called from a background thread (e.g. an agent worker),
    writes are dispatched to the Textual main thread via
    :meth:`App.call_from_thread`.
    """

    def __init__(self, rich_log: RichLog, app: App) -> None:
        self._log = rich_log
        self._app = app
        self._main_thread = threading.current_thread()

    def _write(self, renderable: RenderableType) -> None:
        if threading.current_thread() is self._main_thread:
            self._log.write(renderable)
        else:
            self._app.call_from_thread(self._log.write, renderable)

    # ------------------------------------------------------------------
    # Console-compatible API
    # ------------------------------------------------------------------

    def print(self, *args, style: str | None = None, **_kwargs) -> None:
        if not args:
            self._write(Text(""))
            return
        for renderable in args:
            if isinstance(renderable, str):
                if style:
                    self._write(Text.from_markup(renderable, style=style))
                else:
                    self._write(Text.from_markup(renderable))
            else:
                self._write(renderable)

    def rule(self, title: str = "", *, style: str = "") -> None:
        self._write(Rule(title=title, style=style))
