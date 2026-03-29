"""PromptInput: toad-style input widget with ❯ symbol and status bar."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Label, Static, TextArea


class PromptTextArea(TextArea):
    """TextArea with Enter->submit and Shift+Enter->newline."""

    BINDINGS = [
        Binding("enter", "submit", "Submit", priority=True),
        Binding("shift+enter", "newline", "Newline", priority=True),
    ]

    def action_submit(self) -> None:
        text = self.text.strip()
        if text:
            self.post_message(PromptInput.Submitted(text))
            self.clear()

    def action_newline(self) -> None:
        self.insert("\n")


class PromptInput(Widget):
    """Toad-style input widget: ❯ prompt + text area + status line."""

    can_focus = False  # child TextArea handles focus

    placeholder = reactive("Enter forecasting question...")
    status = reactive("Enter: submit | Ctrl+Q: quit")

    class Submitted(Message):
        """Posted when the user presses Enter with non-empty text."""

        def __init__(self, text: str) -> None:
            self.text = text
            super().__init__()

    def compose(self) -> ComposeResult:
        with Horizontal(classes="prompt-container"):
            yield Static("❯", id="prompt-symbol")
            yield PromptTextArea(id="prompt-textarea")
        yield Label(self.status, id="status-line")

    def on_mount(self) -> None:
        textarea = self.query_one("#prompt-textarea", PromptTextArea)
        textarea.placeholder = self.placeholder
        textarea.show_line_numbers = False
        textarea.focus()

    def watch_placeholder(self, value: str) -> None:
        try:
            self.query_one("#prompt-textarea", PromptTextArea).placeholder = value
        except Exception:
            pass

    def watch_status(self, value: str) -> None:
        try:
            self.query_one("#status-line", Label).update(value)
        except Exception:
            pass

    def focus_input(self) -> None:
        """Focus the text area."""
        self.query_one("#prompt-textarea", PromptTextArea).focus()
