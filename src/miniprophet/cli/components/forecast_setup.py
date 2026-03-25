"""Interactive forecast setup TUI component."""

from __future__ import annotations

from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from miniprophet.cli.utils import get_console

console = get_console()


def _display_current(title: str, ground_truth: dict[str, int] | None) -> None:
    table = Table(show_header=False, expand=False, box=None, padding=(0, 2))
    table.add_column(style="bold")
    table.add_column()
    table.add_row(
        "Title",
        title if title else "[dim](Empty: awaiting input...)[/dim]",
    )
    if ground_truth is not None:
        gt_str = ", ".join(f"{k}={v}" for k, v in ground_truth.items())
        table.add_row("Ground Truth", gt_str)
    else:
        table.add_row("Ground Truth", "[dim](Optional: not set)[/dim]")
    console.print(
        Panel(table, title="[bold cyan]Forecast Parameters[/bold cyan]", border_style="cyan")
    )


def _edit_ground_truth(ground_truth: dict[str, int] | None) -> dict[str, int] | None:
    """Prompt for binary ground truth (0 or 1). Skippable."""
    if not Confirm.ask("  Set ground truth?", default=False):
        return ground_truth

    while True:
        default_val = ""
        if ground_truth and "Yes" in ground_truth:
            default_val = str(ground_truth["Yes"])
        val = Prompt.ask("  Ground truth: did Yes happen? (0 or 1)", default=default_val).strip()
        if val in ("0", "1"):
            v = int(val)
            return {"Yes": v, "No": 1 - v}
        console.print("  [red]Must be 0 or 1.[/red]")


def prompt_forecast_params(
    prefill_title: str = "",
    prefill_ground_truth: dict[str, int] | None = None,
) -> tuple[str, dict[str, int] | None]:
    """Interactive TUI for entering/editing forecast parameters.

    Returns (title, ground_truth_or_none).
    """
    title = prefill_title
    ground_truth = dict(prefill_ground_truth) if prefill_ground_truth else None

    console.print()
    console.rule("[bold cyan]Forecast Setup[/bold cyan]", style="cyan")

    _display_current(title, ground_truth)
    console.print()

    title = Prompt.ask("  [bold]Title[/bold]", default=title or "").strip()

    ground_truth = _edit_ground_truth(ground_truth)

    console.print()
    _display_current(title, ground_truth)

    if not Confirm.ask("  [bold]Confirm and run?[/bold]", default=True):
        return prompt_forecast_params(title, ground_truth)

    return title, ground_truth
