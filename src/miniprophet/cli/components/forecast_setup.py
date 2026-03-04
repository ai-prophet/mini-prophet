"""Interactive forecast setup TUI component."""

from __future__ import annotations

from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from miniprophet.cli.utils import get_console

console = get_console()


def _display_current(title: str, outcomes: list[str], ground_truth: dict[str, int] | None) -> None:
    table = Table(show_header=False, expand=False, box=None, padding=(0, 2))
    table.add_column(style="bold")
    table.add_column()
    table.add_row(
        "Title",
        title if title else "[dim](Empty: awaiting input...)[/dim]",
    )
    table.add_row(
        "Outcomes",
        ", ".join(outcomes) if outcomes else "[dim](Empty: awaiting input...)[/dim]",
    )
    if ground_truth is not None:
        gt_str = ", ".join(f"{k}={v}" for k, v in ground_truth.items())
        table.add_row("Ground Truth", gt_str)
    else:
        table.add_row("Ground Truth", "[dim](Optional: not set)[/dim]")
    console.print(
        Panel(table, title="[bold cyan]Forecast Parameters[/bold cyan]", border_style="cyan")
    )


def _edit_outcomes(outcomes: list[str]) -> list[str]:
    """Interactive outcome editor: show numbered list, add/delete outcomes."""
    while True:
        if outcomes:
            for i, o in enumerate(outcomes):
                console.print(f"  [bold]{i + 1}.[/bold] {o}")
        else:
            console.print("  [dim](Empty: awaiting input...)[/dim]")

        console.print(
            "\n  [dim]Type an outcome to add (comma-separated for multiple), a number to delete, or press Enter to confirm.[/dim]"
        )
        inp = Prompt.ask("  [bold]>>>[/bold]", default="").strip()

        if not inp:
            if len(outcomes) >= 2:
                break
            console.print("  [bold red]At least 2 outcomes are required.[/bold red]")
            continue

        if inp.isdigit():
            idx = int(inp) - 1
            if 0 <= idx < len(outcomes):
                removed = outcomes.pop(idx)
                console.print(f"  [red]Removed: {removed}[/red]")
            else:
                console.print("  [red]Invalid number.[/red]")
        else:
            new_items = [item.strip() for item in inp.split(",") if item.strip()]
            for item in new_items:
                outcomes.append(item)
                console.print(f"  [green]Added: {item}[/green]")

    return outcomes


def _edit_ground_truth(
    outcomes: list[str], ground_truth: dict[str, int] | None
) -> dict[str, int] | None:
    """Prompt for ground truth values (0 or 1) per outcome. Skippable."""
    if not Confirm.ask("  Set ground truth?", default=False):
        return ground_truth

    gt: dict[str, int] = {}
    for outcome in outcomes:
        default_val = str(ground_truth.get(outcome, "")) if ground_truth else ""
        while True:
            val = Prompt.ask(f"  {outcome} (0 or 1)", default=default_val).strip()
            if val in ("0", "1"):
                gt[outcome] = int(val)
                break
            console.print("  [red]Must be 0 or 1.[/red]")
    return gt


def prompt_forecast_params(
    prefill_title: str = "",
    prefill_outcomes: list[str] | None = None,
    prefill_ground_truth: dict[str, int] | None = None,
) -> tuple[str, list[str], dict[str, int] | None]:
    """Interactive TUI for entering/editing forecast parameters.

    Returns (title, outcomes, ground_truth_or_none).
    """
    title = prefill_title
    outcomes = list(prefill_outcomes or [])
    ground_truth = dict(prefill_ground_truth) if prefill_ground_truth else None

    console.print()
    console.rule("[bold cyan]Forecast Setup[/bold cyan]", style="cyan")

    _display_current(title, outcomes, ground_truth)
    console.print()

    title = Prompt.ask("  [bold]Title[/bold]", default=title or "").strip()

    console.print("\n  [bold]Outcomes[/bold] (edit list):")
    outcomes = _edit_outcomes(outcomes)

    ground_truth = _edit_ground_truth(outcomes, ground_truth)

    console.print()
    _display_current(title, outcomes, ground_truth)

    if not Confirm.ask("  [bold]Confirm and run?[/bold]", default=True):
        return prompt_forecast_params(title, outcomes, ground_truth)

    return title, outcomes, ground_truth
