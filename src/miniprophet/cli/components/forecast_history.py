"""Persistent forecast history stored as JSONL."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict

from rich.prompt import Prompt
from rich.table import Table

from miniprophet import global_config_dir
from miniprophet.cli.utils import get_console

HISTORY_FILE: Path = Path(global_config_dir) / "forecast_history.jsonl"

console = get_console()


class HistoryEntry(TypedDict):
    timestamp: str
    title: str
    outcomes: list[str]
    ground_truth: dict[str, int] | None
    submission: dict[str, float]
    model_name: str
    model_class: str


def append_history(
    title: str,
    outcomes: list[str],
    ground_truth: dict[str, int] | None,
    submission: dict[str, float],
    model_name: str,
    model_class: str,
) -> None:
    """Append a single forecast entry to the history JSONL file."""
    entry: HistoryEntry = {
        "timestamp": datetime.now(UTC).isoformat(),
        "title": title,
        "outcomes": outcomes,
        "ground_truth": ground_truth,
        "submission": submission,
        "model_name": model_name,
        "model_class": model_class,
    }
    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def load_history() -> list[HistoryEntry]:
    """Load all history entries, newest first. Skips malformed lines."""
    if not HISTORY_FILE.exists():
        return []
    entries: list[HistoryEntry] = []
    with open(HISTORY_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    entries.reverse()
    return entries


PAGE_SIZE = 10


def browse_history_interactive() -> tuple[str, list[str], dict[str, int] | None] | None:
    """Show a paginated table of forecast history. Returns selected entry data or None."""
    entries = load_history()
    if not entries:
        console.print("  [dim]No forecast history found.[/dim]")
        return None

    total_pages = math.ceil(len(entries) / PAGE_SIZE)
    page = 0

    while True:
        start = page * PAGE_SIZE
        page_entries = entries[start : start + PAGE_SIZE]

        table = Table(title=f"Forecast History (page {page + 1}/{total_pages})", expand=False)
        table.add_column("#", style="bold", justify="right")
        table.add_column("Title", max_width=40)
        table.add_column("Outcomes")
        table.add_column("Ground Truth")
        table.add_column("Timestamp")

        for i, entry in enumerate(page_entries):
            global_idx = start + i + 1
            gt = entry.get("ground_truth")
            gt_str = ", ".join(f"{k}={v}" for k, v in gt.items()) if gt else "[dim]-[/dim]"
            ts = entry.get("timestamp", "")[:19].replace("T", " ")
            table.add_row(
                str(global_idx),
                entry.get("title", ""),
                ", ".join(entry.get("outcomes", [])),
                gt_str,
                ts,
            )

        console.print()
        console.print(table)

        nav_hints = []
        if page > 0:
            nav_hints.append("[cyan]q[/cyan] prev page")
        if page < total_pages - 1:
            nav_hints.append("[cyan]e[/cyan] next page")
        nav_hints.append("number to select")
        nav_hints.append("Enter to cancel")

        console.print(f"  [dim]{' | '.join(nav_hints)}[/dim]")
        inp = Prompt.ask("  [bold]>>>[/bold]", default="").strip()

        if not inp:
            return None

        if inp.lower() == "q" and page > 0:
            page -= 1
            continue
        if inp.lower() == "e" and page < total_pages - 1:
            page += 1
            continue

        if inp.isdigit():
            idx = int(inp) - 1
            if 0 <= idx < len(entries):
                selected = entries[idx]
                console.print(f"  [green]Selected:[/green] {selected.get('title', '')}")
                return (
                    selected.get("title", ""),
                    selected.get("outcomes", []),
                    selected.get("ground_truth"),
                )
            else:
                console.print("  [red]Invalid number.[/red]")
        else:
            console.print("  [red]Invalid input.[/red]")
