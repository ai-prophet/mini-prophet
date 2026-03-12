"""Step header and model response display components."""

from __future__ import annotations

import json

from rich.panel import Panel

from miniprophet.cli.utils import get_console

console = get_console()


def _format_token_count(n: int) -> str:
    """Format large token counts with k/M suffix for readability."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def print_step_header(
    step: int,
    model_cost: float,
    search_cost: float,
    total_cost: float,
    *,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    max_context_tokens: int | None = None,
) -> None:
    cost_str = f"model=${model_cost:.4f}  search=${search_cost:.4f}  total=${total_cost:.4f}"
    parts = [f"[bold]Step {step}[/bold]  |  {cost_str}"]

    if prompt_tokens > 0:
        ctx_used = _format_token_count(prompt_tokens)
        if max_context_tokens:
            ctx_max = _format_token_count(max_context_tokens)
            pct = prompt_tokens / max_context_tokens * 100
            parts.append(f"context={ctx_used}/{ctx_max} ({pct:.0f}%)")
        else:
            parts.append(f"context={ctx_used}")
        if completion_tokens > 0:
            parts.append(f"completion={_format_token_count(completion_tokens)}")

    console.rule("  ".join(parts), style="cyan")


def print_model_response(message: dict, *, max_thinking_chars: int = 500) -> None:
    content = message.get("content") or ""
    actions = message.get("extra", {}).get("actions", [])

    if content:
        truncated = content[:max_thinking_chars] + (
            "..." if len(content) > max_thinking_chars else ""
        )
        console.print(
            Panel(
                truncated,
                title="[bold yellow]Model Thinking[/bold yellow]",
                border_style="yellow",
                expand=False,
            )
        )

    for action in actions:
        name = action.get("name", "?")
        raw_args = action.get("arguments", "{}")
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            args_display = json.dumps(args, indent=2, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            args_display = str(raw_args)

        if len(args_display) > 300:
            args_display = args_display[:300] + "\n..."

        console.print(
            Panel(
                f"[bold]{name}[/bold]({args_display})",
                title="[bold green]Tool Call[/bold green]",
                border_style="green",
                expand=False,
            )
        )
