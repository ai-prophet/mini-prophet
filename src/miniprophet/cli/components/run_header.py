"""Run header and footer display components."""

from __future__ import annotations

from rich.console import Console

console = Console()


def print_run_header(
    title: str, outcomes: str, step_limit: int, cost_limit: float, search_limit: int
) -> None:
    console.print()
    console.rule("[bold magenta]Forecasting Agent[/bold magenta]", style="magenta")
    console.print(f"  [bold]Question:[/bold] {title}")
    console.print(f"  [bold]Outcomes:[/bold] {outcomes}")
    console.print(
        f"  [bold]Limits:[/bold]   steps={step_limit}  "
        f"cost=${cost_limit:.2f}  searches={search_limit}"
    )
    console.print()


def _format_token_count(n: int) -> str:
    """Format large token counts with k/M suffix for readability."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def print_run_footer(
    exit_status: str,
    n_calls: int,
    n_searches: int,
    model_cost: float,
    search_cost: float,
    total_cost: float,
    *,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    max_context_tokens: int | None = None,
) -> None:
    console.print()
    console.rule("[bold magenta]Agent Finished[/bold magenta]", style="magenta")
    lines = [
        f"  [bold]Status:[/bold]   {exit_status}",
        f"  [bold]Steps:[/bold]    {n_calls}   Searches: {n_searches}",
        f"  [bold]Cost:[/bold]     model=${model_cost:.4f}  "
        f"search=${search_cost:.4f}  total=${total_cost:.4f}",
    ]
    if prompt_tokens > 0:
        ctx_used = _format_token_count(prompt_tokens)
        if max_context_tokens:
            ctx_max = _format_token_count(max_context_tokens)
            pct = prompt_tokens / max_context_tokens * 100
            token_str = f"context={ctx_used}/{ctx_max} ({pct:.0f}%)"
        else:
            token_str = f"context={ctx_used}"
        if completion_tokens > 0:
            token_str += f"  completion={_format_token_count(completion_tokens)}"
        lines.append(f"  [bold]Tokens:[/bold]   {token_str}")
    console.print("\n".join(lines))
    console.print()
