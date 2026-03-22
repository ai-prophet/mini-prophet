"""Run header and footer display components."""

from __future__ import annotations

from miniprophet.cli.utils import format_token_summary, get_console

console = get_console()


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


def print_run_footer(
    exit_status: str,
    n_calls: int,
    n_searches: int,
    model_cost: float,
    search_cost: float,
    total_cost: float,
    *,
    prompt_tokens: int = 0,
    max_context_tokens: int | None = None,
    total_prompt_tokens: int = 0,
    total_cached_tokens: int = 0,
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
        lines.append(
            f"  [bold]Tokens:[/bold]   "
            f"{format_token_summary(prompt_tokens, 0, max_context_tokens, total_prompt_tokens=total_prompt_tokens, total_cached_tokens=total_cached_tokens)}"
        )
    console.print("\n".join(lines))
    console.print()
