"""CLI entry point for mini-prophet."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.prompt import Confirm, Prompt

from miniprophet import __version__
from miniprophet.cli.components.banner import print_cli_banner, print_run_info
from miniprophet.cli.utils import get_console
from miniprophet.utils.serialize import UNSET, recursive_merge

app = typer.Typer(
    name="run",
    help="Run a single forecast (interactive or CLI args).",
    add_completion=False,
)
console = get_console()


@app.callback(invoke_without_command=True)
def main(
    title: str | None = typer.Option(None, "--title", "-t", help="The forecasting question."),
    ground_truth_json: str | None = typer.Option(
        None,
        "--ground-truth",
        "-g",
        help="Ground truth: 0 or 1, or JSON like '{\"Yes\": 1}'.",
    ),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Enter interactive mode."),
    model_name: str | None = typer.Option(
        None, "--model", "-m", help="Model name (e.g. openai/gpt-4o-mini)."
    ),
    cost_limit: float | None = typer.Option(
        None, "--cost-limit", "-l", help="Total cost limit in USD."
    ),
    search_limit: int | None = typer.Option(
        None, "--search-limit", help="Max number of search queries."
    ),
    step_limit: int | None = typer.Option(None, "--step-limit", help="Max number of agent steps."),
    config_spec: list[str] | None = typer.Option(
        None, "--config", "-c", help="Config file(s) or key=value overrides."
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Output directory for run artifacts (info.json + trajectory.json + sources.json).",
    ),
    model_class: str | None = typer.Option(
        None, "--model-class", help="Override model class (e.g. openrouter, litellm)."
    ),
    search_class: str | None = typer.Option(
        None, "--search-class", help="Override search class (e.g. perplexity, exa, brave)."
    ),
    disable_history: bool = typer.Option(
        False, "--disable-history", help="Skip writing to forecast history."
    ),
    disable_interrupt: bool = typer.Option(
        False, "--disable-interrupt", help="Disable Ctrl+C pause-and-inject during agent run."
    ),
) -> None:
    """Run the forecasting agent on a binary yes/no question."""
    from miniprophet.agent.cli_agent import CliForecastAgent
    from miniprophet.agent.context import get_context_manager
    from miniprophet.config import get_config_from_spec
    from miniprophet.environment.forecast_env import ForecastEnvironment, create_default_tools
    from miniprophet.environment.source_board import SourceBoard
    from miniprophet.models import get_model
    from miniprophet.tools.search import get_search_backend
    from miniprophet.utils.metrics import normalize_ground_truth

    print_cli_banner(__version__, mode_label="single run")
    # ---- Load and merge configs ----
    configs = [get_config_from_spec("default")]
    for spec in config_spec or []:
        configs.append(get_config_from_spec(spec))

    configs.append(
        {
            "agent": {
                "cost_limit": cost_limit or UNSET,
                "search_limit": search_limit or UNSET,
                "step_limit": step_limit or UNSET,
                "output_path": str(output) if output else UNSET,
                "enable_interrupt": False if disable_interrupt else UNSET,
            },
            "model": {
                "model_name": model_name or UNSET,
                "model_class": model_class or UNSET,
            },
            "search": {
                "search_class": search_class or UNSET,
            },
        }
    )

    config = recursive_merge(*configs)

    model_cfg = config.get("model", {})
    search_cfg_top = config.get("search", {})
    print_run_info(
        model_class=model_cfg.get("model_class", "litellm"),
        model_name=model_cfg.get("model_name", ""),
        search_class=search_cfg_top.get("search_class", "perplexity"),
    )

    # ---- Resolve title and ground_truth ----
    ground_truth: dict[str, int] | None = None
    if ground_truth_json:
        try:
            raw = json.loads(ground_truth_json)
        except json.JSONDecodeError:
            # Accept bare 0 or 1
            if ground_truth_json.strip() in ("0", "1"):
                raw = int(ground_truth_json.strip())
            else:
                console.print(
                    f"[bold red]Error:[/bold red] Invalid --ground-truth: {ground_truth_json}"
                )
                raise typer.Exit(1)
        ground_truth = normalize_ground_truth(raw)

    if interactive:
        resolved_title, ground_truth = _interactive_flow(
            prefill_title=title or "",
            prefill_ground_truth=ground_truth,
        )
    else:
        if not title:
            console.print(
                "[bold red]Error:[/bold red] --title is required (or use --interactive / -i)."
            )
            raise typer.Exit(1)
        resolved_title = title

    # ---- Forecast loop (re-enter setup in interactive mode) ----
    while True:
        model = get_model(config=config.get("model", {}))

        search_cfg = config.get("search", {})
        search_backend = get_search_backend(search_cfg=search_cfg)

        agent_search_limit = config.get("agent", {}).get("search_limit", 10)
        board = SourceBoard()
        tools = create_default_tools(
            search_tool=search_backend,
            board=board,
            search_limit=agent_search_limit,
            search_results_limit=search_cfg.get("search_results_limit", 5),
            max_source_display_chars=search_cfg.get("max_source_display_chars", 2000),
        )
        env = ForecastEnvironment(tools, board=board)

        cm_cfg = config.get("context_manager", {})
        ctx_mgr = get_context_manager(cm_cfg)
        agent = CliForecastAgent(
            model=model, env=env, context_manager=ctx_mgr, **config.get("agent", {})
        )

        runtime_kwargs = {}
        if search_cfg.get("search_date_before", None):
            runtime_kwargs["search_date_before"] = search_cfg["search_date_before"]
        if search_cfg.get("search_date_after", None):
            runtime_kwargs["search_date_after"] = search_cfg["search_date_after"]

        result = agent.run_sync(title=resolved_title, ground_truth=ground_truth, **runtime_kwargs)

        if result.get("submission") and not disable_history:
            from miniprophet.cli.components.forecast_history import append_history

            model_cfg = config.get("model", {})
            append_history(
                title=resolved_title,
                ground_truth=ground_truth,
                submission=result["submission"],
                model_name=model_cfg.get("model_name", ""),
                model_class=model_cfg.get("model_class", ""),
            )

        if not result.get("submission"):
            console.print(
                f"\n[bold yellow]Agent exited without submitting.[/bold yellow] "
                f"Status: {result.get('exit_status', 'unknown')}"
            )

        if not interactive:
            break

        console.print()
        if not Confirm.ask("  [bold]Run another forecast?[/bold]", default=False):
            break

        resolved_title, ground_truth = _interactive_flow(
            prefill_title=resolved_title,
            prefill_ground_truth=ground_truth,
        )


def _interactive_flow(
    prefill_title: str,
    prefill_ground_truth: dict[str, int] | None,
) -> tuple[str, dict[str, int] | None]:
    """Run the interactive TUI flow using manual setup or market services."""
    from miniprophet.cli.components.forecast_setup import prompt_forecast_params

    console.print()
    choice = Prompt.ask(
        "  [bold]How would you like to set up the forecast?[/bold]\n"
        "  [cyan]1[/cyan] Manual input\n"
        "  [cyan]2[/cyan] Kalshi ticker (auto-detect event/market)\n"
        "  [cyan]3[/cyan] Polymarket identifier (auto-detect event/market)\n"
        "  [cyan]4[/cyan] From historical forecasts\n"
        "  Choose",
        choices=["1", "2", "3", "4"],
        default="1",
    )

    if choice == "2":
        prefill_title, _, prefill_ground_truth = _fetch_kalshi()
    elif choice == "3":
        prefill_title, _, prefill_ground_truth = _fetch_polymarket()
    elif choice == "4":
        result = _browse_history()
        if result is not None:
            prefill_title, prefill_ground_truth = result

    return prompt_forecast_params(
        prefill_title=prefill_title,
        prefill_ground_truth=prefill_ground_truth,
    )


def _browse_history() -> tuple[str, dict[str, int] | None] | None:
    """Browse forecast history and return selected entry data or None."""
    from miniprophet.cli.components.forecast_history import browse_history_interactive

    return browse_history_interactive()


def _fetch_kalshi() -> tuple[str, list[str], dict[str, int] | None]:
    """Prompt for a Kalshi ticker and auto-detect event vs market."""
    from miniprophet.run.services import get_market_service

    ticker = Prompt.ask("  [bold]Kalshi ticker (event or market)[/bold]").strip()
    if not ticker:
        console.print("  [red]No ticker provided.[/red]")
        return "", [], None

    try:
        service = get_market_service("kalshi")
        data = service.fetch(ticker, ticker_type="auto")
    except Exception as exc:
        console.print(f"  [bold red]Failed to fetch market:[/bold red] {exc}")
        return "", [], None

    entity = data.metadata.get("entity", "market")
    console.print(f"  [green]Fetched Kalshi {entity}:[/green] {data.title}")
    if data.metadata.get("last_price"):
        console.print(
            f"  [dim]Last price: {data.metadata['last_price']}  Volume: {data.metadata.get('volume', '?')}[/dim]"
        )

    return data.title, data.outcomes, data.ground_truth


def _fetch_polymarket() -> tuple[str, list[str], dict[str, int] | None]:
    """Prompt for a Polymarket identifier and auto-detect event vs market."""
    from miniprophet.run.services import get_market_service

    identifier = Prompt.ask("  [bold]Polymarket identifier (id or slug)[/bold]").strip()
    if not identifier:
        console.print("  [red]No identifier provided.[/red]")
        return "", [], None

    try:
        service = get_market_service("polymarket")
        data = service.fetch(identifier, entity="auto", identifier_type="auto")
    except Exception as exc:
        console.print(f"  [bold red]Failed to fetch market:[/bold red] {exc}")
        return "", [], None

    entity = data.metadata.get("entity", "market")
    console.print(f"  [green]Fetched Polymarket {entity}:[/green] {data.title}")
    if data.metadata.get("last_price") != "":
        console.print(
            f"  [dim]Last price: {data.metadata.get('last_price', '?')}  "
            f"Volume: {data.metadata.get('volume', '?')}[/dim]"
        )

    return data.title, data.outcomes, data.ground_truth


if __name__ == "__main__":
    app()
