"""Eval CLI entry point for mini-prophet."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import typer

from miniprophet import __version__
from miniprophet.cli.components.banner import print_cli_banner, print_run_info
from miniprophet.cli.utils import get_console
from miniprophet.eval.datasets.loader import DatasetSourceKind, resolve_dataset_to_jsonl
from miniprophet.eval.datasets.validate import load_problems
from miniprophet.eval.runner import EvalRunArgs, load_existing_summary, run_eval_sync
from miniprophet.exceptions import BatchFatalError
from miniprophet.utils.serialize import UNSET, recursive_merge

app = typer.Typer(
    name="eval",
    help="Run forecast evaluation from JSONL or dataset references.",
    add_completion=False,
)
console = get_console()


def _parse_agent_kwargs(raw: list[str] | None) -> dict[str, Any]:
    if not raw:
        return {}
    out: dict[str, Any] = {}
    for item in raw:
        if "=" not in item:
            raise ValueError(f"Invalid --agent-kwarg '{item}' (expected key=value)")
        k, v = item.split("=", 1)
        key = k.strip()
        value = v.strip()
        if not key:
            raise ValueError(f"Invalid --agent-kwarg '{item}' (empty key)")
        try:
            out[key] = json.loads(value)
        except json.JSONDecodeError:
            out[key] = value
    return out


def _build_eval_config(
    config_spec: list[str] | None,
    max_cost_per_run: float | None,
    model_name: str | None,
    model_class: str | None,
) -> dict:
    from miniprophet.config import get_config_from_spec

    configs = [get_config_from_spec("default")]
    for spec in config_spec or []:
        configs.append(get_config_from_spec(spec))

    configs.append(
        {
            "agent": {
                "cost_limit": max_cost_per_run or UNSET,
                "output_path": UNSET,
            },
            "model": {
                "model_name": model_name or UNSET,
                "model_class": model_class or UNSET,
            },
        }
    )
    return recursive_merge(*configs)


def _resolve_resume_state(
    *,
    enabled: bool,
    output: Path,
    problems: list,
) -> tuple[list, dict, float]:
    if not enabled:
        return problems, {}, 0.0

    summary_path = output / "summary.json"
    if not summary_path.exists():
        console.print("  Resume mode: no existing summary.json found, starting fresh.")
        return problems, {}, 0.0

    try:
        resume_results, resume_total_cost = load_existing_summary(summary_path)
    except ValueError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(1) from exc

    input_task_ids = {p.task_id for p in problems}
    unexpected = sorted(set(resume_results) - input_task_ids)
    if unexpected:
        preview = ", ".join(unexpected[:10])
        suffix = " ..." if len(unexpected) > 10 else ""
        console.print(
            "[bold red]Error:[/bold red] Resume summary contains task_ids not present "
            f"in the input file: {preview}{suffix}"
        )
        raise typer.Exit(1)

    original_count = len(problems)
    filtered_problems = [p for p in problems if p.task_id not in resume_results]
    skipped = original_count - len(filtered_problems)
    console.print(
        f"  Resume mode: skipping [cyan]{skipped}[/cyan] existing run(s), "
        f"[cyan]{len(filtered_problems)}[/cyan] remaining."
    )
    return filtered_problems, resume_results, resume_total_cost


def _default_output_dir(dataset_label: str, agent_name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_dataset = dataset_label.replace("/", "__").replace("@", "_")
    safe_agent = agent_name.replace("/", "__")
    return Path("runs") / safe_agent / f"{safe_dataset}_{stamp}"


@app.callback(invoke_without_command=True)
def main(
    input_file: Path | None = typer.Option(
        None,
        "--input",
        "-f",
        help="Path to a .jsonl file with forecasting problems.",
    ),
    dataset: str | None = typer.Option(
        None,
        "--dataset",
        "-d",
        help="Dataset ref: name[@version|@latest] or username/dataset[@revision].",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output meta-directory for all run artifacts.",
    ),
    workers: int = typer.Option(1, "--workers", "-w", help="Number of parallel workers."),
    max_cost: float = typer.Option(
        0.0, "--max-cost", help="Total cost budget across all runs (0 = unlimited)."
    ),
    max_cost_per_run: float | None = typer.Option(
        None, "--max-cost-per-run", help="Per-run cost limit (overrides agent config)."
    ),
    model_name: str | None = typer.Option(None, "--model", "-m", help="Model name override."),
    model_class: str | None = typer.Option(
        None, "--model-class", help="Model class override (e.g. openrouter, litellm)."
    ),
    config_spec: list[str] | None = typer.Option(
        None, "--config", "-c", help="Config file(s) or key=value overrides."
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Resume from existing output summary: skip already-seen task_ids.",
    ),
    search_date_before: str | None = typer.Option(
        None, "--search-date-before", help="Search date before (MM/DD/YYYY) for search upper bound."
    ),
    search_date_after: str | None = typer.Option(
        None, "--search-date-after", help="Search date after (MM/DD/YYYY) for search lower bound."
    ),
    offset: int = typer.Option(
        0,
        "--offset",
        help="Subtract N days from each task predict_by date before search upper bound derivation.",
    ),
    registry_path: Path | None = typer.Option(
        None,
        "--registry-path",
        help="Path to local dataset registry JSON (testing/dev).",
    ),
    registry_url: str | None = typer.Option(
        None,
        "--registry-url",
        help="URL of dataset registry JSON.",
    ),
    hf_split: str = typer.Option(
        "train",
        "--hf-split",
        help="Split to load for Hugging Face datasets.",
    ),
    overwrite_cache: bool = typer.Option(
        False,
        "--overwrite-cache",
        help="Force refresh of cached dataset artifacts.",
    ),
    agent_name: str = typer.Option(
        "default",
        "--agent",
        help="Built-in eval agent alias.",
    ),
    agent_import_path: str | None = typer.Option(
        None,
        "--agent-import-path",
        help="Import path for custom agent class: module.path:ClassName",
    ),
    agent_kwarg: list[str] | None = typer.Option(
        None,
        "--agent-kwarg",
        help="Agent kwarg in key=value format (repeatable).",
    ),
) -> None:
    """Run evaluation over forecast tasks."""
    print_cli_banner(__version__, mode_label="eval mode")

    if (input_file is None) == (dataset is None):
        console.print(
            "[bold red]Error:[/bold red] Provide exactly one of --input/-f or --dataset/-d."
        )
        raise typer.Exit(1)

    if search_date_before is not None and offset != 0:
        console.print(
            "[bold red]Error:[/bold red] Cannot combine `--search-date-before` and `--offset`."
        )
        raise typer.Exit(1)

    try:
        parsed_agent_kwargs = _parse_agent_kwargs(agent_kwarg)
    except ValueError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(1) from exc

    dataset_info: dict[str, Any] = {}

    if input_file is not None:
        if not input_file.exists():
            console.print(f"[bold red]Error:[/bold red] Input file not found: {input_file}")
            raise typer.Exit(1)
        resolved_input = input_file
        dataset_label = input_file.stem
        dataset_info = {
            "dataset_source_kind": DatasetSourceKind.LOCAL_FILE.value,
            "dataset_ref": str(input_file),
            "dataset_name": input_file.stem,
            "dataset_version_or_revision": None,
            "cached_dataset_path": str(input_file.resolve()),
        }
    else:
        assert dataset is not None
        try:
            resolved = resolve_dataset_to_jsonl(
                dataset,
                registry_path=registry_path,
                registry_url=registry_url,
                hf_split=hf_split,
                overwrite_cache=overwrite_cache,
            )
        except Exception as exc:
            console.print(f"[bold red]Error:[/bold red] Failed to resolve dataset: {exc}")
            raise typer.Exit(1) from exc

        resolved_input = resolved.path
        dataset_label = resolved.source_ref
        dataset_info = {
            "dataset_source_kind": resolved.kind.value,
            "dataset_ref": resolved.source_ref,
            "dataset_name": resolved.name,
            "dataset_version_or_revision": resolved.version,
            "cached_dataset_path": str(resolved.path),
            "dataset_metadata": resolved.metadata,
        }

    out_dir = output or _default_output_dir(dataset_label=dataset_label, agent_name=agent_name)

    try:
        problems = load_problems(resolved_input, offset=offset)
    except ValueError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(1) from exc

    if not problems:
        console.print("[bold yellow]No problems found in the input dataset.[/bold yellow]")
        raise typer.Exit(0)

    console.print(f"  Loaded [cyan]{len(problems)}[/cyan] problem(s) from {resolved_input}")
    console.print(f"  Workers: [cyan]{workers}[/cyan]  Output: [cyan]{out_dir}[/cyan]\n")

    config = _build_eval_config(config_spec, max_cost_per_run, model_name, model_class)

    model_cfg = config.get("model", {})
    search_cfg_top = config.get("search", {})
    print_run_info(
        model_class=model_cfg.get("model_class", "litellm"),
        model_name=model_cfg.get("model_name", ""),
        search_class=search_cfg_top.get("search_class", "perplexity"),
    )

    problems, resume_results, resume_total_cost = _resolve_resume_state(
        enabled=resume, output=out_dir, problems=problems
    )

    if not problems:
        console.print("[bold yellow]No remaining runs to process.[/bold yellow]")
        raise typer.Exit(0)

    eval_cfg = config.get("eval", {})
    timeout_seconds = max(0.0, float(eval_cfg.get("timeout", 180.0)))

    resolved_agent_class: type | None = None
    if agent_import_path:
        from miniprophet.eval.agent_factory import EvalAgentFactory

        try:
            resolved_agent_class = EvalAgentFactory._import_agent_class(agent_import_path)
        except ValueError as exc:
            console.print(f"[bold red]Error:[/bold red] {exc}")
            raise typer.Exit(1) from exc

    run_args = EvalRunArgs(
        output_dir=out_dir,
        config=config,
        workers=workers,
        max_cost=max_cost,
        timeout_seconds=timeout_seconds,
        initial_results=resume_results,
        initial_total_cost=resume_total_cost,
        search_date_before=search_date_before,
        search_date_after=search_date_after,
        dataset_info=dataset_info,
        agent_name=agent_name,
        agent_class=resolved_agent_class,
        agent_kwargs=parsed_agent_kwargs,
    )

    try:
        results = run_eval_sync(problems, run_args)
    except BatchFatalError as exc:
        console.print(f"\n[bold red]Eval aborted:[/bold red] {exc}")
        raise typer.Exit(1) from exc

    n_submitted = sum(1 for r in results.values() if r.status == "submitted")
    n_failed = sum(1 for r in results.values() if r.status not in ("submitted", "pending"))
    console.print(
        f"\n[bold]Eval complete:[/bold] "
        f"[green]{n_submitted}[/green] submitted, "
        f"[red]{n_failed}[/red] failed, "
        f"[cyan]{len(results)}[/cyan] total"
    )
    console.print(f"  Summary: {out_dir / 'summary.json'}")
