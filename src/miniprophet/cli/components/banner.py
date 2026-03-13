"""ASCII branding banner for CLI startup."""

from __future__ import annotations

from miniprophet import global_config_file
from miniprophet.cli.utils import get_console

console = get_console()

_PROJECT_WORDMARK = r"""
 __  __ _____ _   _ _____   _____  _____   ____  _____  _    _ ______ _______
|  \/  |_   _| \ | |_   _| |  __ \|  __ \ / __ \|  __ \| |  | |  ____|__   __|
| \  / | | | |  \| | | |   | |__) | |__) | |  | | |__) | |__| | |__     | |
| |\/| | | | | . ` | | |   |  ___/|  _  /| |  | |  ___/|  __  |  __|    | |
| |  | |_| |_| |\  |_| |_  | |    | | \ \| |__| | |    | |  | | |____   | |
|_|  |_|_____|_| \_|_____| |_|    |_|  \_\\____/|_|    |_|  |_|______|  |_|
"""


def print_cli_banner(version: str, *, mode_label: str | None = None) -> None:
    """Render the startup icon + project wordmark for CLI usage."""
    console.print()
    console.print(_PROJECT_WORDMARK, style="bold bright_blue", highlight=False)

    subtitle = f"v{version}"
    if mode_label:
        subtitle = f"{subtitle} | {mode_label}"
    console.print(f"[dim]{subtitle} | minimal LLM forecasting agent[/dim]\n")
    console.print(f"Loading global config from: [bold green]{global_config_file}[/bold green]")
    console.print(
        "[dim]To change the global config, set the [bold yellow]`MINIPROPHET_GLOBAL_CONFIG_DIR`[/bold yellow] env variable.[/dim]\n"
    )


def print_run_info(*, model_class: str, model_name: str, search_class: str) -> None:
    """Display the active model and searcher after config resolution."""
    console.print(f"  Model:    [bold cyan]{model_name} (via {model_class})[/bold cyan]")
    console.print(f"  Searcher: [bold cyan]{search_class}[/bold cyan]\n")
