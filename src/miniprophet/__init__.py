"""
mini-prophet: A minimal LLM forecasting agent scaffolding.

Provides:
- Version numbering
- Protocol definitions for core components (Model, Environment, Tool)
"""

__version__ = "0.1.9"

import os
from pathlib import Path
from typing import Any, Protocol

from dotenv import load_dotenv
from platformdirs import user_config_dir

package_dir = Path(__file__).resolve().parent
global_dir = package_dir.parent.parent

global_config_dir = (
    Path(os.getenv("MINIPROPHET_GLOBAL_CONFIG_DIR") or user_config_dir("mini-prophet"))
    or global_dir
)
global_config_dir.mkdir(parents=True, exist_ok=True)
global_config_file = Path(global_config_dir) / ".env"

load_dotenv(dotenv_path=global_config_file)


class Model(Protocol):
    """Protocol for language models."""

    config: Any

    async def query(self, messages: list[dict], tools: list[dict]) -> dict: ...

    def format_message(self, **kwargs) -> dict: ...

    def format_observation_messages(self, message: dict, outputs: list[dict]) -> list[dict]: ...

    def serialize(self) -> dict: ...


class Tool(Protocol):
    """Protocol for modular forecast tools."""

    @property
    def name(self) -> str: ...

    def get_schema(self) -> dict: ...

    async def execute(self, args: dict) -> dict: ...

    def display(self, output: dict) -> None: ...


class Environment(Protocol):
    """Protocol for forecast environments."""

    _tools: dict[str, Tool]

    async def execute(self, action: dict, **kwargs) -> dict: ...

    def get_tool_schemas(self) -> list[dict]: ...

    def serialize(self) -> dict: ...


class ContextManager(Protocol):
    """Protocol for managing the message context between steps."""

    def manage(self, messages: list[dict], *, step: int, **kwargs) -> list[dict]: ...

    def display(self) -> None: ...


class Agent(Protocol):
    """Protocol for forecast agents."""

    async def run(
        self,
        title: str,
        outcomes: list[str],
        ground_truth: dict[str, int] | None = None,
        **kw,
    ) -> dict: ...

    def run_sync(
        self,
        title: str,
        outcomes: list[str],
        ground_truth: dict[str, int] | None = None,
        **kw,
    ) -> dict: ...

    @property
    def total_cost(self) -> float: ...

    @property
    def model_cost(self) -> float: ...

    @property
    def search_cost(self) -> float: ...

    def save(self, path: Path | None, *extra_dicts: dict) -> dict: ...


def __getattr__(name: str) -> Any:
    """Lazy imports for public batch API to avoid circular imports."""
    _lazy = {
        "batch_forecast": "miniprophet.eval.batch",
        "batch_forecast_sync": "miniprophet.eval.batch",
        "ForecastProblem": "miniprophet.eval.types",
        "ForecastResult": "miniprophet.eval.types",
        "BatchProgressCallback": "miniprophet.eval.types",
    }
    if name in _lazy:
        import importlib

        module = importlib.import_module(_lazy[name])
        return getattr(module, name)
    raise AttributeError(f"module 'miniprophet' has no attribute {name!r}")


__all__ = [
    "Model",
    "Tool",
    "Environment",
    "ContextManager",
    "Agent",
    "package_dir",
    "__version__",
    "batch_forecast",
    "batch_forecast_sync",
    "ForecastProblem",
    "ForecastResult",
    "BatchProgressCallback",
]
