# Batch Forecasting API

`batch_forecast` is the public Python API for running multiple forecasting
problems in parallel. It handles worker orchestration, rate-limit coordination,
cost tracking, retries, and timeouts.

The `prophet eval` CLI ([docs/eval.md](eval.md)) is one consumer of this API —
you can also call it directly from your own code with any agent implementation.

## Quick start

```python
from miniprophet import batch_forecast, ForecastProblem

problems = [
    ForecastProblem(
        task_id="q1",
        title="Will inflation in the US be above 3% by Dec 2026?",
        outcomes=["Yes", "No"],
    ),
    ForecastProblem(
        task_id="q2",
        title="Which team wins the 2026 NBA title?",
        outcomes=["Celtics", "Thunder", "Other"],
        ground_truth={"Celtics": 0, "Thunder": 1, "Other": 0},
    ),
]

results = batch_forecast(problems, workers=2)

for r in results:
    print(r.task_id, r.status, r.submission)
```

## `batch_forecast` signature

```python
def batch_forecast(
    problems: list[ForecastProblem],
    config: dict | str | Path | None = None,
    *,
    workers: int = 1,
    timeout_seconds: float = 180.0,
    max_total_cost: float = 0.0,
    search_date_before: str | None = None,
    search_date_after: str | None = None,
    on_progress: BatchProgressCallback | None = None,
    include_trajectory: bool = False,
    agent_name: str | None = None,
    agent_class: type | None = None,
    agent_kwargs: dict[str, Any] | None = None,
    model: Model | None = None,
    search_backend: Any | None = None,
) -> list[ForecastResult]:
```

| Parameter | Description |
| --- | --- |
| `problems` | List of `ForecastProblem` instances to solve |
| `config` | Config dict, path to a YAML file, or `None` for defaults |
| `workers` | Number of parallel worker threads |
| `timeout_seconds` | Per-problem timeout (0 = unlimited) |
| `max_total_cost` | Stop when cumulative cost exceeds this (0 = unlimited) |
| `search_date_before` | Restrict searches to before this date (`MM/DD/YYYY`) |
| `search_date_after` | Restrict searches to after this date (`MM/DD/YYYY`) |
| `on_progress` | Optional `BatchProgressCallback` for progress updates |
| `include_trajectory` | If `True`, include agent trajectory in each result |
| `agent_name` | Built-in agent alias (currently only `"default"`) |
| `agent_class` | Agent class directly |
| `agent_kwargs` | Extra kwargs forwarded to the agent constructor |
| `model` | Pre-constructed `Model` instance (skips `get_model()` when provided) |
| `search_backend` | Pre-constructed search backend instance (skips `get_search_backend()` when provided) |

Returns a `list[ForecastResult]` in the same order as the input problems.

## Data types

### `ForecastProblem`

```python
@dataclass
class ForecastProblem:
    task_id: str
    title: str
    outcomes: list[str]
    ground_truth: dict[str, int] | None = None
    predict_by: str | None = None
    context: str | None = None
    source: str | None = None
    criteria: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

If `predict_by` is set, it is automatically converted to `MM/DD/YYYY` format
and used as the default `search_date_before` for that problem.

### `ForecastResult`

```python
@dataclass
class ForecastResult:
    task_id: str
    title: str
    status: str = "pending"
    submission: dict[str, float] | None = None
    evaluation: dict[str, float] | None = None
    cost: dict[str, float] = ...  # {"model", "search", "total"}
    error: str | None = None
    trajectory: dict | None = None
```

### `BatchProgressCallback`

```python
class BatchProgressCallback(Protocol):
    def on_run_start(self, task_id: str) -> None: ...
    def update_run_status(self, task_id: str, message: str, cost_delta: float = 0.0) -> None: ...
    def on_run_end(self, task_id: str, exit_status: str | None) -> None: ...
```

## Using a custom agent

There are two ways to provide a custom agent:

### 1. Pass the class directly

```python
from miniprophet import batch_forecast, ForecastProblem

results = batch_forecast(
    problems,
    agent_class=MyForecastAgent,
    agent_kwargs={"temperature": 0.2},
)
```

### 2. Use the CLI with an import path

```bash
prophet eval -f tasks.jsonl --agent-import-path mypackage.agents:MyForecastAgent
```

### Agent contract

Your agent class must accept at least `model` and `env` as constructor
arguments, and expose a `run()` method:

```python
class MyForecastAgent:
    def __init__(self, model, env, **kwargs):
        ...

    def run(self, title, outcomes, ground_truth=None, **runtime_kwargs) -> dict:
        # Must return a dict with at least "exit_status"
        # Optionally include "submission" and "evaluation"
        ...
```

If your constructor also accepts `context_manager`, it will be injected
automatically. Any extra `agent_kwargs` are forwarded as keyword arguments to
the constructor.

## Config resolution

When `config` is:

- `None` → built-in defaults are loaded
- a `str` or `Path` → treated as a YAML config file, merged over defaults
- a `dict` → merged over defaults directly

The resolved config controls model, search backend, agent limits, and context
manager settings. See [docs/extension.md](extension.md) for config structure.

## Error handling

- **Auth errors** (`SearchAuthError` or heuristic detection) abort the entire
  batch immediately via `BatchFatalError`.
- **Rate limits** trigger a shared `RateLimitCoordinator` that pauses all
  workers, then retries the problem (up to 3 times).
- **Timeouts** mark the individual result as `BatchRunTimeoutError` and continue.
- **Other exceptions** are recorded in the result's `error` field and the batch
  continues.

## Relationship to `prophet eval`

`prophet eval` is a CLI wrapper that adds:

- Dataset resolution (registry, Hugging Face, local JSONL)
- Resume from existing `summary.json`
- Rich live progress display
- Per-run artifact saving to disk

Under the hood, it uses its own orchestration loop that shares core
infrastructure (`EvalAgentFactory`, `RateLimitCoordinator`,
`EvalBatchAgentWrapper`) with `batch_forecast`. See [docs/eval.md](eval.md) for
CLI-specific options.
