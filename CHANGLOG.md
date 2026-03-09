# Changelog


## v0.1.7

### New: Polymarket integration

- Added `PolymarketService` for fetching markets and events from Polymarket, supporting ticker-based and URL-based lookups.
- `prophet run` now supports Polymarket event/market tickers and URLs as input alongside Kalshi.

### Improved: Search backend refactor

- Refactored search backends (`brave`, `exa`, `perplexity`) into `tools/search/` subpackage, separating backend implementations from the tool interface.

### Improved: Eval and dataset handling

- `prophet eval` progress tracking fixed: renamed internal `task_id` field to `run_id` to prevent crashes.
- Relaxed dataset validation requirements.
- Improved `prophet datasets` CLI display.
- Removed legacy dataset keys.

### New: Run history

- Added save/load forecast history to `prophet run`, allowing review of past runs.

### Testing and CI

- Added GitHub Actions coverage workflow.
- Reorganized test structure to mirror `src/miniprophet/` layout.
- Added tests for search backends, services, models, eval components, and tools.

## v0.1.6

### Major: `prophet eval` replaces `prophet batch`

- Removed the `prophet batch` command and migrated the batch/eval execution entrypoint to
  `prophet eval`.
- Added support for two eval input modes:
  - local JSONL (`--input/-f`)
  - dataset reference (`--dataset/-d`)
- Preserved core parallel-run behavior (workers, resume, max-cost, per-run timeout, summary
  artifacts) under the new eval command.

### New: standardized dataset management

- Added `prophet datasets` command group with:
  - `prophet datasets list`
  - `prophet datasets download`
  - `prophet datasets validate`
- Added forecast task schema validation with required `title` and `outcomes`, plus support for
  optional fields such as `task_id`, `context`, `ground_truth`, `predict_by`, `source`, `criteria`,
  and `metadata`.
- Added two dataset source patterns:
  - registry datasets (`name`, `name@version`, `name@latest`)
  - Hugging Face datasets (`username/dataset`, `username/dataset@revision`)
- Added dataset caching under `MINIPROPHET_GLOBAL_CONFIG_DIR/datasets/...`.

### New: Exa search backend

- Added `ExaSearchTool` (`search.search_class=exa`) using the official `exa-py` SDK.
- Exa now supports direct content retrieval during search via configurable modes:
  - `search.exa.content_mode=text` (uses `contents.text`)
  - `search.exa.content_mode=highlights` (uses `contents.highlights`)
- Runtime date filters are now mapped for Exa:
  - `search_date_after` -> `start_published_date`
  - `search_date_before` -> `end_published_date`
  - with conversion from `MM/DD/YYYY` to ISO-8601 UTC timestamps.
- Added optional dependency group: `pip install -e ".[exa]"`.
- Added Exa config block under `search.exa` in default config.

## v0.1.5

### Major: Search stack upgrades (SDK + backend-aware schemas)

- Perplexity integration now uses the official `perplexityai` SDK instead of raw HTTP calls.
- Added optional dependency group: `pip install -e ".[perplexity]"`.
- Search tool schemas are now backend-aware:
  - Perplexity exposes date-related fields (for example `search_after_date_filter`, `last_updated_after_filter`)
  - Brave exposes `freshness` (including custom date ranges)
- Search tool arguments are forwarded as `**kwargs` into backends, enabling backend-specific filtering behavior.

### New: Date-aware search flow

- Added runtime search date filters passed through the full stack:
  - `search.search_date_before`
  - `search.search_date_after`
- `DefaultForecastAgent.run(...)` now accepts runtime kwargs and forwards them into tool execution.
- `ForecastEnvironment.execute(...)` now merges runtime kwargs with action arguments.
- Perplexity maps runtime date bounds to corresponding API filters (`search_*_date_filter`, `last_updated_*_filter`).
- Batch mode supports per-problem `predict_by` in JSONL; it is parsed and used as `search_date_before`.
- Brave currently logs that runtime before/after filtering is not yet supported (while retaining `freshness` support).

### New: Source date metadata end-to-end

- `Source` now includes `date`.
- Search backends now populate `date`:
  - Perplexity uses publication/update dates (latest when both exist)
  - Brave uses age fields from API results
- Source board serialization/rendering includes source date.
- CLI now displays source dates in:
  - search result cards
  - source board cards
- Search observation text now includes a `Date:` line per result.

### Config and wiring changes

- Search config layout was refactored under `search.*`:
  - shared keys (`search_results_limit`, `max_source_display_chars`, runtime date bounds)
  - backend-specific nested blocks (`search.perplexity.*`, `search.brave.*`)
- `get_search_tool(...)` now reads nested backend config and defaults to `perplexity`.
- `run/cli.py` and batch runner now consume the new search config structure.
- Forecast environment no longer serializes an `environment` config block in run metadata.

### Fixes and DX improvements

- Added `python-dotenv` dependency and automatic project-root `.env` loading at import time.
- Simplified Typer dependency from `typer[all]` to `typer`.
- Updated default model name to `gemini/gemini-3-flash-preview`.
- `.gitignore` now ignores `scripts/` and all `*.jsonl` except `examples/example_batch_job.jsonl`.
- Version bumped to `0.1.5`.

## v0.1.4

### CLI: Source board redesign

The source board panel is now rendered as grouped source cards for better scanability during runs:
- Each entry shows `[#id] title`, URL, snippet preview, and note content in a dedicated nested panel
- Reactions are rendered with explicit sentiment colors/symbols (`++`, `+`, `~`, `-`, `--`)
- Long snippets/notes now show truncation hints (for example, `...N characters omitted`)
- Panel subtitle includes source count and empty-state text is cleaner

### Improved: Search results readability

Search result rendering has been refined to reduce visual noise:
- Result metadata and title now render on one line
- Snippet truncation now explicitly reports omitted character count

### Improved: Batch run progress and cost accounting

Batch progress behavior is now more informative and simpler:
- `BatchForecastAgent` now tracks per-step cost deltas and forwards them to the progress manager
- Per-run status now displays both consumed cost and run limit (`$used/$limit`)
- Batch progress UI removed ETA estimation (elapsed/progress/cost remain)
- Added `examples/example_batch_job.jsonl` as a ready-to-run batch input sample

### Important: Trajectory key format update

Trajectory message keys are now role-prefixed instead of generic `mN` IDs:
- Previous format example: `m0`, `m1`, `m2`
- New format example: `S0`, `U0`, `A0`, `T0` (with `O*` for non-standard roles)

This improves trace readability but changes `trajectory.json` key naming for downstream consumers.

### Config and docs updates

- Default config now explicitly sets `model.model_class: "litellm"`
- README quickstart examples now include `--model-class litellm` with Gemini model usage
- Minor context-summary wording cleanup in truncated context text

## v0.1.3

### Major: Agent trajectory observability

Agent runs now record full per-step input/output trajectories via a `TrajectoryRecorder`. Each message is stored once in a global pool and referenced by key, allowing exact reconstruction of what the LLM saw at every step -- even when the context manager truncates or replaces messages between steps.

Serialization output has changed from a single JSON file to a **directory** containing:
- `info.json` -- config, cost stats, version, exit status, submission, evaluation
- `trajectory.json` -- message pool + per-step input/output indices

The `--output` CLI option now points to a directory instead of a file.

### Improved: XML-tagged prompt format

Structured data in prompts and tool outputs now uses XML tags for clearer parsing by LLMs:
- Source board: `<source_board>` / `<source>` tags
- Tool output: `<output>` / `<error>` tags
- Search results: `<search_results>` / `<result>` tags
- Instance template: `<forecast_problem>` tag for the problem definition

Instruction sections (Strategy, Guidelines) remain as markdown lists.

### Major: Batch agent running mode

New `prophet batch` CLI command for running multiple forecasting problems in parallel:
- Input: `.jsonl` file with `title`, `outcomes`, optional `ground_truth` and `task_id` per line
- Output: meta-directory with per-run artifacts and a `summary.json` with status, costs, submissions, and evaluations
- Parallel execution via `--workers N` with a queue-based scheduler
- Global `RateLimitCoordinator`: when any worker hits a rate limit, all workers pause for a backoff period
- Automatic retry (up to 3 attempts) for rate-limited runs; permanent failures are recorded immediately
- Total cost budget (`--max-cost`) and per-run cost limit (`--max-cost-per-run`)
- Rich live progress display with overall progress bar, per-worker status, and exit status summary table

### CLI restructured

The CLI is now organized as sub-commands under a root `prophet` app:
- `prophet run` -- single forecast run (previously the default command)
- `prophet batch` -- batch forecasting from JSONL

## v0.1.2

### New: LiteLLM model support

Added `LitellmModel` as a second model class alongside `OpenRouterModel`. Use `--model-class litellm` to access any provider supported by LiteLLM (OpenAI, Anthropic, Google, etc.) without needing an OpenRouter key.

### Major: Agent and CLI separation

The agent loop no longer contains any display logic. `DefaultForecastAgent` is now a pure backend class with hook methods (`on_run_start`, `on_step_start`, `on_model_response`, `on_observation`, `on_run_end`) that subclasses can override.

A new `CliForecastAgent` inherits from it and provides all Rich-based CLI display, including a live spinner while the model is thinking. The CLI entry point (`run/cli.py`) now uses `CliForecastAgent`.

All display components have been moved into a dedicated `cli/components/` package, replacing the old monolithic `utils/display.py` and `run/tui.py`.

### Major: Modular tool system

Tools are no longer defined inline inside `ForecastEnvironment`. A new `Tool` protocol (`name`, `get_schema`, `execute`, `display`) allows each tool to be a self-contained module under the `tools/` package:

- `tools/search_tool.py` -- web search with source ID tracking
- `tools/source_board_tools.py` -- `add_source` and `edit_note`
- `tools/submit.py` -- final forecast submission

`ForecastEnvironment` is now a thin dispatcher that receives a list of `Tool` instances and delegates execution by name.

### Improved: Source board as invariant context

The source board state is now injected as a persistent message (position 2, after system and user prompts) that refreshes every step, rather than being appended to each tool's output. This means the model always sees the current board at a fixed location, and tool responses only contain their own results.

### CLI improvements

- A live spinner shows the model name while waiting for a response (e.g. "openai/gpt-4o is forecasting...").
- In interactive mode, the agent loops back to the forecast setup screen after each run instead of exiting.
- Empty fields in the forecast setup TUI now show "(Empty: awaiting input...)" instead of blank parentheses.
- Forecast results, evaluation metrics, and the run footer are displayed in that order after the agent finishes (previously metrics appeared before the forecast).
- `run()` now returns a `ForecastResult` TypedDict with structured `exit_status`, `submission`, `evaluation`, and `board` fields.
