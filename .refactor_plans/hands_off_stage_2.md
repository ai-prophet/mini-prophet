# Handoff: Stage 2 Complete — Context for Next Agent

## How This Refactoring Works

The human supervisor drives a staged refactoring of the mini-prophet forecasting agent. The coordination pipeline is:

1. **`stages.md`** — high-level stage breakdown with goals, dependencies, and design rationale (written once, may be updated)
2. For each stage:
   - Human writes **`stageX_user.md`** with feedback/requirements
   - Agent reads it and writes **`stageX.md`** — a detailed implementation plan
   - Human reviews, may request adjustments (more back-and-forth in chat)
   - Agent implements, tests pass, code is committed
   - Agent updates `stageX.md` to reflect what was actually done (including deviations)
3. The **`overall.md`** file contains the original human-written vision document

All plan files live in `.refactor_plans/`. Read `stages.md` first for the big picture, then `stageX.md` for details of each completed stage.

---

## What Has Been Done

### Stage 1: Binary Forecasting Simplification (complete)
- System restricted to binary Yes/No questions only
- Agent produces single `probability` float (P(Yes))
- `outcomes` parameter removed from agent `run()` API and all call sites
- Ground truth accepts bare `0`/`1` or case-insensitive dict via `normalize_ground_truth()`
- Non-binary questions: agent explains why, submits `probability=0`
- See `stage1.md` for full details

### Stage 2: Source Registry & Context Architecture (complete)
- **SourceBoard replaced by SourceRegistry** (`environment/source_registry.py`) — async-safe with `asyncio.Lock`, `SourceSummary` dataclass, `problem_id` context tracking, configurable `max_gist_chars`
- **New tools**: `ReadSourceTool`, `ListSourcesTool`
- **Removed tools**: `AddSourceTool`, `EditNoteTool` (file deleted: `source_board_tools.py`)
- **Board injection at position 2 removed** — context is now append-only (KV-cache friendly)
- **Context manager disabled** (config set to `"none"`, `context.py` code preserved)
- **Submit tool requires `rationale`** — displayed in CLI green panel
- **Source preview formatting**: `<source_preview>` tags with word-boundary truncation and `[preview: N/M chars]` indicators
- **Shared `render_source_preview()`** function in `source_registry.py` used by both search and list_sources tools
- See `stage2.md` for full details including post-plan deviations

---

## Current State of the Code

### Key files to explore
| Area | Files |
|------|-------|
| Agent loop | `src/miniprophet/agent/default.py` (core loop, `_prepare_messages_for_step`, `ForecastResult`) |
| CLI agent | `src/miniprophet/agent/cli_agent.py` (hooks, display, interrupt handling) |
| Source registry | `src/miniprophet/environment/source_registry.py` (`SourceRegistry`, `SourceSummary`, `render_source_preview`, `truncate_at_word_boundary`) |
| Tools | `src/miniprophet/tools/search_tool.py`, `read_source_tool.py`, `list_sources_tool.py`, `submit.py` |
| Tool factory | `src/miniprophet/environment/forecast_env.py` (`create_default_tools`, `ForecastEnvironment`) |
| Prompts & config | `src/miniprophet/config/default.yaml` (system_template, instance_template, all defaults) |
| Protocols | `src/miniprophet/__init__.py` (`Model`, `Tool`, `Environment`, `ContextManager`, `Agent` protocols) |
| Context manager | `src/miniprophet/agent/context.py` (preserved but disabled via config) |
| Eval pipeline | `src/miniprophet/eval/batch.py`, `runner.py`, `agent_runtime.py`, `types.py` |
| CLI entry | `src/miniprophet/run/cli.py` |
| Tests | `tests/conftest.py` (DummyModel, DummyEnvironment, fixtures) |

### Test status
- 272 tests passing (`pytest -q -m "not live_api"`)
- Lint clean (`ruff check src/ tests/`)

---

## Known Temporary States (will change in later stages)

1. **`ReadSourceTool` is in the main agent's tool set.** The original plan reserved it for subagents only. It was added because the system prompt teaches the model to use `read_source` and it caused "Unknown tool" errors without it. When subagents are implemented (Stage 4), the main agent should delegate source reading to subagents instead.

2. **`SourceBoard` and `Source` dataclass still exist** in `environment/source_board.py`. The `Source` dataclass is used everywhere as a DTO. `SourceBoard` and `BoardEntry` are dead code but the file hasn't been deleted because `Source` lives there. A future cleanup could move `Source` elsewhere.

3. **Context manager is disabled, not removed.** `SlidingWindowContextManager` in `agent/context.py` is fully intact. Config `context_manager_class: "none"` disables it. It may be re-enabled or replaced in a later stage.

4. **`context_window` field** still exists in `AgentConfig` (default 6) — it's unused now but wasn't removed to avoid breaking custom configs that set it.

---

## What Comes Next (from `stages.md`)

### Stage 3: Planning Schema & Planning Phase
- Pydantic models for structured plan (sub-queries, sub-problems, trigger conditions)
- Planning phase before execution — agent produces plan, human reviews
- `PlanTool` for the agent to submit plans

### Stage 4: Subagent Infrastructure
- Subagent base class, lifecycle management
- Source Reading Agent (reads sources, summarizes — uses `ReadSourceTool`)
- Subproblem Agent (searches, reads, returns probability)
- At this point, `ReadSourceTool` should be removed from the main agent's tool set

### Stage 5: Plan Execution Engine
- Wire planning schema to subagent execution
- Trigger condition evaluation, plan updates, synthesis phase

### Stage 6: Polish, Testing & Migration

See `stages.md` for full details, dependency graph, and design rationale for each stage.
