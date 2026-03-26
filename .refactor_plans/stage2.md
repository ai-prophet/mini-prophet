# Stage 2: Source Registry & Context Architecture — Final Record

## Goal
Replace the invariant source board with a harness-level **SourceRegistry**. Remove board injection, add `ReadSourceTool` and `ListSourcesTool`, remove `AddSourceTool`/`EditNoteTool`, disable the context manager. Make the main agent's context append-only and KV-cache friendly.

## User Feedback Incorporated (from `stage2_user.md`)
1. Source retrieval tool named `ReadSourceTool`
2. Max gist length configurable (default 200 chars)
3. `ListSourcesTool` with `problem_id` filter — sources tagged with their invoking context
4. Disable context manager (keep file, change config to `"none"`)
5. CLI display handled for new tools
6. **`SourceEntry` → `SourceSummary`** — rename to highlight summary purpose
7. **Fully async registry operations** + `asyncio.Lock` for concurrent subagent writes
8. **Batch isolation** — each forecasting problem gets its own independent registry

---

## What Was Actually Implemented

### New files
- `src/miniprophet/environment/source_registry.py` — `SourceSummary`, `SourceRegistry` (async, locked writes), `truncate_at_word_boundary()`, `render_source_preview()` (shared XML formatter)
- `src/miniprophet/tools/read_source_tool.py` — `ReadSourceTool`
- `src/miniprophet/tools/list_sources_tool.py` — `ListSourcesTool`
- `tests/environment/test_source_registry.py`, `tests/tools/test_read_source_tool.py`, `tests/tools/test_list_sources_tool.py`

### Deleted files
- `src/miniprophet/tools/source_board_tools.py` (AddSourceTool, EditNoteTool)
- `tests/tools/test_source_board_tools.py`, `tests/environment/test_source_board.py`

### Updated files
- `src/miniprophet/tools/search_tool.py` — uses `SourceRegistry`, async `_assign_source_id`, `<source_preview>` XML tags with truncation indicators
- `src/miniprophet/tools/submit.py` — takes `SourceRegistry`, adds required `rationale` param, payload carries `"sources"` + `"rationale"`
- `src/miniprophet/environment/forecast_env.py` — `create_default_tools` takes `registry` instead of `board`, tool set is `[search, read_source, list_sources, submit]`
- `src/miniprophet/agent/default.py` — removed board injection, removed `record_query`, `ForecastResult` has `rationale` + `sources` fields
- `src/miniprophet/agent/cli_agent.py` — `on_run_end` displays rationale in green panel
- `src/miniprophet/cli/components/forecast_results.py` — added `print_rationale()`
- `src/miniprophet/config/default.yaml` — context manager disabled, prompts rewritten (see below)
- `src/miniprophet/run/cli.py`, `eval/batch.py`, `eval/runner.py` — `SourceRegistry` replaces `SourceBoard`
- `tests/conftest.py` — removed `DummyBoard`, `DummyEnvironment` simplified
- Various test files updated for new APIs

### Prompt changes
- System template: new "Working with sources" section with `IMPORTANT:` epistemic framing, `<example>` block showing search → read_source workflow, graduated firmness hierarchy
- Submit tool schema: `rationale` is required alongside `probability`
- Search limit message advises using `read_source` and `list_sources` before submitting

---

## Post-Plan Deviations

These features were requested after the initial `stage2_user.md` and `stage2.md` were written:

### Deviation 1: Submit rationale (required)
- `rationale` added as a required string parameter in `SUBMIT_SCHEMA`
- Carried in `Submitted` payload → `ForecastResult["rationale"]`
- Displayed in CLI via `print_rationale()` (green `[Rationale]` panel after probability bar)
- System prompt guideline: "Before submitting, summarize your rationale"

### Deviation 2: Source preview formatting overhaul
- `<result>` → `<source_preview>` XML tags in both search results and list_sources output
- Truncation indicator: `[preview: 200/3400 chars — use read_source("S4") for full text]`
- Word-boundary truncation via `truncate_at_word_boundary()` (never cuts mid-word)
- Shared `render_source_preview()` function in `source_registry.py` — used by both `SearchForecastTool` and `ListSourcesTool` for consistency

### Deviation 3: System prompt rewrite with Claude Code-inspired patterns
- Studied Claude Code system prompt for generalizable agent harness prompting strategies
- Applied: conditional routing ("When X, do Y"), failure scenario teaching, `<example>` blocks, graduated firmness (IMPORTANT / Do NOT / Prefer / Consider), explicit epistemic boundaries
- New "Working with sources" section replaces the old `<tips>` block

### Deviation 4: `ReadSourceTool` added to main agent's tool set
- Originally planned as subagent-only. Added to `create_default_tools()` because the system prompt teaches the model to use it and testing showed "Unknown tool" errors without it.
- **Intended to be removed from the main agent's tool set later** when subagents are implemented (Stage 4). For now the main agent needs it.

### Deviation 5: Search limit message updated
- When max searches reached, message now advises: "You can still use `read_source` to read the full text of sources you have already found, or `list_sources` to review them."
