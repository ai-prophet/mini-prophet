# CLI (Single Run)

This page covers the single-run CLI flow: `prophet run`.

## Commands

Root app:

- `prophet run` - single forecast (interactive or argument-driven)
- `prophet eval` - eval forecasting (see `docs/eval.md`)
- `prophet datasets` - dataset list/download/validate (see `docs/datasets.md`)

## `prophet run` usage

```bash
prophet run --title "..." --outcomes "Yes,No"
```

### Flags

| Flag | Short | Description |
| --- | --- | --- |
| `--title` | `-t` | Forecast question |
| `--outcomes` | `-o` | Comma-separated outcomes |
| `--ground-truth` | `-g` | Ground truth JSON |
| `--interactive` | `-i` | Launch interactive setup flow |
| `--model` | `-m` | Model override |
| `--model-class` |  | Model class override (`litellm`, `openrouter`) |
| `--search-class` |  | Search class override (`perplexity`, `exa`, `brave`, `tavily`) |
| `--cost-limit` | `-l` | Total run cost limit |
| `--search-limit` |  | Max search tool calls |
| `--step-limit` |  | Max loop steps |
| `--config` | `-c` | Config file(s) or `key=value` overrides |
| `--output` |  | Output directory (`info.json`, `trajectory.json`, `sources.json`) |
| `--disable-history` |  | Skip writing to forecast history |
| `--disable-interrupt` |  | Disable Ctrl+C pause-and-inject during agent run |

## Examples

Basic:

```bash
prophet run \
  -t "Will inflation in the US be above 3% by Dec 2026?" \
  -o "Yes,No"
```

With explicit model and limits:

```bash
prophet run \
  -t "Which team wins the 2026 NBA title?" \
  -o "Bucks,Warriors,Celtics,Nuggets,Other" \
  --model-class litellm \
  --model gemini/gemini-3-flash-preview \
  --cost-limit 2.0 \
  --search-limit 12 \
  --step-limit 40
```

With evaluation:

```bash
prophet run \
  -t "Will event X happen by June 2026?" \
  -o "Yes,No" \
  -g '{"Yes": 1, "No": 0}'
```

Interactive setup:

```bash
prophet run -i
```

## Config overrides you will use often

```bash
# switch search backend
prophet run -t "..." -o "..." --search-class perplexity
# this is equivalent to:
prophet run -t "..." -o "..." -c search.search_class=exa

# tune backend-specific search settings
prophet run -t "..." -o "..." -c search.perplexity.max_tokens=8000
prophet run -t "..." -o "..." -c search.exa.content_mode=highlights
prophet run -t "..." -o "..." -c search.tavily.search_depth=advanced

# inject run-time date bounds
prophet run -t "..." -o "..." -c search.search_date_before=01/31/2026
prophet run -t "..." -o "..." -c search.search_date_after=01/01/2025
```

## Interactive mode behavior

Interactive mode supports:

- manual title/outcomes entry
- Kalshi ticker import (auto-detect event/market)
- Polymarket identifier import (auto-detect event/market)
- browsing and rerunning from historical forecasts

## Interrupt mode (human-in-the-loop)

By default, pressing `Ctrl+C` during an agent run pauses the agent after the current operation (LLM call or tool execution) completes. You can then type a message to inject into the conversation (e.g., guidance, corrections) and the agent resumes with your input in context.

- First `Ctrl+C`: pauses after current operation finishes
- Type a message and press Enter to inject it, or press Enter with no input to resume silently
- Second `Ctrl+C` (double-tap): hard abort

Disable with `--disable-interrupt` or `agent.enable_interrupt=false` in config.

## Exit/output behavior

On completion, the run returns:

- `exit_status`
- `submission` (if any)
- optional `evaluation`
- source board snapshot

When `--output` is set, artifacts are saved under that directory.
