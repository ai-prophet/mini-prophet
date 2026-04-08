<p align="center">
  <img src="./assets/icon.svg" alt="mini-prophet icon" style="height:10em"/>
</p>

# mini-prophet

[![CI](https://github.com/ai-prophet/mini-prophet/actions/workflows/ci.yml/badge.svg)](https://github.com/ai-prophet/mini-prophet/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/ai-prophet/mini-prophet/branch/main/graph/badge.svg)](https://codecov.io/gh/ai-prophet/mini-prophet)

A minimal LLM forecasting agent scaffolding. Inspired by [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent).

<p align="left">
  <img src="./assets/mini-prophet-loop.svg" alt="mini-prophet loop" style="height:100%"/>
</p>

## Install

```bash
cd mini-prophet
pip install -e ".[perplexity]"
```

`perplexity` is the default search backend, so this is the recommended install.
If you only use Brave search, `pip install -e .` is enough.
If you want Exa search, use `pip install -e ".[exa]"`.
If you want Tavily search, use `pip install -e ".[tavily]"`.

## Set API keys

Use the built-in CLI to persist keys into prophet's global `.env` file:

```bash
# Step 1: search API keys (one of the below is good)
miniprophet set PERPLEXITY_API_KEY "your-perplexity-key"
miniprophet set BRAVE_API_KEY "your-brave-key"
miniprophet set EXA_API_KEY "your-exa-key"
miniprophet set TAVILY_API_KEY "your-tavily-key"

# Step 2: model API key (if you use OpenRouter, this works with any model)
miniprophet set OPENROUTER_API_KEY "your-openrouter-key"

# Or you go with LiteLLM (--model-class litellm), set model-specific API keys
# For instance, if you want to use OpenAI models
miniprophet set OPENAI_API_KEY "your-openai-key"

# interactive editor can do the same!
miniprophet set -i
```

By default, miniprophet stores and loads values from:

- `~/.config/mini-prophet/.env` (in linux; or your platform's equivalent config directory)

To use a different global config directory, set:

```bash
export MINIPROPHET_GLOBAL_CONFIG_DIR="/path/to/custom/config-dir"
```

You can still set environment variables directly in your shell if you prefer.

## Try this

Single run:

```bash
miniprophet run \
  --title "Which team will win the NBA championship in 2026?" \
  --outcomes "Bucks,Warriors,Celtics,Nuggets,Other" \
  --model-class litellm \
  --model gemini/gemini-3-flash-preview
```

Interactive mode:

```bash
miniprophet run -i --model-class litellm --model gemini/gemini-3-flash-preview
```

Eval mode with the sample file:

```bash
miniprophet eval \
  -f examples/example_batch_job.jsonl \
  -o outputs/eval-demo \
  -w 4 \
  --model-class litellm \
  --model gemini/gemini-3-flash-preview
```

Resume an interrupted eval run and skip completed run IDs:

```bash
miniprophet eval -f examples/example_batch_job.jsonl -o outputs/eval-demo --resume
```

Run eval directly from a standardized dataset:

```bash
miniprophet eval -d weekly-nba@latest -o outputs/weekly-nba
```

List and validate datasets:

```bash
miniprophet datasets list
miniprophet datasets list weekly-nba
miniprophet datasets validate -f examples/example_batch_job.jsonl
```

Run artifacts now include `sources.json` in addition to `info.json` and `trajectory.json`.

## Preview: Planning Phase (`dev` branch)

The `dev` branch includes a new **planning phase** that runs before execution.
Instead of immediately searching, the agent first produces a structured XML plan
that decomposes the question into sub-queries, sub-problems, and factors to consider.
You review and approve the plan before any research (and cost) begins.

```bash
# Switch to the dev branch
git checkout dev
pip install -e ".[perplexity]"

# Run with planning enabled (default on dev)
miniprophet run \
  --title "Will the Lakers beat the Warriors tomorrow night?" \
  --model-class openrouter \
  --model gemini/gemini-3-flash-preview

# Planning is on by default. To disable it:
miniprophet run \
  --title "Will it rain in SF tomorrow?" \
  -c planning.enabled=false
```

The agent will:
1. Enter a **planning phase** — analyze the question and submit a plan
2. Display the plan as a tree and ask for your approval
3. Enter the **execution phase** — follow the plan, search, and submit a probability

The `dev` branch also includes an experimental Textual TUI (`--tui` flag).

## Contributing

Contributor-facing testing and CI notes live in [CONTRIBUTING.md](CONTRIBUTING.md).

## Docs

Detailed docs are split by topic:

- [Architecture](docs/architecture.md)
- [CLI (single run)](docs/cli.md)
- [Batch forecasting API](docs/batch.md)
- [Eval CLI](docs/eval.md)
- [Dataset management](docs/datasets.md)
- [Extending the framework](docs/extension.md)
- [Output and trajectory format](docs/output.md)

Release notes: [CHANGLOG.md](CHANGLOG.md)
