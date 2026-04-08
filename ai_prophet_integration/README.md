# ai-prophet Integration

Bridge module that runs the mini-prophet forecasting agent inside the
[Prophet Arena](https://forecast.prophetarena.co/docs) competition.

## Setup

```bash
# From the mini-llm-prophet root:
pip install -e ".[perplexity]"

# Install the ai-prophet CLI:
cd /path/to/ai-prophet
pip install -e packages/core
pip install -e packages/cli

# Verify both CLIs are available:
miniprophet --help   # mini-prophet agent
prophet --help       # ai-prophet competition CLI
```

Make sure your `.env` (at mini-prophet root) has:

```
PA_SERVER_API_KEY=prophet_...
```

API keys for models and search (e.g. `OPENROUTER_API_KEY`, `PERPLEXITY_API_KEY`)
should be in mini-prophet's global config (`~/.config/mini-prophet/.env` or
platform equivalent). Set them with `miniprophet set KEY value`.

## One-time: register your team

```bash
export $(grep PA_SERVER_API_KEY .env | xargs)
prophet forecast register --team-name mini-prophet
```

## Running predictions

### 1. Fetch open events

```bash
export $(grep PA_SERVER_API_KEY .env | xargs)
prophet forecast events -o events.json
```

### 2. Run mini-prophet on all events

```bash
PYTHONPATH=".:src" prophet forecast predict \
  --events ai_prophet_integration/events.json \
  --local ai_prophet_integration.predict \
  --timeout 300
```

Or target a single ticker:

```bash
PYTHONPATH=".:src" prophet forecast predict \
  --events ai_prophet_integration/events.json \
  --local ai_prophet_integration.predict \
  --timeout 300 \
  --ticker KXHIGHLAX-26APR16-T90
```

### 3. Check artifacts

Each event saves `info.json`, `trajectory.json`, and `sources.json` to:

```
outputs/ai-prophet/<market_ticker>/
```

### 4. Submit

```bash
export $(grep PA_SERVER_API_KEY .env | xargs)
prophet forecast submit --submission submission.json
```

### 5. Check leaderboard

```bash
export $(grep PA_SERVER_API_KEY .env | xargs)
prophet forecast leaderboard
```

## Testing with no open events

If the server has no open events, create a synthetic one:

```bash
python -c "
import json
from datetime import datetime, timezone, timedelta
test = [{'id':1, 'event_ticker':'TEST', 'market_ticker':'TEST-001',
  'title':'Top Song on Weekly Top Songs USA on Apr 16, 2026?',
  'subtitle':':: Taylor Swift',
  'rules':'If The Alchemy by Taylor Swift is #1 on the Weekly Top Songs USA chart dated Apr 16, 2026, then the market resolves to Yes.',
  'category':'Entertainment',
  'close_time':(datetime.now(timezone.utc)+timedelta(days=7)).isoformat(),
  'actual_outcome':None, 'resolved_at':None}]
open('ai_prophet_integration/events_fake.json','w').write(json.dumps(test,indent=2))
print('Created events_fake.json with 1 future event')
"

PYTHONPATH=".:src" prophet forecast predict \
  --events ai_prophet_integration/events_fake.json \
  --local ai_prophet_integration.predict \
  --timeout 300
```

This runs the full pipeline (reformat -> plan -> search -> forecast) but won't
submit to the server since the ticker doesn't exist there.

## How it works

1. **Reformat** (`reformat.py`): Many Kalshi events have misleading titles
  (e.g. "Top Song on Weekly Top Songs USA?"). A cheap LLM call merges
   `title` + `subtitle` + `rules` into a clear binary yes/no question.
2. **Predict** (`predict.py`): Creates a mini-prophet `DefaultForecastAgent`
  with planning enabled. The agent plans (auto-approved in batch mode),
   searches, reads sources, and submits a probability. Output is clamped
   to [0.01, 0.99] per competition rules.
3. **Artifacts**: Saved to `outputs/ai-prophet/<market_ticker>/` with full
  trajectory, sources, and run metadata.

## Configuration

Default overrides for competition (in `predict.py`):


| Setting               | Value | Why                          |
| --------------------- | ----- | ---------------------------- |
| `step_limit`          | 20    | Tighter budget per event     |
| `cost_limit`          | 1.0   | ~$1 max per event            |
| `search_limit`        | 8     | Enough for thorough research |
| `planning.step_limit` | 5     | Quick planning               |
| `planning.cost_limit` | 0.2   | Cheap planning phase         |


To change the model or search backend, edit `src/miniprophet/config/default.yaml`.