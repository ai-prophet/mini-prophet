"""Predict entry point for the ai-prophet competition.

Usage::

    prophet forecast predict \\
        --events events.json \\
        --local ai_prophet_integration.predict \\
        --timeout 180
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Ensure mini-prophet's src is importable
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from ai_prophet_integration.reformat import reformat_event  # noqa: E402

logger = logging.getLogger("ai_prophet_integration.predict")


def predict(event: dict) -> dict:
    """ai-prophet compatible predict function.

    Called by ``prophet forecast predict --local ai_prophet_integration.predict``.

    Args:
        event: dict with keys like market_ticker, title, rules, category, close_time.

    Returns:
        dict with ``p_yes`` (float 0.01-0.99) and ``rationale`` (str).
    """
    from dotenv import load_dotenv

    # Load mini-prophet's global config (has API keys)
    import miniprophet

    load_dotenv(miniprophet.global_config_file)
    load_dotenv()  # also load local .env if present

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    market_ticker = event.get("market_ticker", "unknown")
    logger.info("=== Processing %s: %s ===", market_ticker, event.get("title", "")[:80])

    # 1. Reformat event into a clean binary question
    title = reformat_event(event)
    logger.info("Binary question: %s", title)

    # 2. Run mini-prophet agent
    from miniprophet.agent.context import get_context_manager
    from miniprophet.agent.default import DefaultForecastAgent
    from miniprophet.config import get_config_from_spec
    from miniprophet.environment.forecast_env import (
        ForecastEnvironment,
        create_default_tools,
        create_planning_tools,
    )
    from miniprophet.environment.source_registry import SourceRegistry
    from miniprophet.models import get_model
    from miniprophet.tools.search import get_search_backend
    from miniprophet.utils.serialize import recursive_merge

    config = get_config_from_spec("default")
    # Artifacts directory: outputs/ai-prophet/<market_ticker>/
    output_dir = Path(_ROOT) / "outputs" / "ai-prophet" / market_ticker
    output_dir.mkdir(parents=True, exist_ok=True)

    config = recursive_merge(
        config,
        {
            "agent": {
                "step_limit": 20,
                "cost_limit": 1.0,
                "search_limit": 8,
                "output_path": str(output_dir),
                "planning": {
                    "enabled": True,
                    "step_limit": 5,
                    "cost_limit": 0.2,
                },
            },
        },
    )

    model = get_model(config=config.get("model", {}))
    search_cfg = config.get("search", {})
    search_backend = get_search_backend(search_cfg=search_cfg)

    agent_cfg = config.get("agent", {})
    registry = SourceRegistry(
        max_gist_chars=int(search_cfg.get("max_source_display_chars", 200) or 200)
    )
    tools = create_default_tools(
        search_tool=search_backend,
        registry=registry,
        search_limit=int(agent_cfg.get("search_limit", 8) or 8),
        search_results_limit=int(search_cfg.get("search_results_limit", 5) or 5),
    )
    planning_tools = create_planning_tools()
    env = ForecastEnvironment(tools, planning_tools=planning_tools, registry=registry)

    cm_cfg = config.get("context_manager", {})
    ctx_mgr = get_context_manager(cm_cfg)

    agent = DefaultForecastAgent(
        model=model,
        env=env,
        context_manager=ctx_mgr,
        **agent_cfg,
    )

    try:
        result = agent.run_sync(title=title, ground_truth=None)
    except Exception as exc:
        logger.error("Agent failed for %s: %s", market_ticker, exc)
        return {"p_yes": 0.5, "rationale": f"Agent error: {exc}"}

    # 3. Extract and convert result
    submission = result.get("submission", {})
    p_yes = submission.get("Yes", 0.5)
    p_yes = max(0.01, min(0.99, float(p_yes)))
    rationale = result.get("rationale", "")
    exit_status = result.get("exit_status", "unknown")

    logger.info(
        "Result for %s: p_yes=%.3f, exit=%s, cost=$%.4f",
        market_ticker,
        p_yes,
        exit_status,
        agent.total_cost,
    )

    return {"p_yes": p_yes, "rationale": rationale}
