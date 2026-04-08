"""Reformat ai-prophet events into clean binary yes/no questions.

Many competition events have misleading titles (e.g. "Top Song on Weekly Top
Songs USA on Mar 19, 2026?") that only become binary when combined with the
``rules`` field.  This module uses a lightweight LLM call to produce a clear
binary question suitable for mini-prophet.
"""

from __future__ import annotations

import json
import logging

import litellm

logger = logging.getLogger("ai_prophet_integration.reformat")

REFORMAT_SYSTEM = """\
You convert Kalshi market events into clear, self-contained binary yes/no questions.

You will receive an event with title, subtitle, rules, category, and close_time.
Your job: produce a single binary yes/no question that captures the exact resolution
criteria from the rules field.

Rules:
- The question MUST be answerable with "Yes" or "No".
- Include all specifics: names, dates, thresholds, exact conditions.
- Do NOT add opinions or analysis — just restate the resolution criteria as a question.
- Keep it concise (one sentence).

Respond with ONLY a JSON object: {"question": "Will ...?"}
"""


def reformat_event(
    event: dict,
    *,
    model: str = "openrouter/google/gemini-2.5-flash",
) -> str:
    """Convert a raw ai-prophet event into a binary yes/no question.

    If the event already has a clearly binary title and no rules, returns the
    title directly.  Otherwise uses a cheap LLM call to merge title + rules.
    """
    title = event.get("title", "").strip()
    subtitle = event.get("subtitle", "").strip()
    rules = (event.get("rules") or "").strip()
    description = (event.get("description") or "").strip()
    category = event.get("category", "")
    close_time = event.get("close_time", "")

    # Fast path: if no rules and title looks binary, use it directly
    if not rules and title.lower().startswith("will "):
        logger.info("Using title directly (no rules): %s", title)
        return title

    # Build user prompt
    parts = [f"Title: {title}"]
    if subtitle:
        parts.append(f"Subtitle: {subtitle}")
    if rules:
        parts.append(f"Rules: {rules}")
    if description:
        parts.append(f"Description: {description}")
    parts.append(f"Category: {category}")
    parts.append(f"Close time: {close_time}")

    user_prompt = "\n".join(parts)

    # Ensure API keys are loaded
    from dotenv import load_dotenv

    import miniprophet

    load_dotenv(miniprophet.global_config_file)

    try:
        response = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": REFORMAT_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=200,
            temperature=0.0,
        )
        text = response.choices[0].message.content.strip()

        # Strip markdown fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()

        data = json.loads(text)
        question = data.get("question", title)
        logger.info("Reformatted: %s → %s", title[:60], question[:80])
        return question
    except Exception:
        logger.warning("Reformat LLM call failed for '%s', falling back to title+rules", title)
        # Fallback: combine title and rules manually
        if rules:
            return f"{title} (Resolution: {rules})"
        return title
