"""Default context management strategy for the forecasting agent."""

from __future__ import annotations

import importlib
import inspect
import logging

from miniprophet import ContextManager

logger = logging.getLogger("miniprophet.agent.context")

_CONTEXT_MANAGER_MAPPING: dict[str, str] = {
    "sliding_window": "miniprophet.agent.context.SlidingWindowContextManager",
}


def get_context_manager(config: dict) -> ContextManager | None:
    """Instantiate a context manager from a config dict.

    The 'context_manager_class' key selects the implementation (default: "sliding_window").
    Sub-dict keyed by the class name provides constructor kwargs.
    A value of "none" or empty string disables context management.
    """
    cm_class_key = config.get("context_manager_class", "sliding_window")
    if not cm_class_key or cm_class_key == "none":
        return None

    cm_kwargs = dict(config.get(cm_class_key, {}))

    full_path = _CONTEXT_MANAGER_MAPPING.get(cm_class_key, cm_class_key)
    try:
        module_name, class_name = full_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
    except (ValueError, ImportError, AttributeError) as exc:
        raise ValueError(
            f"Unknown context manager class: {cm_class_key} (resolved to {full_path}, "
            f"available: {list(_CONTEXT_MANAGER_MAPPING)})"
        ) from exc

    sig = inspect.signature(cls.__init__)
    if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
        accepted = cm_kwargs
    else:
        valid_keys = set(sig.parameters.keys()) - {"self"}
        accepted = {k: v for k, v in cm_kwargs.items() if k in valid_keys}

    return cls(**accepted)


class SlidingWindowContextManager:
    """Stateful sliding-window context manager.

    Always preserves:
      - messages[0]: system prompt
      - messages[1]: instance/user prompt
    When the message body exceeds `window_size`, older messages are discarded
    and replaced by a single synthetic summary that includes:
      - The cumulative count of all messages ever truncated
      - A log of all search queries issued so far (to avoid repeats)
    """

    def __init__(self, window_size: int = 6) -> None:
        self.window_size = window_size
        self._total_truncated: int = 0
        self._past_queries: list[str] = []

    @property
    def total_truncated(self) -> int:
        return self._total_truncated

    def record_query(self, query: str) -> None:
        """Called by the environment after each search to track query history."""
        self._past_queries.append(query)

    def manage(self, messages: list[dict], *, step: int, **kwargs) -> list[dict]:
        if self.window_size <= 0:
            return messages

        preamble = messages[:2]
        body = messages[2:]

        # Strip synthetic messages before counting
        body = [
            m
            for m in body
            if not m.get("extra", {}).get("is_truncation_notice")
            and not m.get("extra", {}).get("is_board_state")
        ]

        if len(body) <= self.window_size:
            return preamble + body

        # we make sure that we do not truncate away any tool call (so the tool response is not lost)
        if body[-self.window_size]["role"] == "tool":
            effective_window_size = self.window_size + 1
        else:
            effective_window_size = self.window_size

        newly_removed = len(body) - effective_window_size
        self._total_truncated += newly_removed
        kept = body[-effective_window_size:]

        lines = [
            f"[Context truncated: {self._total_truncated} earlier messages have been "
            f"omitted across this conversation. The query history below tracks all "
            f"searches you've issued so far.]",
        ]

        if self._past_queries:
            lines.append("")
            lines.append("--- Search Queries So Far ---")
            for i, q in enumerate(self._past_queries, 1):
                lines.append(f"  {i}. {q}")
            lines.append("(Do not repeat these queries.)")

        truncation_msg = {
            "role": "user",
            "content": "\n".join(lines),
            "extra": {"is_truncation_notice": True},
        }

        logger.debug(
            "Context truncated: %d messages removed (window=%d)",
            self._total_truncated,
            self.window_size,
        )
        return preamble + [truncation_msg] + kept

    def display(self):
        """Display the truncation info for CLI"""
        from miniprophet.cli.utils import get_console

        console = get_console()
        console.print(
            f"  [dim]Context truncated: {self._total_truncated} total messages "
            f"(window size={self.window_size})[/dim]"
        )
