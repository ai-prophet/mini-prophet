"""TuiForecastAgent: Textual-compatible agent with queue-based interrupts."""

from __future__ import annotations

import asyncio
import queue
from collections.abc import Callable
from typing import Any

from miniprophet import ContextManager, Environment, Model
from miniprophet.agent.cli_agent import CliAgentConfig, CliForecastAgent
from miniprophet.agent.default import DefaultForecastAgent, ForecastResult
from miniprophet.cli.utils import get_console


class TuiForecastAgent(CliForecastAgent):
    """Extends CliForecastAgent for use inside a Textual TUI.

    Differences from CliForecastAgent:
    - No SIGINT handler (TUI captures keyboard input directly)
    - No Rich ``Live`` spinner (incompatible with Textual's renderer)
    - Interrupt via :class:`queue.Queue` instead of signal flag
    - Status callback for updating the TUI status bar

    The agent runs in a background thread (via ``@work(thread=True)``).
    A thread-safe :class:`queue.Queue` is used so that the Textual main
    thread can enqueue interrupt messages while the agent thread drains them.
    """

    def __init__(
        self,
        model: Model,
        env: Environment,
        *,
        context_manager: ContextManager | None = None,
        message_queue: queue.Queue | None = None,
        on_status: Callable[[str], Any] | None = None,
        config_class: type = CliAgentConfig,
        **kwargs,
    ) -> None:
        super().__init__(
            model=model,
            env=env,
            context_manager=context_manager,
            config_class=config_class,
            **kwargs,
        )
        self._message_queue: queue.Queue = message_queue or queue.Queue()
        self._on_status = on_status or (lambda _: None)

    # ------------------------------------------------------------------
    # Hook overrides for status bar
    # ------------------------------------------------------------------

    def on_step_start(self) -> None:
        super().on_step_start()
        self._on_status(f"Step {self.n_calls} | ${self.total_cost:.4f} | \u23f3 forecasting")

    def on_run_end(self, result: ForecastResult) -> None:
        super().on_run_end(result)
        self._on_status("")

    # ------------------------------------------------------------------
    # Planning: TUI-based plan approval via message queue
    # ------------------------------------------------------------------

    async def _approve_plan(self, plan, plan_xml: str) -> bool:
        console = get_console()
        console.print(
            "  [bold]Type 'approve' to proceed, or type feedback to refine the plan.[/bold]"
        )
        self._on_status("Plan ready — type 'approve' or give feedback")

        # Wait for user input via the message queue
        try:
            response = await asyncio.wait_for(self._message_queue.get(), timeout=600)
        except TimeoutError:
            console.print("  [dim]No response. Auto-approving plan.[/dim]")
            return True

        if response.strip().lower() == "approve":
            return True

        self.add_messages(
            self.model.format_message(
                role="user",
                content=(
                    f"User feedback on the plan:\n{response.strip()}\n\n"
                    "Please revise the plan based on this feedback and resubmit."
                ),
            )
        )
        self._on_status("\u23f3 Revising plan...")
        return False

    # ------------------------------------------------------------------
    # Override run: skip SIGINT handler
    # ------------------------------------------------------------------

    async def run(
        self,
        title: str,
        ground_truth: dict[str, int] | None = None,
        **runtime_kwargs,
    ) -> ForecastResult:
        # Bypass CliForecastAgent.run() which installs SIGINT handlers.
        return await DefaultForecastAgent.run(self, title, ground_truth, **runtime_kwargs)

    # ------------------------------------------------------------------
    # Override query: skip Live spinner
    # ------------------------------------------------------------------

    async def query(self) -> dict:
        console = get_console()
        model_name = getattr(self.model.config, "model_name", "model")
        console.print(f"  [dim]\u23f3 {model_name} is forecasting...[/dim]")
        # Bypass CliForecastAgent.query() which uses Live(Spinner).
        return await DefaultForecastAgent.query(self)

    # ------------------------------------------------------------------
    # Override step: queue-based interrupt instead of SIGINT
    # ------------------------------------------------------------------

    async def step(self) -> list[dict]:
        self._prepare_messages_for_step()

        message = await self.query()

        has_actions = bool(message.get("extra", {}).get("actions", []))

        # Interrupt after query, no tool actions
        if self._has_pending() and not has_actions:
            self._inject_pending()
            if self.context_manager is not None:
                self.context_manager.display()
            return list(self.messages)

        # Execute tool actions
        result = await self.execute_actions(message)

        # Interrupt after tool execution
        if self._has_pending():
            self._inject_pending()

        if self.context_manager is not None:
            self.context_manager.display()

        return result

    # ------------------------------------------------------------------
    # Queue helpers
    # ------------------------------------------------------------------

    def _has_pending(self) -> bool:
        return not self._message_queue.empty()

    def _inject_pending(self) -> None:
        """Drain all pending user messages from the queue and inject them."""
        console = get_console()
        messages: list[str] = []
        while not self._message_queue.empty():
            try:
                messages.append(self._message_queue.get_nowait())
            except queue.Empty:
                break

        if not messages:
            return

        console.rule("[bold yellow]User Message[/bold yellow]", style="yellow")
        for text in messages:
            console.print(f"  {text}")
            self.add_messages(
                self.model.format_message(
                    role="user",
                    content=text,
                    extra={"is_user_interrupt": True},
                )
            )
        console.rule(style="yellow")
