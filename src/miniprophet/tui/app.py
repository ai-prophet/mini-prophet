"""ForecastApp: Textual TUI wrapper for the mini-prophet agent loop."""

from __future__ import annotations

import queue
from typing import Any

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import RichLog

from miniprophet.tui.console import TuiConsole
from miniprophet.tui.widgets.prompt_input import PromptInput


class ForecastApp(App):
    """Textual application that wraps the forecasting agent trace.

    The main area is a scrollable :class:`RichLog` that receives all Rich
    output via :class:`TuiConsole`.  A :class:`PromptInput` is docked at the
    bottom for question entry and agent interrupts.

    The agent loop runs in a background **thread** (``@work(thread=True)``)
    so that litellm's async model calls get their own event loop and do not
    conflict with Textual's.  All widget updates from the worker thread go
    through :meth:`call_from_thread` for safety.
    """

    CSS_PATH = "app.tcss"

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit", priority=True),
    ]

    def __init__(
        self,
        config: dict[str, Any],
        *,
        initial_title: str | None = None,
        initial_ground_truth: dict[str, int] | None = None,
        runtime_kwargs: dict[str, Any] | None = None,
        disable_history: bool = False,
        version: str = "",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._config = config
        self._initial_title = initial_title
        self._initial_ground_truth = initial_ground_truth
        self._runtime_kwargs = runtime_kwargs or {}
        self._disable_history = disable_history
        self._version = version
        self._interrupt_queue: queue.Queue = queue.Queue()
        self._agent_running = False

    # ------------------------------------------------------------------
    # Compose & mount
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield RichLog(id="trace", wrap=True, auto_scroll=True)
        yield PromptInput(id="prompt")

    def on_mount(self) -> None:
        from miniprophet.cli.components.banner import print_cli_banner, print_run_info
        from miniprophet.cli.utils import set_console_target

        # Redirect all console output into the trace log
        trace = self.query_one("#trace", RichLog)
        set_console_target(TuiConsole(trace, app=self))

        # Print banner
        model_cfg = self._config.get("model", {})
        search_cfg = self._config.get("search", {})
        print_cli_banner(self._version, mode_label="TUI")
        print_run_info(
            model_class=model_cfg.get("model_class", "litellm"),
            model_name=model_cfg.get("model_name", ""),
            search_class=search_cfg.get("search_class", "perplexity"),
        )

        # Auto-start if a title was given via CLI
        if self._initial_title:
            self._start_run(self._initial_title, self._initial_ground_truth)
        else:
            self.query_one("#prompt", PromptInput).focus_input()

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    @on(PromptInput.Submitted)
    def on_prompt_submitted(self, event: PromptInput.Submitted) -> None:
        if self._agent_running:
            self._interrupt_queue.put_nowait(event.text)
        else:
            self._start_run(event.text, None)

    # ------------------------------------------------------------------
    # Agent lifecycle
    # ------------------------------------------------------------------

    def _start_run(self, title: str, ground_truth: dict[str, int] | None) -> None:
        """Prepare UI state and launch the forecast worker.

        Must be called from the **main** (Textual) thread.
        """
        self._agent_running = True
        self._interrupt_queue = queue.Queue()
        prompt = self.query_one("#prompt", PromptInput)
        prompt.placeholder = "Send a message to the agent..."
        prompt.status = "\u23f3 Starting..."
        self._run_forecast(title, ground_truth)

    @work(thread=True, exclusive=True)
    def _run_forecast(self, title: str, ground_truth: dict[str, int] | None) -> None:
        """Run the agent loop in a background thread.

        All widget interactions go through :meth:`call_from_thread`.
        """
        from miniprophet.cli.utils import get_console

        try:
            agent = self._create_agent()
            result = agent.run_sync(title, ground_truth, **self._runtime_kwargs)

            console = get_console()

            if result.get("submission") and not self._disable_history:
                from miniprophet.cli.components.forecast_history import append_history

                model_cfg = self._config.get("model", {})
                append_history(
                    title=title,
                    ground_truth=ground_truth,
                    submission=result["submission"],
                    model_name=model_cfg.get("model_name", ""),
                    model_class=model_cfg.get("model_class", ""),
                )

            if not result.get("submission"):
                console.print(
                    f"\n[bold yellow]Agent exited without submitting.[/bold yellow] "
                    f"Status: {result.get('exit_status', 'unknown')}"
                )

        except Exception as exc:
            get_console().print(f"\n[bold red]Error: {exc}[/bold red]")

        finally:
            self._agent_running = False
            self._interrupt_queue = queue.Queue()
            self.call_from_thread(self._reset_prompt)

    def _reset_prompt(self) -> None:
        """Reset prompt to idle state.  Must be called on the main thread."""
        prompt = self.query_one("#prompt", PromptInput)
        prompt.placeholder = "Enter forecasting question..."
        prompt.status = "Enter: submit | Ctrl+Q: quit"
        prompt.focus_input()

    def _set_status(self, text: str) -> None:
        """Update the status bar.  Must be called on the main thread."""
        prompt = self.query_one("#prompt", PromptInput)
        prompt.status = text or "Enter: submit | Ctrl+Q: quit"

    # ------------------------------------------------------------------
    # Agent factory
    # ------------------------------------------------------------------

    def _create_agent(self):
        """Build a fresh TuiForecastAgent from the stored config.

        Called from the worker thread.  Widget interactions (status updates)
        are dispatched to the main thread via :meth:`call_from_thread`.
        """
        from miniprophet.agent.context import get_context_manager
        from miniprophet.agent.tui_agent import TuiForecastAgent
        from miniprophet.environment.forecast_env import (
            ForecastEnvironment,
            create_default_tools,
            create_planning_tools,
        )
        from miniprophet.environment.source_registry import SourceRegistry
        from miniprophet.models import get_model
        from miniprophet.tools.search import get_search_backend

        model = get_model(config=self._config.get("model", {}))

        search_cfg = self._config.get("search", {})
        search_backend = get_search_backend(search_cfg=search_cfg)

        agent_search_limit = self._config.get("agent", {}).get("search_limit", 10)
        max_gist = search_cfg.get("max_source_display_chars", 200)
        registry = SourceRegistry(max_gist_chars=max_gist)
        tools = create_default_tools(
            search_tool=search_backend,
            registry=registry,
            search_limit=agent_search_limit,
            search_results_limit=search_cfg.get("search_results_limit", 5),
        )

        # AskUser callback: posts question, waits for user input via queue
        async def _tui_ask_user(question: str) -> str:
            from miniprophet.cli.utils import get_console as _get_console

            console = _get_console()
            console.print(f"\n  [bold cyan]Agent asks:[/bold cyan] {question}")
            self.call_from_thread(self._set_status, "Agent is asking a question — type your answer")
            try:
                return self._interrupt_queue.get(timeout=300)
            except Exception:
                return "(No response from user — timed out)"

        planning_tools = create_planning_tools(ask_user_callback=_tui_ask_user)
        env = ForecastEnvironment(tools, planning_tools=planning_tools, registry=registry)

        cm_cfg = self._config.get("context_manager", {})
        ctx_mgr = get_context_manager(cm_cfg)

        return TuiForecastAgent(
            model=model,
            env=env,
            context_manager=ctx_mgr,
            message_queue=self._interrupt_queue,
            on_status=lambda text: self.call_from_thread(self._set_status, text),
            **self._config.get("agent", {}),
        )
