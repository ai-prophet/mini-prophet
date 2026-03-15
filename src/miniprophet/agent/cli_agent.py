"""CliForecastAgent: CLI-aware agent with Rich display hooks."""

from __future__ import annotations

import signal

from rich.live import Live
from rich.prompt import Prompt
from rich.spinner import Spinner

from miniprophet import ContextManager, Environment, Model
from miniprophet.agent.default import AgentConfig, DefaultForecastAgent, ForecastResult
from miniprophet.cli.components.evaluation import print_evaluation
from miniprophet.cli.components.forecast_results import print_forecast_results
from miniprophet.cli.components.observation import print_observation
from miniprophet.cli.components.run_header import print_run_footer, print_run_header
from miniprophet.cli.components.step_display import print_model_response, print_step_header
from miniprophet.cli.utils import get_console

console = get_console()


class CliAgentConfig(AgentConfig):
    show_thinking: bool = True
    max_display_chars: int = 500
    enable_interrupt: bool = True


class CliForecastAgent(DefaultForecastAgent):
    """Extends DefaultForecastAgent with Rich CLI display via hook overrides."""

    def __init__(
        self,
        model: Model,
        env: Environment,
        *,
        context_manager: ContextManager | None = None,
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
        # For interactive mode signal handling
        self._interrupt_requested: bool = False
        self._original_sigint_handler: signal.Handlers | None = None

    # ------------------------------------------------------------------
    # Hook overrides
    # ------------------------------------------------------------------

    def on_run_start(self, title: str, outcomes: str, config: AgentConfig) -> None:
        print_run_header(title, outcomes, config.step_limit, config.cost_limit, config.search_limit)

    def on_step_start(self) -> None:
        print_step_header(
            self.n_calls,
            self.model_cost,
            self.search_cost,
            self.total_cost,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            max_context_tokens=self.max_context_tokens,
        )

    def on_model_response(self, message: dict) -> None:
        max_chars = (
            self.config.max_display_chars if hasattr(self.config, "max_display_chars") else 500
        )
        print_model_response(message, max_thinking_chars=max_chars)

    def on_observation(self, action: dict, output: dict) -> None:
        tool_name = action.get("name", "")

        tool = self.env._tools.get(tool_name)
        if tool is not None:
            tool.display(output)
            return

        print_observation(output)

    def on_run_end(self, result: ForecastResult) -> None:
        submission = result.get("submission", {})
        if submission:
            print_forecast_results(submission)

        evaluation = result.get("evaluation", {})
        if evaluation:
            print_evaluation(evaluation)

        print_run_footer(
            result.get("exit_status", "unknown"),
            self.n_calls,
            self.n_searches,
            self.model_cost,
            self.search_cost,
            self.total_cost,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            max_context_tokens=self.max_context_tokens,
        )

    # ------------------------------------------------------------------
    # Interactive interrupt: signal handling
    # ------------------------------------------------------------------

    def _handle_sigint(self, signum: int, frame: object) -> None:
        """Custom SIGINT handler: first Ctrl+C sets flag, second hard-aborts."""
        if self._interrupt_requested:
            # Second Ctrl+C: restore default handler and hard-abort.
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            console.print("\n[bold red]  Hard abort (double Ctrl+C).[/bold red]")
            raise KeyboardInterrupt
        self._interrupt_requested = True
        console.print(
            "\n[bold yellow]  Interrupt received. "
            "Will pause after current operation...[/bold yellow]"
        )

    def _prompt_user_message(self) -> None:
        """Pause the agent and prompt the user for a message to inject."""
        self._interrupt_requested = False

        console.print()
        console.rule("[bold yellow]Agent Paused[/bold yellow]", style="yellow")
        console.print(
            "  [dim]Type your message to the agent, or press Enter to resume without a message.[/dim]"
        )
        console.print("  [dim]Press Ctrl+C again to abort.[/dim]")

        # Temporarily restore original SIGINT so Ctrl+C during input aborts.
        if self._original_sigint_handler is not None:
            signal.signal(signal.SIGINT, self._original_sigint_handler)

        try:
            user_input = Prompt.ask("  [bold cyan]Message[/bold cyan]", default="")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold red]  Aborting.[/bold red]")
            raise KeyboardInterrupt
        finally:
            # Re-install our handler for the resumed run.
            if self._original_sigint_handler is not None:
                signal.signal(signal.SIGINT, self._handle_sigint)

        if user_input.strip():
            self.add_messages(
                self.model.format_message(
                    role="user",
                    content=user_input.strip(),
                    extra={"is_user_interrupt": True},
                )
            )
            console.print("  [green]Message injected. Resuming...[/green]")
        else:
            console.print("  [dim]No message. Resuming...[/dim]")

        console.rule(style="yellow")
        console.print()

    # ------------------------------------------------------------------
    # Override run to install/restore signal handler
    # ------------------------------------------------------------------

    async def run(
        self,
        title: str,
        outcomes: list[str],
        ground_truth: dict[str, int] | None = None,
        **runtime_kwargs,
    ) -> ForecastResult:
        interactive = getattr(self.config, "enable_interrupt", False)
        if interactive:
            self._original_sigint_handler = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, self._handle_sigint)
        try:
            return await super().run(title, outcomes, ground_truth, **runtime_kwargs)
        finally:
            if self._original_sigint_handler is not None:
                signal.signal(signal.SIGINT, self._original_sigint_handler)
                self._original_sigint_handler = None

    # ------------------------------------------------------------------
    # Override step: interrupt check points + context truncation display
    # ------------------------------------------------------------------

    async def step(self) -> list[dict]:
        self._prepare_messages_for_step()

        message = await self.query()

        has_actions = bool(message.get("extra", {}).get("actions", []))

        # Interrupt after query, no tool actions: A -> C
        if self._interrupt_requested and not has_actions:
            self._prompt_user_message()
            if self.context_manager is not None:
                self.context_manager.display()
            return list(self.messages)

        # Execute tool actions
        result = await self.execute_actions(message)

        # Interrupt after tool execution: A -> B -> C
        if self._interrupt_requested:
            self._prompt_user_message()

        if self.context_manager is not None:
            self.context_manager.display()

        return result

    # ------------------------------------------------------------------
    # Override query to show a spinner while waiting for the model
    # ------------------------------------------------------------------

    async def query(self) -> dict:
        model_name = getattr(self.model.config, "model_name", "model")
        spinner = Spinner("dots", text=f"  {model_name} is forecasting...")
        with Live(spinner, console=console, transient=True):
            return await super().query()
