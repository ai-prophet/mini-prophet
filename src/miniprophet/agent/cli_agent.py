"""CliForecastAgent: CLI-aware agent with Rich display hooks."""

from __future__ import annotations

from rich.live import Live
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

    # ------------------------------------------------------------------
    # Hook overrides
    # ------------------------------------------------------------------

    def on_run_start(self, title: str, outcomes: str, config: AgentConfig) -> None:
        print_run_header(title, outcomes, config.step_limit, config.cost_limit, config.search_limit)

    def on_step_start(
        self,
        step: int,
        model_cost: float,
        search_cost: float,
        total_cost: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        max_context_tokens: int | None = None,
    ) -> None:
        print_step_header(
            step, model_cost, search_cost, total_cost,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            max_context_tokens=max_context_tokens,
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
    # Override step to show context truncation notice
    # ------------------------------------------------------------------

    def step(self) -> list[dict]:
        result = super().step()

        if self.context_manager is not None:
            self.context_manager.display()
        return result

    # ------------------------------------------------------------------
    # Override query to show a spinner while waiting for the model
    # ------------------------------------------------------------------

    def query(self) -> dict:
        model_name = getattr(self.model.config, "model_name", "model")
        spinner = Spinner("dots", text=f"  {model_name} is forecasting...")
        with Live(spinner, console=console, transient=True):
            return super().query()
