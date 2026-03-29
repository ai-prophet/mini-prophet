"""Exception hierarchy for mini-prophet.

InterruptAgentFlow and its subclasses carry message dicts as payload,
caught by the agent loop and injected into the conversation.

Search errors are a separate hierarchy, caught inside ForecastEnvironment
and translated into error observations for the model.
"""


class InterruptAgentFlow(Exception):
    """Raised to interrupt the agent flow and inject messages."""

    def __init__(self, *messages: dict):
        self.messages = messages
        super().__init__()


class Submitted(InterruptAgentFlow):
    """Raised when the agent submits a forecast -- terminates the loop."""


class PlanSubmitted(InterruptAgentFlow):
    """Raised when the agent submits a validated plan -- terminates the planning loop."""

    def __init__(self, plan_xml: str, *messages: dict):
        self.plan_xml = plan_xml
        super().__init__(*messages)


class LimitsExceeded(InterruptAgentFlow):
    """Raised when step, cost, or search limits are exceeded."""


class FormatError(InterruptAgentFlow):
    """Raised when the model's output is not a valid tool call."""


class SearchError(Exception):
    """Base class for search tool errors."""


class SearchAuthError(SearchError):
    """Authentication failure -- fatal, should abort the agent."""


class SearchRateLimitError(SearchError):
    """Rate limit hit -- retryable."""


class SearchNetworkError(SearchError):
    """Network/connection error -- retryable."""


"""
Batch-processing related errors.
"""


class BatchRunTimeoutError(Exception):
    """Raised when a single batch run exceeds the configured timeout."""


class BatchFatalError(Exception):
    """Raised when batch execution must terminate immediately."""
