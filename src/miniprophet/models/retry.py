"""Retry utility for model queries, thin wrapper around tenacity."""

import logging
import os

from tenacity import (
    AsyncRetrying,
    Retrying,
    before_sleep_log,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)


def _retry_kwargs(*, logger: logging.Logger, abort_exceptions: list[type[Exception]]) -> dict:
    return dict(
        reraise=True,
        stop=stop_after_attempt(int(os.getenv("MINIPROPHET_MODEL_RETRY_ATTEMPTS", "10"))),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_not_exception_type(tuple(abort_exceptions)),
    )


def retry(*, logger: logging.Logger, abort_exceptions: list[type[Exception]]) -> Retrying:
    """Return a tenacity Retrying object with exponential backoff.

    Retries all exceptions except those in abort_exceptions.
    """
    return Retrying(**_retry_kwargs(logger=logger, abort_exceptions=abort_exceptions))


def async_retry(
    *, logger: logging.Logger, abort_exceptions: list[type[Exception]]
) -> AsyncRetrying:
    """Return a tenacity AsyncRetrying object with exponential backoff.

    Async counterpart of :func:`retry`.
    """
    return AsyncRetrying(**_retry_kwargs(logger=logger, abort_exceptions=abort_exceptions))
