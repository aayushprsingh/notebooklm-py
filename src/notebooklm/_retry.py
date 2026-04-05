"""Retry utilities with exponential backoff for transient errors.

Provides decorators and helpers for retrying operations that fail due to
network issues, rate limits, or temporary server errors.

Example:
    from notebooklm._retry import retry_on_transient_errors

    @retry_on_transient_errors(max_attempts=3)
    async def flaky_operation():
        ...

    # Or with custom settings:
    @retry_on_transient_errors(
        initial_delay=1.0,
        max_delay=30.0,
        backoff_factor=2.0,
        jitter=True,
    )
    async def another_operation():
        ...
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, TypeVar

from .exceptions import (
    NetworkError,
    RateLimitError,
    RPCTimeoutError,
    ServerError,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

T = TypeVar("T")


# =============================================================================
# Retry Decorator
# =============================================================================


def retry_on_transient_errors(
    *,
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 30.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    retriable_errors: tuple[type[Exception], ...] | None = None,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Decorator that retries an async function on transient errors.

    Retries on:
    - NetworkError, RPCTimeoutError (connection/timeout issues)
    - RateLimitError (with respecting retry_after if available)
    - ServerError (5xx responses)

    Does NOT retry:
    - AuthError / authentication failures
    - ClientError (4xx other than rate limit)
    - ValidationError, ConfigurationError

    Args:
        max_attempts: Maximum number of attempts (default 3).
        initial_delay: Initial delay in seconds before first retry (default 1.0).
        max_delay: Maximum delay in seconds between retries (default 30.0).
        backoff_factor: Multiplier for delay after each retry (default 2.0).
        jitter: Add random jitter to delays to avoid thundering herd (default True).
        retriable_errors: Additional exception types to retry on. The standard
            set (NetworkError, RPCTimeoutError, RateLimitError, ServerError)
            is always included.

    Returns:
        Decorated function that retries on transient errors.

    Example:
        @retry_on_transient_errors(max_attempts=4, initial_delay=0.5)
        async def get_data():
            return await client.notebooks.get(notebook_id)
    """
    _retriable: tuple[type[Exception], ...] = (
        NetworkError,
        RPCTimeoutError,
        RateLimitError,
        ServerError,
    )
    if retriable_errors:
        _retriable = _retriable + retriable_errors

    def decorator(fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Exception | None = None
            delay = initial_delay

            for attempt in range(1, max_attempts + 1):
                try:
                    return await fn(*args, **kwargs)
                except _retriable as e:
                    last_exception = e

                    if isinstance(e, RateLimitError) and e.retry_after is not None:
                        delay = float(e.retry_after)
                        logger.warning(
                            "Rate limited — using server-suggested retry-after "
                            "of %.1fs (attempt %d/%d)",
                            delay,
                            attempt,
                            max_attempts,
                        )
                    elif attempt < max_attempts:
                        logger.warning(
                            "Transient error (%s) on attempt %d/%d — "
                            "retrying in %.1fs",
                            type(e).__name__,
                            attempt,
                            max_attempts,
                            delay,
                        )

                    if attempt < max_attempts:
                        await asyncio.sleep(delay)
                        # Apply backoff AFTER sleeping for the NEXT iteration
                        if not (isinstance(e, RateLimitError) and e.retry_after is not None):
                            delay *= backoff_factor
                            if jitter:
                                delay *= 0.75 + random.random() * 0.5
                            delay = min(delay, max_delay)
                    else:
                        logger.error(
                            "All %d attempts failed for %s (last error: %s)",
                            max_attempts,
                            fn.__name__,
                            e,
                        )

            # All attempts exhausted
            if last_exception is not None:
                raise last_exception
            raise RuntimeError(f"All {max_attempts} attempts failed for {fn.__name__}")

        return wrapper

    return decorator


# =============================================================================
# Retry Context Manager
# =============================================================================


class RetryManager:
    """Manual retry loop context manager for fine-grained control.

    Use this when you need to retry a block of code that doesn't map cleanly
    to a single function call, or when you need access to the result each attempt.

    Example:
        retry = RetryManager(max_attempts=3, initial_delay=1.0)
        async with retry:
            while retry.attempt < retry.max_attempts:
                try:
                    result = await client.sources.add_url(nb_id, url)
                    retry.complete()
                    break
                except RateLimitError as e:
                    if not retry.on_transient_error(e):
                        raise
                    await retry.sleep_before_retry()
    """

    def __init__(
        self,
        *,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 30.0,
        backoff_factor: float = 2.0,
        jitter: bool = True,
    ):
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter

        self.attempt: int = 0
        self._current_delay: float = initial_delay
        self._done: bool = False

    async def __aenter__(self) -> RetryManager:
        self.attempt = 0
        self._current_delay = self.initial_delay
        self._done = False
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        return False  # Don't suppress exceptions

    @property
    def exhausted(self) -> bool:
        """True if all retry attempts have been used."""
        return self.attempt >= self.max_attempts

    def on_transient_error(self, error: Exception) -> bool:
        """Called when a transient error occurs.

        Records the error, applies backoff, and returns True if more
        attempts remain (so the caller knows to keep retrying).

        Returns:
            True if retries remain; False if all attempts exhausted.
        """
        if isinstance(error, RateLimitError) and error.retry_after is not None:
            self._current_delay = float(error.retry_after)
        else:
            self._current_delay = min(
                self._current_delay * self.backoff_factor, self.max_delay
            )
            if self.jitter:
                self._current_delay *= 0.75 + random.random() * 0.5

        self.attempt += 1
        if self.attempt < self.max_attempts:
            logger.warning(
                "Transient error on attempt %d/%d — retrying in %.1fs (%s)",
                self.attempt,
                self.max_attempts,
                self._current_delay,
                type(error).__name__,
            )
            return True
        return False

    async def sleep_before_retry(self) -> None:
        """Sleep for the current backoff delay. Call after on_transient_error."""
        await asyncio.sleep(self._current_delay)

    def complete(self) -> None:
        """Mark the operation as successfully completed."""
        self._done = True


# =============================================================================
# Convenience Helpers
# =============================================================================


async def retry_with_backoff(
    coro_fn: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 30.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
) -> T:
    """Simple retry wrapper for a single coroutine.

    Wraps any awaitable-factory and retries it on transient errors with
    exponential backoff. Respects RateLimitError.retry_after when available.

    Args:
        coro_fn: A callable that returns the coroutine to execute and retry.
            Called fresh on each attempt so the coroutine is never reused.
        max_attempts: Maximum number of attempts (default 3).
        initial_delay: Initial delay in seconds (default 1.0).
        max_delay: Maximum delay in seconds (default 30.0).
        backoff_factor: Exponential backoff multiplier (default 2.0).
        jitter: Add random jitter to delays (default True).

    Returns:
        The result of the coroutine on success.

    Raises:
        The last exception if all attempts fail.

    Example:
        result = await retry_with_backoff(
            lambda: client.notebooks.get(notebook_id),
            max_attempts=4,
        )
    """
    _retriable: tuple[type[Exception], ...] = (
        NetworkError,
        RPCTimeoutError,
        RateLimitError,
        ServerError,
    )

    delay = initial_delay
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            return await coro_fn()
        except _retriable as e:
            last_error = e

            if attempt < max_attempts:
                logger.warning(
                    "Retry %s — attempt %d/%d, sleeping %.1fs",
                    type(e).__name__,
                    attempt,
                    max_attempts,
                    delay,
                )
                await asyncio.sleep(delay)
                # Apply backoff AFTER sleeping for the next iteration
                if isinstance(e, RateLimitError) and e.retry_after is not None:
                    delay = float(e.retry_after)
                else:
                    delay *= backoff_factor
                    if jitter:
                        delay *= 0.75 + random.random() * 0.5
                    delay = min(delay, max_delay)
            else:
                logger.error(
                    "All %d attempts failed (last: %s)",
                    max_attempts,
                    e,
                )

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"All {max_attempts} attempts failed")
