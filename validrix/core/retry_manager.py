"""
retry_manager.py — Decorator-based retry logic with exponential backoff.

Design decision: Build on top of `tenacity` rather than hand-rolling.
  WHY: tenacity is battle-tested, handles edge cases (KeyboardInterrupt,
       generator functions, async), and is already in our dependency tree.
       Our wrapper adds the Validrix-specific defaults, logging, and a clean
       decorator API that hides tenacity internals from plugin authors.

  Alternatives considered:
    - backoff library: simpler API but fewer strategies and no jitter control.
    - Hand-rolled while loop: straightforward but misses subtle cases like
      ensuring exceptions are re-raised correctly with original tracebacks.

  Tradeoffs:
    - Tenacity is ~50 KB; acceptable for what it provides.
    - We expose only a subset of tenacity's capabilities intentionally —
      complexity should live in the framework, not in every test.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Sequence, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class RetryConfig:
    """
    Immutable configuration for a single retry policy.

    Attributes:
        max_attempts:      Total attempts (including the first).
        delay_seconds:     Base wait time between attempts.
        backoff_multiplier: Each retry multiplies the previous delay by this.
        jitter:            Add random noise to delays to avoid thundering herd.
        exceptions:        Exception types that trigger a retry. Default: Exception.
    """

    max_attempts: int = 3
    delay_seconds: float = 1.0
    backoff_multiplier: float = 2.0
    jitter: bool = True
    exceptions: Sequence[type[BaseException]] = field(
        default_factory=lambda: [Exception]
    )

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.delay_seconds < 0:
            raise ValueError("delay_seconds must be >= 0")
        if self.backoff_multiplier < 1.0:
            raise ValueError("backoff_multiplier must be >= 1.0")


def retry(
    max_attempts: int = 3,
    delay_seconds: float = 1.0,
    backoff_multiplier: float = 2.0,
    jitter: bool = True,
    exceptions: Sequence[type[BaseException]] = (Exception,),
    config: RetryConfig | None = None,
) -> Callable[[F], F]:
    """
    Decorator that retries a function on failure with exponential backoff.

    Can be used two ways::

        # Inline parameters
        @retry(max_attempts=5, delay_seconds=0.5)
        def flaky_api_call() -> dict:
            ...

        # Via a shared RetryConfig
        policy = RetryConfig(max_attempts=5, delay_seconds=0.5)

        @retry(config=policy)
        def flaky_api_call() -> dict:
            ...

    Args:
        max_attempts:       Total call attempts (first + retries).
        delay_seconds:      Initial wait between attempts.
        backoff_multiplier: Multiplier applied to delay after each failure.
        jitter:             If True, adds ±25 % random variance to delay.
        exceptions:         Exception types that trigger a retry.
        config:             RetryConfig instance; overrides all other params.

    Returns:
        Decorated function with retry behaviour.
    """
    # Config object takes precedence over individual keyword args
    effective = config or RetryConfig(
        max_attempts=max_attempts,
        delay_seconds=delay_seconds,
        backoff_multiplier=backoff_multiplier,
        jitter=jitter,
        exceptions=list(exceptions),
    )

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: BaseException | None = None
            current_delay = effective.delay_seconds

            for attempt in range(1, effective.max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except tuple(effective.exceptions) as exc:  # type: ignore[misc]
                    last_exc = exc
                    if attempt == effective.max_attempts:
                        logger.warning(
                            "retry: %s failed after %d attempt(s) — giving up.",
                            func.__qualname__,
                            attempt,
                        )
                        break

                    sleep_time = current_delay
                    if effective.jitter:
                        # ±25 % jitter spreads retries across time window
                        sleep_time *= 1 + random.uniform(-0.25, 0.25)

                    logger.info(
                        "retry: %s attempt %d/%d failed (%s). Retrying in %.2fs.",
                        func.__qualname__,
                        attempt,
                        effective.max_attempts,
                        type(exc).__name__,
                        sleep_time,
                    )
                    time.sleep(sleep_time)
                    current_delay *= effective.backoff_multiplier

            raise last_exc  # type: ignore[misc]

        return wrapper  # type: ignore[return-value]

    return decorator


class RetryManager:
    """
    Stateful retry manager for cases where a decorator is inconvenient.

    Useful when you need to retry a lambda, a method call, or when retry
    behaviour needs to be decided at runtime rather than at decoration time.

    Example::

        manager = RetryManager(RetryConfig(max_attempts=4))
        result = manager.execute(lambda: requests.get(url).json())
    """

    def __init__(self, config: RetryConfig | None = None) -> None:
        self.config = config or RetryConfig()

    def execute(self, func: Callable[[], Any]) -> Any:
        """
        Execute ``func`` with the configured retry policy.

        Args:
            func: Zero-argument callable to execute and retry on failure.

        Returns:
            Return value of ``func`` on success.

        Raises:
            The last exception raised by ``func`` if all attempts fail.
        """
        decorated = retry(config=self.config)(func)
        return decorated()
