"""
Example: Retry decorator and RetryManager.

Demonstrates all retry features: decorator syntax, config object,
exponential backoff, and exception filtering.
"""

from __future__ import annotations

import pytest

from validrix.core.retry_manager import RetryConfig, RetryManager, retry


class TestRetryDecorator:
    """Tests for the @retry decorator API."""

    def test_succeeds_on_first_attempt(self) -> None:
        """No retries needed when function succeeds immediately."""
        calls: list[int] = []

        @retry(max_attempts=3)
        def always_succeeds() -> str:
            calls.append(1)
            return "ok"

        result = always_succeeds()
        assert result == "ok"
        assert len(calls) == 1, "Should not retry on success"

    def test_retries_on_failure_then_succeeds(self) -> None:
        """Retry until success within max_attempts."""
        attempts: list[int] = []

        @retry(max_attempts=5, delay_seconds=0.001, jitter=False)
        def fails_twice_then_passes() -> str:
            attempts.append(1)
            if len(attempts) < 3:
                raise ValueError("Temporary failure")
            return "recovered"

        result = fails_twice_then_passes()
        assert result == "recovered"
        assert len(attempts) == 3

    def test_raises_after_max_attempts_exhausted(self) -> None:
        """Raise the last exception when all attempts fail."""

        @retry(max_attempts=3, delay_seconds=0.001, jitter=False)
        def always_fails() -> None:
            raise RuntimeError("Permanent failure")

        with pytest.raises(RuntimeError, match="Permanent failure"):
            always_fails()

    def test_only_retries_specified_exception_types(self) -> None:
        """Exceptions not in the list should propagate immediately."""
        calls: list[int] = []

        @retry(max_attempts=5, delay_seconds=0.001, exceptions=(ValueError,))
        def raises_type_error() -> None:
            calls.append(1)
            raise TypeError("Not a ValueError — should not be retried")

        with pytest.raises(TypeError):
            raises_type_error()

        assert len(calls) == 1, "TypeError should not trigger retries"

    def test_retry_config_object(self) -> None:
        """Config object takes precedence over individual parameters."""
        calls: list[int] = []
        policy = RetryConfig(max_attempts=4, delay_seconds=0.001, jitter=False)

        @retry(config=policy)
        def fails_three_times() -> str:
            calls.append(1)
            if len(calls) < 4:
                raise OSError("IO error")
            return "done"

        result = fails_three_times()
        assert result == "done"
        assert len(calls) == 4


class TestRetryManager:
    """Tests for the stateful RetryManager class."""

    def test_execute_callable_with_retry(self) -> None:
        """RetryManager.execute wraps a zero-arg callable."""
        attempts: list[int] = []

        def flaky() -> int:
            attempts.append(1)
            if len(attempts) < 2:
                raise ConnectionError("Not ready")
            return 42

        manager = RetryManager(RetryConfig(max_attempts=3, delay_seconds=0.001))
        result = manager.execute(flaky)
        assert result == 42
        assert len(attempts) == 2

    def test_config_validation_rejects_zero_attempts(self) -> None:
        """RetryConfig should reject max_attempts < 1."""
        with pytest.raises(ValueError):
            RetryConfig(max_attempts=0)

    def test_config_validation_rejects_negative_delay(self) -> None:
        """RetryConfig should reject negative delay_seconds."""
        with pytest.raises(ValueError):
            RetryConfig(delay_seconds=-1.0)
