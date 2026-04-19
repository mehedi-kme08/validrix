"""
Unit tests for retry_manager — fast, no I/O, no external deps.
"""

from __future__ import annotations

import pytest

import validrix.core.retry_manager as retry_manager_module
from validrix.core.retry_manager import RetryConfig, RetryManager, retry


def test_retry_config_default_values() -> None:
    cfg = RetryConfig()
    assert cfg.max_attempts == 3
    assert cfg.delay_seconds == 1.0
    assert cfg.backoff_multiplier == 2.0
    assert cfg.jitter is True


def test_retry_config_rejects_zero_attempts() -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        RetryConfig(max_attempts=0)


def test_retry_config_rejects_low_multiplier() -> None:
    with pytest.raises(ValueError, match="backoff_multiplier"):
        RetryConfig(backoff_multiplier=0.5)


def test_retry_config_rejects_negative_delay() -> None:
    with pytest.raises(ValueError, match="delay_seconds"):
        RetryConfig(delay_seconds=-0.1)


def test_decorator_preserves_function_name() -> None:
    @retry(max_attempts=1)
    def my_function() -> None:
        pass

    assert my_function.__name__ == "my_function"


def test_decorator_preserves_return_value() -> None:
    @retry(max_attempts=1)
    def returns_42() -> int:
        return 42

    assert returns_42() == 42


def test_manager_delegates_to_retry() -> None:
    calls: list[int] = []

    def succeed_on_third() -> str:
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("not yet")
        return "ok"

    manager = RetryManager(RetryConfig(max_attempts=5, delay_seconds=0.001, jitter=False))
    result = manager.execute(succeed_on_third)
    assert result == "ok"
    assert len(calls) == 3


def test_retry_re_raises_last_exception_and_uses_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(retry_manager_module.random, "uniform", lambda start, end: 0.0)
    monkeypatch.setattr(retry_manager_module.time, "sleep", lambda seconds: sleeps.append(seconds))

    @retry(max_attempts=2, delay_seconds=0.1, jitter=True)
    def always_fails() -> None:
        raise RuntimeError("still broken")

    with pytest.raises(RuntimeError, match="still broken"):
        always_fails()

    assert sleeps == [0.1]


def test_retry_handles_invalid_constructed_config_branch() -> None:
    invalid = object.__new__(RetryConfig)
    invalid.max_attempts = 0
    invalid.delay_seconds = 0.0
    invalid.backoff_multiplier = 1.0
    invalid.jitter = False
    invalid.exceptions = [Exception]

    @retry(config=invalid)
    def never_called() -> None:
        raise RuntimeError("unreachable")

    with pytest.raises(AssertionError):
        never_called()
