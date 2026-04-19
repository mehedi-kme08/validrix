"""
base_test.py — Abstract base class for all Validrix test suites.

Design decision: ABC over plain class.
  WHY: Marking BaseTest as abstract (via ABC) prevents instantiation and
       makes it clear that subclasses must own the setup/teardown lifecycle.
       It also forces IDEs to surface missing method implementations at write
       time rather than at runtime.

  Alternatives considered:
    - pytest fixtures only (no base class): composable but verbose for teams
      that want a unified "just extend this class" DX.
    - dataclass: lightweight but no abstract method enforcement.

  Tradeoffs:
    - Inheriting from a base class creates coupling. We mitigate this by
      keeping BaseTest thin — no heavy state, no Playwright references.
    - Teams that prefer fixture composition can skip BaseTest entirely;
      the rest of the framework works without it.

Lifecycle hooks (called by pytest's xunit-style setup/teardown):
  setup_method()     → per-test initialisation
  teardown_method()  → per-test cleanup (always runs, even on failure)
  setup_class()      → once per class
  teardown_class()   → once per class
"""

from __future__ import annotations

import logging
from abc import ABC
from typing import Any

import pytest

from validrix.core.config_manager import ConfigManager, FrameworkConfig
from validrix.core.retry_manager import RetryConfig, RetryManager

logger = logging.getLogger(__name__)


class BaseTest(ABC):
    """
    Abstract base for all Validrix test classes.

    Provides:
    - Typed access to framework config via ``self.config``
    - Pre-configured ``self.retry`` manager
    - ``self.log`` logger namespaced to the concrete subclass
    - ``soft_assert`` for collecting multiple failures in one test
    - ``assert_eventually`` for polling-based assertions

    Subclass example::

        class TestLogin(BaseTest):
            def setup_method(self, method):
                super().setup_method(method)
                self.page = self.browser.new_page()

            def test_valid_login(self):
                self.page.goto(self.config.env.base_url + "/login")
                self.soft_assert(self.page.title() == "Login", "Page title mismatch")
                self.assert_soft_failures()
    """

    # Populated by setup_method — available for the lifetime of each test
    config: FrameworkConfig
    retry: RetryManager
    log: logging.Logger
    _soft_failures: list[str]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup_method(self, method: Any) -> None:
        """Per-test setup — called automatically by pytest before each test."""
        self.config = ConfigManager.load()
        self.log = logging.getLogger(type(self).__qualname__)
        self.retry = RetryManager(RetryConfig(
            max_attempts=self.config.retry.max_attempts,
            delay_seconds=self.config.retry.delay_seconds,
            backoff_multiplier=self.config.retry.backoff_multiplier,
            jitter=self.config.retry.jitter,
        ))
        self._soft_failures = []
        self.log.info("Starting test: %s", method.__name__ if callable(method) else method)

    def teardown_method(self, method: Any) -> None:
        """Per-test teardown — called automatically by pytest after each test."""
        self.log.info(
            "Finished test: %s",
            method.__name__ if callable(method) else method,
        )

    # ------------------------------------------------------------------
    # Soft assertions
    # ------------------------------------------------------------------

    def soft_assert(self, condition: bool, message: str = "") -> None:
        """
        Record an assertion failure without immediately stopping the test.

        Failures are collected and surfaced by ``assert_soft_failures()``.
        Use when you want to check multiple conditions in a single test
        and see ALL failures at once rather than stopping at the first.

        Args:
            condition: The boolean expression to assert.
            message:   Human-readable description of what was expected.
        """
        if not condition:
            self._soft_failures.append(message or "Unnamed soft assertion failed")
            self.log.warning("Soft assertion failed: %s", message)

    def assert_soft_failures(self) -> None:
        """
        Raise AssertionError if any soft assertions have failed.

        Call this at the end of a test that used ``soft_assert()``.

        Raises:
            AssertionError: Summarising all collected failures.
        """
        if self._soft_failures:
            summary = "\n  - ".join(self._soft_failures)
            pytest.fail(f"{len(self._soft_failures)} soft assertion(s) failed:\n  - {summary}")

    # ------------------------------------------------------------------
    # Polling assertion
    # ------------------------------------------------------------------

    def assert_eventually(
        self,
        condition: Any,
        timeout_seconds: float = 10.0,
        poll_interval: float = 0.5,
        message: str = "",
    ) -> None:
        """
        Poll ``condition`` until it is truthy or ``timeout_seconds`` elapses.

        Useful for eventually-consistent state (e.g., waiting for a background
        job to complete, a UI element to appear, a DB row to update).

        Args:
            condition:       Callable → bool, or a bool value.
            timeout_seconds: How long to keep polling before failing.
            poll_interval:   Seconds between polls.
            message:         Failure message if timeout is reached.

        Raises:
            AssertionError: If condition is never truthy within the timeout.
        """
        import time

        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            result = condition() if callable(condition) else condition
            if result:
                return
            time.sleep(poll_interval)

        pytest.fail(message or f"Condition not met within {timeout_seconds}s")

