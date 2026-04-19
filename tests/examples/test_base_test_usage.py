"""
Example: Using BaseTest for structured test classes.

Demonstrates:
  - Inheriting from BaseTest for config + retry + soft-assert access
  - assert_eventually for polling-based assertions
  - RetryManager for wrapping unreliable calls
"""

from __future__ import annotations

import pytest

from validrix.core.base_test import BaseTest


class TestUserRegistration(BaseTest):
    """
    Example test class demonstrating BaseTest features.

    In a real project these tests would call your application's API;
    here we keep them self-contained so they run without external deps.
    """

    def setup_method(self, method: object) -> None:
        super().setup_method(method)
        self._registered_users: list[str] = []

    def test_valid_registration_succeeds(self) -> None:
        """Happy path: registering with valid data should succeed."""
        # Arrange
        email = "alice@example.com"
        password = "S3cure!Pass"

        # Act — simulate a registration call
        success = self._mock_register(email, password)

        # Assert
        assert success, f"Registration should succeed for {email}"
        self.soft_assert(email in self._registered_users, "User should be in registry")
        self.assert_soft_failures()

    @pytest.mark.parametrize(
        "email,password,reason",
        [
            ("", "S3cure!Pass", "empty email"),
            ("not-an-email", "S3cure!Pass", "malformed email"),
            ("alice@test.com", "", "empty password"),
            ("alice@test.com", "short", "password too short"),
        ],
    )
    def test_invalid_inputs_are_rejected(
        self,
        email: str,
        password: str,
        reason: str,
    ) -> None:
        """Negative path: invalid inputs must be rejected."""
        success = self._mock_register(email, password)
        assert not success, f"Registration should fail for: {reason}"

    def test_assert_eventually_polling(self) -> None:
        """assert_eventually waits for a condition to become true."""
        counter = {"value": 0}

        def increment_and_check() -> bool:
            counter["value"] += 1
            return counter["value"] >= 3

        # Should pass within 2 s (counter reaches 3 in ~1.5 polls)
        self.assert_eventually(
            increment_and_check,
            timeout_seconds=5.0,
            poll_interval=0.1,
            message="Counter should reach 3 within 5 seconds",
        )
        assert counter["value"] >= 3

    def test_retry_manager_recovers_from_transient_failure(self) -> None:
        """RetryManager should succeed on the 3rd attempt."""
        attempts = {"count": 0}

        def unreliable_call() -> str:
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise ConnectionError("Transient network error")
            return "success"

        from validrix.core.retry_manager import RetryConfig, RetryManager

        manager = RetryManager(RetryConfig(max_attempts=5, delay_seconds=0.01, jitter=False))
        result = manager.execute(unreliable_call)
        assert result == "success"
        assert attempts["count"] == 3, "Should have taken exactly 3 attempts"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _mock_register(self, email: str, password: str) -> bool:
        """Simulate a registration API without real HTTP calls."""
        if not email or "@" not in email:
            return False
        if not password or len(password) < 8:
            return False
        self._registered_users.append(email)
        return True
