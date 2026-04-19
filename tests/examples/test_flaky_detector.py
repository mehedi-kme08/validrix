"""
Example: FlakyDetectorPlugin and flakiness metrics.

Demonstrates:
  - FlakinessLabel classification
  - FlakinessMetric.compute() pass-rate calculation
  - Flakiness score formula (peaks at 0.5 pass rate)
"""

from __future__ import annotations

import pytest

from validrix.plugins.flaky_detector import (
    FlakinessLabel,
    FlakinessMetric,
    RunResult,
)


class TestFlakinessComputation:
    """Unit tests for flakiness metric calculation."""

    def _make_results(self, passed: list[bool]) -> list[RunResult]:
        return [
            RunResult(run_number=i + 1, passed=p, duration_seconds=0.1)
            for i, p in enumerate(passed)
        ]

    def test_all_passing_is_stable(self) -> None:
        """A test that always passes should be STABLE with score 0.0."""
        results = self._make_results([True, True, True])
        metric = FlakinessMetric.compute("tests/test_foo.py::test_bar", results)

        assert metric.label == FlakinessLabel.STABLE
        assert metric.pass_rate == 1.0
        assert metric.flakiness_score == 0.0

    def test_all_failing_is_failing(self) -> None:
        """A test that always fails should be FAILING with score 0.0."""
        results = self._make_results([False, False, False])
        metric = FlakinessMetric.compute("tests/test_foo.py::test_bar", results)

        assert metric.label == FlakinessLabel.FAILING
        assert metric.pass_rate == 0.0
        assert metric.flakiness_score == 0.0

    def test_mixed_results_are_flaky(self) -> None:
        """A test with mixed pass/fail should be FLAKY."""
        results = self._make_results([True, False, True])
        metric = FlakinessMetric.compute("tests/test_foo.py::test_bar", results)

        assert metric.label == FlakinessLabel.FLAKY
        assert metric.passed_runs == 2
        assert metric.failed_runs == 1
        assert abs(metric.pass_rate - 2 / 3) < 0.001

    def test_flakiness_score_peaks_at_fifty_percent_pass_rate(self) -> None:
        """Flakiness score should be 1.0 when pass rate is exactly 50%."""
        # 1 pass, 1 fail → 50% pass rate → maximum uncertainty
        results = self._make_results([True, False])
        metric = FlakinessMetric.compute("tests/test_foo.py::test_baz", results)

        assert metric.flakiness_score == 1.0, "Score should be 1.0 at 50% pass rate"

    @pytest.mark.parametrize("passed,expected_label", [
        ([True, True, True, True],        FlakinessLabel.STABLE),
        ([False, False, False, False],     FlakinessLabel.FAILING),
        ([True, False, True, False],       FlakinessLabel.FLAKY),
        ([True, True, True, False],        FlakinessLabel.FLAKY),
    ])
    def test_label_classification(
        self,
        passed: list[bool],
        expected_label: FlakinessLabel,
    ) -> None:
        """Verify label assignment across various pass/fail patterns."""
        results = self._make_results(passed)
        metric = FlakinessMetric.compute("tests/test_foo.py::test_param", results)
        assert metric.label == expected_label

    def test_to_dict_includes_all_fields(self) -> None:
        """to_dict() should include all required keys for JSON serialisation."""
        results = self._make_results([True, False, True])
        metric = FlakinessMetric.compute("tests/test_foo.py::test_bar", results)
        d = metric.to_dict()

        required_keys = {
            "test_id", "test_name", "total_runs", "passed_runs",
            "failed_runs", "pass_rate", "label", "flakiness_score", "runs",
        }
        assert required_keys.issubset(d.keys()), f"Missing keys: {required_keys - d.keys()}"
        assert isinstance(d["label"], str), "label should be serialised as string"
