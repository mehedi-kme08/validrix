"""
flaky_detector.py — Flaky test detector with configurable multi-run analysis.

Design decision: Re-run at the protocol level, not via pytest-rerunfailures.
  WHY: pytest-rerunfailures only reruns failing tests, which misses the
       critical "intermittently failing" case — a test that passes most of the
       time but fails occasionally.  We run EVERY marked test N times and
       compute a pass-rate, catching tests that are 90 % reliable but still
       wrong.

  Alternatives considered:
    - pytest-rerunfailures: reruns only on failure; no flakiness metric.
    - External tooling (Flaky, flake8-bugbear): static analysis only;
      cannot detect runtime race conditions.
    - Quarantine via custom marker: helps manage known flaky tests but does
      not help discover unknown ones.

  Tradeoffs:
    - Running N times multiplies test execution time by N.  We mitigate this
      by making flaky detection opt-in via --detect-flaky flag and by running
      only tests marked @pytest.mark.flaky_check by default.
    - Threshold-based classification (pass_rate < 0.5 = FLAKY) is a heuristic;
      real flakiness analysis needs historical data.  The JSON report is
      designed to feed into a time-series store for trend analysis.

  Output formats:
    - flaky_report.json — machine-readable, suitable for CI dashboards
    - flaky_report.html — human-readable, self-contained HTML (no server needed)
"""

from __future__ import annotations

import json
import logging
import textwrap
from collections import defaultdict
from collections.abc import Generator
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

import pytest

from validrix.core.config_manager import ConfigManager, FlakyConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class FlakinessLabel(StrEnum):
    """Classification of a test's reliability based on its pass rate."""
    STABLE   = "STABLE"    # Always passes
    FLAKY    = "FLAKY"     # Mixed results
    FAILING  = "FAILING"   # Never passes


@dataclass
class RunResult:
    """Outcome of a single test execution."""

    run_number: int
    passed: bool
    duration_seconds: float
    error_message: str = ""


@dataclass
class FlakinessMetric:
    """Aggregated flakiness data for one test across all runs."""

    test_id: str
    test_name: str
    total_runs: int
    passed_runs: int
    failed_runs: int
    pass_rate: float
    label: FlakinessLabel
    flakiness_score: float  # 0.0 = stable, 1.0 = maximally flaky
    runs: list[RunResult] = field(default_factory=list)

    @classmethod
    def compute(cls, test_id: str, results: list[RunResult]) -> FlakinessMetric:
        """Compute flakiness metrics from a list of run results."""
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed
        pass_rate = passed / total if total > 0 else 0.0

        # Flakiness score peaks at 0.5 pass_rate (maximum uncertainty)
        # Using 1 - |2p - 1| which maps [0,1] → [0,1] with peak at 0.5
        flakiness_score = 1.0 - abs(2 * pass_rate - 1.0)

        if pass_rate == 1.0:
            label = FlakinessLabel.STABLE
        elif pass_rate == 0.0:
            label = FlakinessLabel.FAILING
        else:
            label = FlakinessLabel.FLAKY

        return cls(
            test_id=test_id,
            test_name=test_id.split("::")[-1],
            total_runs=total,
            passed_runs=passed,
            failed_runs=failed,
            pass_rate=pass_rate,
            label=label,
            flakiness_score=round(flakiness_score, 4),
            runs=results,
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)  # type: ignore[arg-type]
        d["label"] = self.label.value
        return d


@dataclass
class FlakyReport:
    """Session-level flakiness report."""

    generated_at: str
    total_tests_checked: int
    total_flaky: int
    total_failing: int
    total_stable: int
    threshold: float
    results: list[FlakinessMetric]

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "total_tests_checked": self.total_tests_checked,
            "total_flaky": self.total_flaky,
            "total_failing": self.total_failing,
            "total_stable": self.total_stable,
            "threshold": self.threshold,
            "results": [r.to_dict() for r in self.results],
        }


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------

class FlakyDetectorPlugin:
    """
    Pytest plugin that runs tests multiple times and detects flakiness.

    Activated via --detect-flaky flag or by marking tests with
    @pytest.mark.flaky_check.

    Hooks implemented:
    - pytest_addoption               → add --detect-flaky / --flaky-runs flags
    - pytest_configure               → register flaky_check marker
    - pytest_runtest_protocol        → intercept and re-run tests N times
    - pytest_sessionfinish           → write JSON and HTML reports
    """

    def __init__(self, config: pytest.Config | None = None) -> None:
        self._pytest_config = config
        self._cfg = ConfigManager.load()
        self._flaky_cfg: FlakyConfig = self._cfg.flaky
        self._run_counts: dict[str, list[RunResult]] = defaultdict(list)
        self._detect_all: bool = False

    # ------------------------------------------------------------------
    # pytest hooks
    # ------------------------------------------------------------------

    def pytest_addoption(self, parser: pytest.Parser) -> None:
        group = parser.getgroup("validrix-flaky", "Validrix flaky test detection")
        group.addoption(
            "--detect-flaky",
            action="store_true",
            default=False,
            help="Run all tests N times to detect flakiness.",
        )
        group.addoption(
            "--flaky-runs",
            type=int,
            default=None,
            help=f"Number of runs per test (default: {self._flaky_cfg.runs}).",
        )

    def pytest_configure(self, config: pytest.Config) -> None:
        config.addinivalue_line(
            "markers",
            "flaky_check: mark test for flakiness detection (run N times).",
        )
        self._detect_all = config.getoption("--detect-flaky", default=False)
        runs_override = config.getoption("--flaky-runs", default=None)
        if runs_override:
            self._flaky_cfg.runs = int(runs_override)

    def pytest_runtest_protocol(
        self,
        item: pytest.Item,
        nextitem: pytest.Item | None,
    ) -> bool | None:
        """
        Intercept test execution for items that need flakiness detection.

        Returns True to signal that we've handled the protocol ourselves.
        Returns None to let pytest handle it normally.
        """
        should_detect = self._detect_all or item.get_closest_marker("flaky_check")
        if not should_detect or not self._flaky_cfg.enabled:
            return None  # Let pytest handle normally

        results = self._run_n_times(item, self._flaky_cfg.runs)
        self._run_counts[item.nodeid] = results

        flakiness = FlakinessMetric.compute(item.nodeid, results)

        if flakiness.label == FlakinessLabel.FLAKY:
            logger.warning(
                "FLAKY: %s — pass rate %.0f%% over %d runs (score: %.2f)",
                item.nodeid,
                flakiness.pass_rate * 100,
                flakiness.total_runs,
                flakiness.flakiness_score,
            )
        else:
            logger.info(
                "Flaky check: %s → %s (%.0f%% pass rate)",
                item.nodeid,
                flakiness.label.value,
                flakiness.pass_rate * 100,
            )

        # Signal to pytest that we managed this item's protocol
        return True

    def _run_n_times(self, item: pytest.Item, n: int) -> list[RunResult]:
        """Execute ``item`` exactly ``n`` times and collect results."""
        results: list[RunResult] = []
        for run_number in range(1, n + 1):
            reports: list[pytest.TestReport] = []

            # Run the test using pytest's internal runner
            for report in self._execute_item(item):
                reports.append(report)

            call_report = next(
                (r for r in reports if r.when == "call"), None
            )

            passed = call_report.passed if call_report else False
            duration = call_report.duration if call_report else 0.0
            error_msg = ""

            if call_report and call_report.failed:
                longrepr = (
                    call_report.longreprtext
                    if hasattr(call_report, "longreprtext")
                    else str(call_report.longrepr)
                )
                error_msg = longrepr[-500:]  # Truncate to avoid huge reports

            results.append(RunResult(
                run_number=run_number,
                passed=passed,
                duration_seconds=duration,
                error_message=error_msg,
            ))
            logger.debug("Run %d/%d for %s: %s", run_number, n, item.nodeid,
                         "PASS" if passed else "FAIL")

        return results

    @staticmethod
    def _execute_item(item: pytest.Item) -> Generator[pytest.TestReport, None, None]:
        """Run setup/call/teardown for ``item`` and yield each phase report."""
        reports = pytest.runner.runtestprotocol(item, log=False)
        yield from reports

    def pytest_sessionfinish(
        self,
        session: pytest.Session,
        exitstatus: int | pytest.ExitCode,
    ) -> None:
        """Write flakiness reports after the session ends."""
        if not self._run_counts:
            return

        results = [
            FlakinessMetric.compute(test_id, runs)
            for test_id, runs in self._run_counts.items()
        ]

        report = FlakyReport(
            generated_at=datetime.now(UTC).isoformat(),
            total_tests_checked=len(results),
            total_flaky=sum(1 for r in results if r.label == FlakinessLabel.FLAKY),
            total_failing=sum(1 for r in results if r.label == FlakinessLabel.FAILING),
            total_stable=sum(1 for r in results if r.label == FlakinessLabel.STABLE),
            threshold=self._flaky_cfg.threshold,
            results=results,
        )

        self._write_json_report(report)
        self._write_html_report(report)

    def _write_json_report(self, report: FlakyReport) -> None:
        path = self._flaky_cfg.report_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(report.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )
        logger.info("FlakyDetector: JSON report → %s", path)

    def _write_html_report(self, report: FlakyReport) -> None:
        html_path = self._flaky_cfg.report_path.with_suffix(".html")

        rows = ""
        for r in sorted(report.results, key=lambda x: x.flakiness_score, reverse=True):
            label_color = {
                "FLAKY": "#f59e0b",
                "FAILING": "#ef4444",
                "STABLE": "#10b981",
            }.get(r.label.value, "#6b7280")

            rows += textwrap.dedent(f"""
                <tr>
                  <td>{r.test_name}</td>
                  <td><span style="color:{label_color};font-weight:bold">{r.label.value}</span></td>
                  <td>{r.pass_rate:.0%}</td>
                  <td>{r.flakiness_score:.2f}</td>
                  <td>{r.total_runs}</td>
                </tr>
            """)

        html = textwrap.dedent(f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
              <meta charset="UTF-8"/>
              <title>Validrix Flaky Report</title>
              <style>
                body {{ font-family: system-ui, sans-serif; padding: 2rem; background: #f8fafc; }}
                h1   {{ color: #1e293b; }}
                .stats {{ display: flex; gap: 1rem; margin: 1rem 0; }}
                .stat {{ background: white; border-radius: 8px; padding: 1rem 1.5rem;
                         box-shadow: 0 1px 3px rgba(0,0,0,.1); }}
                table {{ width: 100%; border-collapse: collapse; background: white;
                         box-shadow: 0 1px 3px rgba(0,0,0,.1); border-radius: 8px; overflow: hidden; }}
                th, td {{ padding: .75rem 1rem; text-align: left; border-bottom: 1px solid #e2e8f0; }}
                th {{ background: #1e293b; color: white; }}
                tr:hover {{ background: #f1f5f9; }}
              </style>
            </head>
            <body>
              <h1>&#x1F9EA; Validrix Flaky Test Report</h1>
              <p>Generated: {report.generated_at}</p>
              <div class="stats">
                <div class="stat"><strong>{report.total_tests_checked}</strong><br/>Tests checked</div>
                <div class="stat" style="color:#ef4444"><strong>{report.total_flaky}</strong><br/>Flaky</div>
                <div class="stat" style="color:#6b7280"><strong>{report.total_failing}</strong><br/>Failing</div>
                <div class="stat" style="color:#10b981"><strong>{report.total_stable}</strong><br/>Stable</div>
              </div>
              <table>
                <thead>
                  <tr>
                    <th>Test</th><th>Label</th><th>Pass Rate</th>
                    <th>Flakiness Score</th><th>Runs</th>
                  </tr>
                </thead>
                <tbody>{rows}</tbody>
              </table>
            </body>
            </html>
        """).strip()

        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text(html, encoding="utf-8")
        logger.info("FlakyDetector: HTML report → %s", html_path)
