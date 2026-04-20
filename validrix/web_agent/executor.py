"""
executor.py — Runs generated pytest + Playwright test suites and captures results.

Design decision: subprocess pytest invocation over in-process pytest.main().
  WHY: Running pytest.main() in the same process as the FastAPI server is
       dangerous — pytest modifies sys.path, installs plugins globally, and
       can corrupt the parent process's state. A subprocess gives us a clean
       Python environment per run, proper stdout/stderr isolation, and a
       well-defined exit code.

  Alternatives considered:
    - pytest.main() in-process: fast, but global side effects break isolation
      in long-lived server processes. Discovered empirically in early prototypes.
    - xdist parallel execution: powerful but adds complexity; the subprocess
      approach already parallelises at the job level (one job per API request).
    - Celery worker: correct isolation, but a heavy Redis/RabbitMQ dependency
      for a feature that only needs single-node concurrency.

  Tradeoffs:
    - subprocess adds ~0.3-0.5 s overhead per run (Python startup).
      Acceptable because test runs take seconds to minutes.
    - We write generated code to a temp file, run pytest against it, then
      parse the JSON report. The temp file is cleaned up after the run.
    - Screenshots are saved to the report directory; paths are stored in
      TestResult for the HTML report to reference.

  Result parsing:
    pytest's --json-report plugin writes a structured JSON file that we
    parse rather than screen-scraping terminal output. This is stable
    across pytest versions and locale settings.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import cast

from validrix.web_agent.models import GeneratedTestSuite, TestResult, TestSuiteResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_TIMEOUT_SECONDS: int = 60  # per-test timeout


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_pytest_json(json_path: Path, screenshot_dir: Path) -> list[TestResult]:
    """
    Parse pytest-json-report output into TestResult objects.

    The json-report schema: https://github.com/numirias/pytest-json-report
    Top-level key: ``tests`` — list of test node records.
    """
    if not json_path.exists():
        logger.warning("pytest JSON report not found at: %s", json_path)
        return []

    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse pytest JSON report: %s", exc)
        return []

    results: list[TestResult] = []
    for item in data.get("tests", []):
        node_id: str = item.get("nodeid", "unknown")
        outcome: str = item.get("outcome", "error")
        duration: float = item.get("duration", 0.0)

        # Collect error info from the call phase (the actual test body)
        call_obj = item.get("call", {}) or {}
        call_data: dict[str, object] = cast(dict[str, object], call_obj) if isinstance(call_obj, dict) else {}
        longrepr = str(call_data.get("longrepr", "") or "")
        crash_obj = call_data.get("crash", {}) or {}
        crash: dict[str, object] = cast(dict[str, object], crash_obj) if isinstance(crash_obj, dict) else {}

        error_message: str | None = None
        traceback: str | None = None
        if outcome in ("failed", "error") and longrepr:
            traceback = longrepr
            # Use crash.message if available for a cleaner one-liner
            error_message = str(crash.get("message", "")) or longrepr.splitlines()[-1]

        # Check if a screenshot was saved alongside the test node
        safe_name = node_id.replace("/", "_").replace("::", "__").replace(" ", "_")
        screenshot_path: str | None = None
        candidate = screenshot_dir / f"{safe_name}.png"
        if candidate.exists():
            screenshot_path = str(candidate)

        # Map pytest outcomes to our Literal type
        status_map = {"passed": "passed", "failed": "failed", "skipped": "skipped"}
        status = status_map.get(outcome, "error")

        results.append(
            TestResult(
                test_name=node_id,
                status=status,  # type: ignore[arg-type]
                error_message=error_message,
                traceback=traceback,
                screenshot_path=screenshot_path,
                duration=duration,
            )
        )

    return results


def _write_conftest(directory: Path, url: str, timeout_ms: int, screenshot_dir: Path) -> None:
    """
    Write a conftest.py that provides the ``page`` fixture and failure screenshots.

    This is injected alongside the generated test file so the generated tests
    can use ``page: Page`` directly without any extra setup.
    """
    conftest = f"""\
import pytest
from pathlib import Path
from playwright.sync_api import sync_playwright, Page


BASE_URL = {url!r}
SCREENSHOT_DIR = Path({str(screenshot_dir)!r})
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)


@pytest.fixture(scope="session")
def browser_context():
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(viewport={{"width": 1280, "height": 800}})
        yield context
        context.close()
        browser.close()


@pytest.fixture()
def page(browser_context):
    p = browser_context.new_page()
    p.set_default_timeout({timeout_ms})
    yield p
    p.close()


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when == "call" and report.failed:
        page_fixture = item.funcargs.get("page")
        if page_fixture:
            safe = item.nodeid.replace("/", "_").replace("::", "__").replace(" ", "_")
            path = SCREENSHOT_DIR / f"{{safe}}.png"
            try:
                page_fixture.screenshot(path=str(path), full_page=True)
            except Exception:
                pass
"""
    (directory / "conftest.py").write_text(conftest, encoding="utf-8")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class TestExecutor:
    """
    Writes generated tests to a temp directory, executes them via pytest,
    and returns a structured TestSuiteResult.

    Example::

        executor = TestExecutor(timeout_seconds=60, headless=True)
        result = executor.run(suite, report_dir=Path("validrix_reports"))
    """

    def __init__(
        self,
        timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
        headless: bool = True,
    ) -> None:
        """
        Args:
            timeout_seconds: Per-test execution timeout passed to Playwright.
            headless:        Run Chromium in headless mode (no visible window).
        """
        self._timeout_seconds = timeout_seconds
        self._headless = headless

    def run(
        self,
        suite: GeneratedTestSuite,
        report_dir: Path | None = None,
    ) -> TestSuiteResult:
        """
        Execute the generated test suite and return results.

        Args:
            suite:       The GeneratedTestSuite produced by WebTestGenerator.
            report_dir:  Directory for screenshots and JSON report artefacts.
                         Defaults to a temp directory that persists for the
                         lifetime of this call.

        Returns:
            TestSuiteResult — always returned, never raises.
        """
        if not suite.succeeded:
            return TestSuiteResult(
                url=suite.url,
                prompt=suite.prompt,
                ai_summary=f"Test generation failed — cannot execute: {suite.error}",
                generated_code=suite.combined_code,
            )

        report_dir = report_dir or Path(tempfile.mkdtemp(prefix="validrix_"))
        report_dir.mkdir(parents=True, exist_ok=True)
        screenshot_dir = report_dir / "screenshots"
        screenshot_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Executing %d test(s) for %s", len(suite.tests), suite.url)
        start = time.monotonic()

        with tempfile.TemporaryDirectory(prefix="validrix_exec_") as tmpdir:
            tmp = Path(tmpdir)
            test_file = tmp / "test_generated.py"
            json_report = tmp / "report.json"

            test_file.write_text(suite.combined_code, encoding="utf-8")
            _write_conftest(tmp, suite.url, self._timeout_seconds * 1000, screenshot_dir)

            cmd = self._build_pytest_command(test_file, json_report)
            exit_code, stdout, stderr = self._run_subprocess(cmd, tmpdir)

            tests = _parse_pytest_json(json_report, screenshot_dir)

            # If json-report is absent (plugin not installed), fall back to
            # inferring pass/fail from the exit code alone.
            if not tests:
                tests = self._infer_from_exit_code(exit_code, suite, stdout + stderr)

        duration = time.monotonic() - start
        passed = sum(1 for t in tests if t.status == "passed")
        failed = sum(1 for t in tests if t.status in ("failed", "error"))
        skipped = sum(1 for t in tests if t.status == "skipped")

        logger.info(
            "Execution complete in %.1fs: %d passed, %d failed, %d skipped",
            duration,
            passed,
            failed,
            skipped,
        )

        return TestSuiteResult(
            url=suite.url,
            prompt=suite.prompt,
            total_tests=len(tests),
            passed=passed,
            failed=failed,
            skipped=skipped,
            duration=duration,
            tests=tests,
            generated_code=suite.combined_code,
        )

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _build_pytest_command(test_file: Path, json_report: Path) -> list[str]:
        """Build the pytest invocation, requesting JSON output."""
        return [
            sys.executable,
            "-m",
            "pytest",
            str(test_file),
            "--tb=short",
            "-q",
            "--no-header",
            # pytest-json-report is a soft dependency — we handle its absence
            f"--json-report-file={json_report}",
            "--json-report",
        ]

    @staticmethod
    def _run_subprocess(cmd: list[str], cwd: str) -> tuple[int, str, str]:
        """Execute pytest in a subprocess and capture output."""
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=cwd,
                env=env,
                timeout=300,  # hard cap for the whole subprocess (5 min)
            )
            return proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired:
            logger.error("pytest subprocess timed out after 300s")
            return 1, "", "pytest subprocess timed out"
        except Exception as exc:
            logger.error("Failed to launch pytest subprocess: %s", exc)
            return 1, "", str(exc)

    @staticmethod
    def _infer_from_exit_code(
        exit_code: int,
        suite: GeneratedTestSuite,
        output: str,
    ) -> list[TestResult]:
        """
        Fallback when pytest-json-report is not installed.

        Creates one synthetic TestResult per parsed test function name,
        all marked as the same outcome (all-pass or all-fail).
        """
        status = "passed" if exit_code == 0 else "failed"
        error_message = output if exit_code != 0 else None
        return [
            TestResult(
                test_name=t.name,
                status=status,  # type: ignore[arg-type]
                error_message=error_message,
                duration=0.0,
            )
            for t in suite.tests
        ] or [
            TestResult(
                test_name="test_suite",
                status=status,  # type: ignore[arg-type]
                error_message=error_message,
                duration=0.0,
            )
        ]
