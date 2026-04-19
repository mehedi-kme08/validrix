"""
ai_reporter.py — AI-powered failure summariser and root-cause analyser.

Design decision: Post-session hook, not inline failure handling.
  WHY: Collecting all failures before calling the AI lets us send a single
       batched request instead of one per failure, dramatically reducing
       API cost and latency. It also gives the model the full picture —
       pattern-matching across 10 failures is far more insightful than
       analysing each in isolation.

  Alternatives considered:
    - Per-failure AI call in pytest_runtest_logreport: too many API round-
      trips, high cost, no cross-failure pattern analysis.
    - Post-processing a JUnit XML file: decoupled from the live session,
      which means you need a separate invocation step.

  Tradeoffs:
    - If the session is killed mid-run, no report is generated.
      Mitigation: we write a JSON checkpoint after each failure so the
      CLI's `validrix report` command can resume from it.
    - Report generation adds ~5-15 s at session end. We make this opt-in
      (requires VALIDRIX_AI_ANTHROPIC_API_KEY to be set).

  Output formats:
    - report.md  — human-readable Markdown for GitHub PR comments / Slack
    - report.json — machine-readable for downstream tooling
"""

from __future__ import annotations

import json
import logging
import textwrap
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import anthropic
import openai
import pytest

from validrix.core.config_manager import AIConfig, ConfigManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Report prompt
# ---------------------------------------------------------------------------

_ANALYSIS_SYSTEM_PROMPT: Final[str] = textwrap.dedent("""
    You are a senior software engineer and QA lead reviewing automated test
    failures. Your audience is a developer who needs to quickly understand
    what broke and why.

    Given a batch of test failure reports (test name, error type, traceback,
    and test code snippet), produce a structured root-cause analysis.

    Output format (Markdown):
    ## Executive Summary
    <2-3 sentence overview of what failed and common patterns>

    ## Failure Analysis

    ### <Test Name>
    - **Root Cause**: <concise diagnosis>
    - **Likely Fix**: <specific, actionable suggestion>
    - **Severity**: Critical | High | Medium | Low

    ## Common Patterns
    <If multiple tests share the same root cause, call it out here>

    ## Recommended Next Steps
    <Ordered list of actions>

    Be specific. Reference actual variable names, URLs, and error messages
    from the tracebacks. Avoid generic advice.
""").strip()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class FailureRecord:
    """Captures everything we know about a single test failure."""

    test_id: str
    test_name: str
    error_type: str
    error_message: str
    traceback: str
    duration_seconds: float
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


@dataclass
class AIReport:
    """The complete post-session AI analysis."""

    session_id: str
    total_failures: int
    generated_at: str
    summary_markdown: str
    failures: list[FailureRecord]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Reporter
# ---------------------------------------------------------------------------


class AIReporterPlugin:
    """
    Pytest plugin that analyses test failures with Claude/OpenAI after the run.

    Registered automatically via pytest11 entry point when validrix is installed.
    Skips silently if no AI API key is configured (opt-in behaviour).

    Hooks implemented:
    - pytest_runtest_logreport  → collect failures during the session
    - pytest_sessionfinish      → send failures to AI and write reports
    """

    def __init__(self, config: pytest.Config | None = None) -> None:
        self._pytest_config = config
        self._failures: list[FailureRecord] = []
        self._cfg = ConfigManager.load()
        self._report_dir = self._cfg.report_dir
        self._ai_config: AIConfig = self._cfg.ai

    # ------------------------------------------------------------------
    # pytest hooks
    # ------------------------------------------------------------------

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        """Capture each failing test's traceback and metadata."""
        if report.when != "call" or not report.failed:
            return

        longrepr = report.longreprtext if hasattr(report, "longreprtext") else str(report.longrepr)
        error_lines = longrepr.splitlines()
        error_type = "UnknownError"
        error_message = longrepr

        # Extract error type from the last line of the traceback
        for line in reversed(error_lines):
            if "Error:" in line or "Exception:" in line or "Failure:" in line:
                parts = line.split(":", 1)
                error_type = parts[0].strip()
                error_message = parts[1].strip() if len(parts) > 1 else line
                break

        self._failures.append(
            FailureRecord(
                test_id=report.nodeid,
                test_name=report.nodeid.split("::")[-1],
                error_type=error_type,
                error_message=error_message,
                traceback=longrepr,
                duration_seconds=report.duration,
            )
        )

        # Write a checkpoint so `validrix report` can resume if session dies
        self._write_checkpoint()

    def pytest_sessionfinish(
        self,
        session: pytest.Session,
        exitstatus: int | pytest.ExitCode,
    ) -> None:
        """Generate the AI report after all tests have run."""
        if not self._failures:
            logger.info("AIReporter: no failures to analyse.")
            return

        api_key = self._ai_config.anthropic_api_key or self._ai_config.openai_api_key
        if not api_key:
            logger.warning(
                "AIReporter: no API key configured. Skipping AI analysis. Set VALIDRIX_AI_ANTHROPIC_API_KEY to enable."
            )
            return

        logger.info(
            "AIReporter: analysing %d failure(s) with %s…",
            len(self._failures),
            self._ai_config.provider,
        )

        try:
            summary = self._analyse_failures(self._failures)
        except Exception as exc:
            logger.error("AIReporter: AI analysis failed — %s", exc)
            return

        report = AIReport(
            session_id=datetime.now(UTC).strftime("%Y%m%dT%H%M%S"),
            total_failures=len(self._failures),
            generated_at=datetime.now(UTC).isoformat(),
            summary_markdown=summary,
            failures=self._failures,
        )

        self._write_reports(report)

    # ------------------------------------------------------------------
    # AI interaction
    # ------------------------------------------------------------------

    def _analyse_failures(self, failures: list[FailureRecord]) -> str:
        """Send failures to the configured AI provider and return Markdown."""
        user_content = self._format_failures_for_prompt(failures)

        if self._ai_config.provider == "anthropic":
            return self._call_anthropic(user_content)
        return self._call_openai(user_content)

    def _call_anthropic(self, user_content: str) -> str:
        client = anthropic.Anthropic(
            api_key=self._ai_config.anthropic_api_key,
            timeout=self._ai_config.timeout_seconds,
        )
        response = client.messages.create(
            model=self._ai_config.model,
            max_tokens=self._ai_config.max_tokens,
            system=_ANALYSIS_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
        return response.content[0].text  # type: ignore[union-attr]

    def _call_openai(self, user_content: str) -> str:
        client = openai.OpenAI(
            api_key=self._ai_config.openai_api_key,
            timeout=self._ai_config.timeout_seconds,
        )
        response = client.chat.completions.create(
            model=self._ai_config.model,
            max_tokens=self._ai_config.max_tokens,
            messages=[
                {"role": "system", "content": _ANALYSIS_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        return response.choices[0].message.content or ""

    @staticmethod
    def _format_failures_for_prompt(failures: list[FailureRecord]) -> str:
        """Serialize failures into a prompt-friendly text block."""
        sections: list[str] = [
            f"Total failures: {len(failures)}\n",
        ]
        for i, f in enumerate(failures, start=1):
            sections.append(
                f"--- Failure {i} ---\n"
                f"Test: {f.test_id}\n"
                f"Error type: {f.error_type}\n"
                f"Error message: {f.error_message}\n"
                f"Duration: {f.duration_seconds:.2f}s\n"
                f"Traceback:\n{f.traceback}\n"
            )
        return "\n".join(sections)

    # ------------------------------------------------------------------
    # Report writing
    # ------------------------------------------------------------------

    def _write_reports(self, report: AIReport) -> None:
        self._report_dir.mkdir(parents=True, exist_ok=True)

        md_path = self._report_dir / "report.md"
        md_path.write_text(report.summary_markdown, encoding="utf-8")
        logger.info("AIReporter: Markdown report → %s", md_path)

        json_path = self._report_dir / "report.json"
        json_path.write_text(
            json.dumps(report.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )
        logger.info("AIReporter: JSON report → %s", json_path)

    def _write_checkpoint(self) -> None:
        """Persist failure list so the CLI can resume if the session dies."""
        self._report_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = self._report_dir / "_failures_checkpoint.json"
        checkpoint.write_text(
            json.dumps(
                [asdict(f) for f in self._failures],
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Standalone entry point for `validrix report`
    # ------------------------------------------------------------------

    def generate_from_checkpoint(self) -> Path | None:
        """
        Read the failure checkpoint and generate a report without a live session.

        Used by ``validrix report`` when called after a test run.

        Returns:
            Path to the generated Markdown report, or None if no checkpoint exists.
        """
        checkpoint = self._report_dir / "_failures_checkpoint.json"
        if not checkpoint.exists():
            logger.warning("AIReporter: no failure checkpoint found at %s", checkpoint)
            return None

        raw = json.loads(checkpoint.read_text(encoding="utf-8"))
        self._failures = [FailureRecord(**f) for f in raw]

        api_key = self._ai_config.anthropic_api_key or self._ai_config.openai_api_key
        if not api_key:
            logger.error("AIReporter: API key required. Set VALIDRIX_AI_ANTHROPIC_API_KEY.")
            return None

        summary = self._analyse_failures(self._failures)
        report = AIReport(
            session_id=datetime.now(UTC).strftime("%Y%m%dT%H%M%S"),
            total_failures=len(self._failures),
            generated_at=datetime.now(UTC).isoformat(),
            summary_markdown=summary,
            failures=self._failures,
        )
        self._write_reports(report)
        return self._report_dir / "report.md"
