"""
web_agent — AI-powered website crawl, test generation, execution, and reporting.

Full pipeline example::

    from validrix.web_agent import run_pipeline
    from pathlib import Path

    result = run_pipeline(
        url="https://example.com",
        prompt="Test the contact form and navigation links",
        report_dir=Path("validrix_reports"),
    )
    print(f"Report: {result.report_path}")
    print(f"{result.passed}/{result.total_tests} tests passed")
"""

from __future__ import annotations

from pathlib import Path

from validrix.web_agent.crawler import WebCrawler
from validrix.web_agent.executor import TestExecutor
from validrix.web_agent.models import TestSuiteResult
from validrix.web_agent.reporter import WebReporter
from validrix.web_agent.test_generator import WebTestGenerator

__all__ = [
    "WebCrawler",
    "WebTestGenerator",
    "TestExecutor",
    "WebReporter",
    "run_pipeline",
]


def run_pipeline(
    url: str,
    prompt: str,
    report_dir: Path | None = None,
    max_tests: int = 10,
    timeout_seconds: int = 60,
    headless: bool = True,
) -> TestSuiteResult:
    """
    Run the complete web_agent pipeline synchronously.

    Convenience wrapper for use in scripts and the CLI. For async/API use,
    call the individual components or use the FastAPI server instead.

    Args:
        url:              The live website URL to crawl and test.
        prompt:           Plain-English description of what to test.
        report_dir:       Directory where report.html will be written.
        max_tests:        Maximum number of test functions to generate.
        timeout_seconds:  Per-test Playwright timeout.
        headless:         Run Chromium without a visible window.

    Returns:
        TestSuiteResult with ai_summary and report_path populated.
    """
    report_dir = report_dir or Path("validrix_reports")

    crawl = WebCrawler(timeout_ms=timeout_seconds * 1000, headless=headless).crawl(url)
    suite = WebTestGenerator().generate(crawl=crawl, prompt=prompt, max_tests=max_tests)
    result = TestExecutor(timeout_seconds=timeout_seconds, headless=headless).run(
        suite, report_dir=report_dir
    )
    result.crawl_result = crawl
    WebReporter().generate(result, report_dir=report_dir)
    return result
