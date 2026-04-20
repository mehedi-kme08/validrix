"""
test_web_agent_e2e.py — End-to-end integration tests for the Web Agent pipeline.

Covers the full flow against a live website:
  Stage 1  WebCrawler       crawls https://mhdhasan.netlify.app/
  Stage 2  WebTestGenerator AI generates pytest + Playwright tests (requires API key)
  Stage 3  TestExecutor     runs generated tests in headless Chromium
  Stage 4  WebReporter      produces AI summary + HTML report

Run all stages (requires VALIDRIX_AI__ANTHROPIC_API_KEY):
    pytest tests/integration/test_web_agent_e2e.py -v -s

Run crawl-only (no API key needed):
    pytest tests/integration/test_web_agent_e2e.py -v -s -m "not requires_api_key"

Design decision: each pipeline stage is its own test function.
  WHY: Granular tests pinpoint exactly which stage broke. A single monolithic
  test would report "e2e failed" with no indication of whether the crawler,
  AI, or executor was at fault. Stage isolation also lets CI run the crawler
  test on every PR and gate the AI stages behind a secrets check.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from validrix.web_agent.crawler import WebCrawler
from validrix.web_agent.models import (
    CrawlResult,
    GeneratedTestSuite,
    TestSuiteResult,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_URL = "https://mhdhasan.netlify.app/"
TEST_PROMPT = (
    "Verify the page title loads correctly. "
    "Check all main navigation links (About, Skills, Experience, "
    "Achievements, Education, Contact) are present and visible. "
    "Confirm the hero heading contains text. "
    "Test that the contact section is reachable on the page."
)
MAX_TESTS = 8
TIMEOUT_SEC = 45
REPORT_DIR = Path("validrix_reports") / "e2e_run"

# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.integration

requires_api_key = pytest.mark.skipif(
    not os.environ.get("VALIDRIX_AI__ANTHROPIC_API_KEY"),
    reason="VALIDRIX_AI__ANTHROPIC_API_KEY not set — skipping AI stages",
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def report_dir() -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    return REPORT_DIR


@pytest.fixture(scope="module")
def crawl_result(report_dir: Path) -> CrawlResult:
    """Run the crawler once and share the result across all stage tests."""
    crawler = WebCrawler(timeout_ms=TIMEOUT_SEC * 1000, headless=True)
    return crawler.crawl(TARGET_URL)


@pytest.fixture(scope="module")
def generated_suite(crawl_result: CrawlResult, report_dir: Path) -> GeneratedTestSuite:
    """Generate tests from the crawl result (requires API key)."""
    from validrix.web_agent.test_generator import WebTestGenerator

    generator = WebTestGenerator()
    return generator.generate(
        crawl=crawl_result,
        prompt=TEST_PROMPT,
        max_tests=MAX_TESTS,
        output_path=report_dir / "test_generated.py",
    )


@pytest.fixture(scope="module")
def suite_result(generated_suite: GeneratedTestSuite, crawl_result: CrawlResult, report_dir: Path) -> TestSuiteResult:
    """Execute the generated test suite."""
    from validrix.web_agent.executor import TestExecutor

    executor = TestExecutor(timeout_seconds=TIMEOUT_SEC, headless=True)
    result = executor.run(generated_suite, report_dir=report_dir)
    result.crawl_result = crawl_result
    return result


# ---------------------------------------------------------------------------
# Stage 1 — Crawler (no API key required)
# ---------------------------------------------------------------------------


class TestCrawler:
    """Stage 1: Playwright-based DOM extraction."""

    def test_crawl_succeeds(self, crawl_result: CrawlResult) -> None:
        """Crawler must complete without error for the target URL."""
        assert crawl_result.succeeded, f"Crawl failed: {crawl_result.error}"

    def test_page_title_present(self, crawl_result: CrawlResult) -> None:
        """Page title must be a non-empty string."""
        assert crawl_result.title, "Page title was empty"
        assert "Mehedi" in crawl_result.title or "SDET" in crawl_result.title, (
            f"Unexpected title: {crawl_result.title!r}"
        )

    def test_headings_found(self, crawl_result: CrawlResult) -> None:
        """At least one h1-h3 heading must exist on the page."""
        assert len(crawl_result.headings) > 0, "No headings found on page"

    def test_navigation_links_present(self, crawl_result: CrawlResult) -> None:
        """Portfolio nav sections must be discoverable as links."""
        all_link_text = " ".join(lk.text.lower() for lk in crawl_result.links)
        for section in ("about", "skills", "experience", "contact"):
            assert section in all_link_text, (
                f"Navigation link for '{section}' not found. Links found: {[lk.text for lk in crawl_result.links]}"
            )

    def test_internal_links_dominate(self, crawl_result: CrawlResult) -> None:
        """Most links on a single-page portfolio should be internal anchors."""
        internal = [lk for lk in crawl_result.links if not lk.is_external]
        assert len(internal) >= 3, f"Expected >= 3 internal links, got {len(internal)}"

    def test_crawl_duration_reasonable(self, crawl_result: CrawlResult) -> None:
        """Crawl must complete within the configured timeout."""
        assert crawl_result.crawl_duration < TIMEOUT_SEC, (
            f"Crawl took {crawl_result.crawl_duration:.1f}s, limit is {TIMEOUT_SEC}s"
        )

    def test_visible_text_extracted(self, crawl_result: CrawlResult) -> None:
        """Visible text sample must be non-empty (proves JS rendered correctly)."""
        assert len(crawl_result.visible_text_sample) > 100, "Visible text sample too short — page may not have rendered"


# ---------------------------------------------------------------------------
# Stage 2 — AI Test Generator (requires API key)
# ---------------------------------------------------------------------------


@requires_api_key
class TestGenerator:
    """Stage 2: Claude API generates pytest + Playwright test code."""

    def test_generation_succeeds(self, generated_suite: GeneratedTestSuite) -> None:
        """Generation must complete without error."""
        assert generated_suite.succeeded, f"Generation failed: {generated_suite.error}"

    def test_tests_were_produced(self, generated_suite: GeneratedTestSuite) -> None:
        """At least one test function must be returned."""
        assert len(generated_suite.tests) > 0, "No tests were generated"

    def test_respects_max_tests_limit(self, generated_suite: GeneratedTestSuite) -> None:
        """Generator must not exceed the MAX_TESTS cap."""
        assert len(generated_suite.tests) <= MAX_TESTS, (
            f"Generated {len(generated_suite.tests)} tests, cap is {MAX_TESTS}"
        )

    def test_combined_code_is_valid_python(self, generated_suite: GeneratedTestSuite) -> None:
        """The combined code file must be syntactically valid Python."""
        import ast

        try:
            ast.parse(generated_suite.combined_code)
        except SyntaxError as exc:
            pytest.fail(f"Generated code is not valid Python: {exc}")

    def test_combined_code_contains_test_functions(self, generated_suite: GeneratedTestSuite) -> None:
        """Code must contain at least one pytest-compatible test function."""
        assert "def test_" in generated_suite.combined_code, "Generated code has no test_ functions"

    def test_combined_code_navigates_to_url(self, generated_suite: GeneratedTestSuite) -> None:
        """Every test suite must navigate to the target URL."""
        assert "page.goto" in generated_suite.combined_code, (
            "Generated code never calls page.goto() — tests won't reach the site"
        )

    def test_generated_file_saved_to_disk(self, report_dir: Path) -> None:
        """Generator must persist the combined code to the report directory."""
        generated_file = report_dir / "test_generated.py"
        assert generated_file.exists(), f"Generated file not found at {generated_file}"
        assert generated_file.stat().st_size > 0, "Generated file is empty"

    def test_each_test_has_name_and_description(self, generated_suite: GeneratedTestSuite) -> None:
        """Every GeneratedTest model must have a name and description."""
        for t in generated_suite.tests:
            assert t.name.startswith("test_"), f"Test name {t.name!r} does not start with 'test_'"
            assert t.description, f"Test {t.name!r} has no description"

    def test_generation_duration_reasonable(self, generated_suite: GeneratedTestSuite) -> None:
        """AI call must complete within 120 seconds."""
        assert generated_suite.generation_duration < 120, (
            f"Generation took {generated_suite.generation_duration:.1f}s — possible timeout"
        )


# ---------------------------------------------------------------------------
# Stage 3 — Executor (requires API key, runs generated tests)
# ---------------------------------------------------------------------------


@requires_api_key
class TestExecutor:
    """Stage 3: pytest subprocess runs the generated tests in headless Chromium."""

    def test_execution_produces_results(self, suite_result: TestSuiteResult) -> None:
        """Executor must return at least one test result."""
        assert suite_result.total_tests > 0, "Executor ran 0 tests"

    def test_result_counts_are_consistent(self, suite_result: TestSuiteResult) -> None:
        """passed + failed + skipped must equal total_tests."""
        total = suite_result.passed + suite_result.failed + suite_result.skipped
        assert total == suite_result.total_tests, (
            f"Count mismatch: {suite_result.passed}+{suite_result.failed}+"
            f"{suite_result.skipped} != {suite_result.total_tests}"
        )

    def test_each_result_has_duration(self, suite_result: TestSuiteResult) -> None:
        """Every TestResult must record a duration (even if 0.0)."""
        for t in suite_result.tests:
            assert t.duration >= 0, f"Negative duration on {t.test_name}"

    def test_failed_tests_have_error_message(self, suite_result: TestSuiteResult) -> None:
        """Every failed test must carry an error message for debugging."""
        for t in suite_result.tests:
            if t.status in ("failed", "error"):
                assert t.error_message, f"Test {t.test_name!r} failed but has no error message"

    def test_pass_rate_is_valid_fraction(self, suite_result: TestSuiteResult) -> None:
        """Pass rate must be between 0.0 and 1.0 inclusive."""
        assert 0.0 <= suite_result.pass_rate <= 1.0, f"Invalid pass rate: {suite_result.pass_rate}"

    def test_majority_of_tests_pass(self, suite_result: TestSuiteResult) -> None:
        """At least 50% of generated tests should pass against the live site.

        A lower threshold signals the generator produced bad selectors or the
        site's structure differs significantly from what was crawled.
        """
        assert suite_result.pass_rate >= 0.5, (
            f"Only {suite_result.pass_rate:.0%} of tests passed "
            f"({suite_result.passed}/{suite_result.total_tests}). "
            "Review the generated selectors in validrix_reports/e2e_run/test_generated.py"
        )


# ---------------------------------------------------------------------------
# Stage 4 — Reporter (requires API key)
# ---------------------------------------------------------------------------


@requires_api_key
class TestReporter:
    """Stage 4: AI executive summary + HTML/JSON report generation."""

    @pytest.fixture(scope="class")
    def report_path(self, suite_result: TestSuiteResult, report_dir: Path) -> Path:
        from validrix.web_agent.reporter import WebReporter

        reporter = WebReporter()
        return reporter.generate(suite_result, report_dir=report_dir)

    def test_html_report_created(self, report_path: Path) -> None:
        """report.html must exist after reporter runs."""
        assert report_path.exists(), f"HTML report not found at {report_path}"

    def test_html_report_non_empty(self, report_path: Path) -> None:
        """report.html must have substantial content."""
        size = report_path.stat().st_size
        assert size > 1000, f"HTML report suspiciously small: {size} bytes"

    def test_html_contains_target_url(self, report_path: Path, suite_result: TestSuiteResult) -> None:
        """HTML must reference the URL that was tested."""
        html = report_path.read_text(encoding="utf-8")
        assert suite_result.url in html, "Target URL missing from HTML report"

    def test_html_contains_report_title(self, report_path: Path) -> None:
        """HTML must include the Validrix report title string."""
        html = report_path.read_text(encoding="utf-8")
        assert "Validrix Web Agent Report" in html

    def test_json_report_created(self, report_path: Path, report_dir: Path) -> None:
        """report.json must exist alongside report.html."""
        json_path = report_dir / "report.json"
        assert json_path.exists(), f"JSON report not found at {json_path}"

    def test_json_report_is_valid(self, report_dir: Path) -> None:
        """report.json must be parseable and contain expected keys."""
        import json

        json_path = report_dir / "report.json"
        data = json.loads(json_path.read_text(encoding="utf-8"))
        for key in ("url", "prompt", "total_tests", "passed", "failed", "ai_summary"):
            assert key in data, f"Key {key!r} missing from JSON report"

    def test_ai_summary_present(self, suite_result: TestSuiteResult) -> None:
        """AI summary must be populated (not blank or error placeholder)."""
        assert suite_result.ai_summary, "AI summary is empty"
        assert "unavailable" not in suite_result.ai_summary.lower(), (
            f"AI summary indicates failure: {suite_result.ai_summary[:100]}"
        )
