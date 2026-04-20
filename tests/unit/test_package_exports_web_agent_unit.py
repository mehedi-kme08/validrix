from __future__ import annotations

import importlib
from pathlib import Path

import validrix
import validrix.api
import validrix.web_agent
from validrix.web_agent.models import (
    CrawlResult,
    GeneratedTestSuite,
)
from validrix.web_agent.models import (
    TestSuiteResult as WebSuiteResult,
)


def test_validrix_package_exports_include_web_agent() -> None:
    assert "__version__" in validrix.__all__
    assert "web_agent" in validrix.__all__
    assert "app" in validrix.api.__all__
    assert "run_pipeline" in validrix.web_agent.__all__


def test_run_pipeline_uses_components(monkeypatch: object) -> None:
    crawl_result = CrawlResult(url="https://example.com")
    suite = GeneratedTestSuite(url="https://example.com", prompt="goal", combined_code="code")
    executed = WebSuiteResult(
        url="https://example.com",
        prompt="goal",
        total_tests=1,
        passed=1,
        generated_code="code",
    )
    reporter_called: list[tuple[WebSuiteResult, Path]] = []

    class FakeCrawler:
        def __init__(self, timeout_ms: int, headless: bool) -> None:
            assert timeout_ms == 5000
            assert headless is False

        def crawl(self, url: str) -> CrawlResult:
            assert url == "https://example.com"
            return crawl_result

    class FakeGenerator:
        def generate(self, crawl: CrawlResult, prompt: str, max_tests: int) -> GeneratedTestSuite:
            assert crawl is crawl_result
            assert prompt == "goal"
            assert max_tests == 3
            return suite

    class FakeExecutor:
        def __init__(self, timeout_seconds: int, headless: bool) -> None:
            assert timeout_seconds == 5
            assert headless is False

        def run(self, suite: GeneratedTestSuite, report_dir: Path) -> WebSuiteResult:
            assert suite is suite
            assert report_dir == Path("reports")
            return executed

    class FakeReporter:
        def generate(self, result: WebSuiteResult, report_dir: Path) -> None:
            reporter_called.append((result, report_dir))

    monkeypatch.setattr(validrix.web_agent, "WebCrawler", FakeCrawler)
    monkeypatch.setattr(validrix.web_agent, "WebTestGenerator", FakeGenerator)
    monkeypatch.setattr(validrix.web_agent, "TestExecutor", FakeExecutor)
    monkeypatch.setattr(validrix.web_agent, "WebReporter", FakeReporter)

    result = validrix.web_agent.run_pipeline(
        url="https://example.com",
        prompt="goal",
        report_dir=Path("reports"),
        max_tests=3,
        timeout_seconds=5,
        headless=False,
    )

    assert result is executed
    assert result.crawl_result is crawl_result
    assert reporter_called == [(executed, Path("reports"))]


def test_reload_api_and_web_agent_modules() -> None:
    importlib.reload(importlib.import_module("validrix.api"))
    importlib.reload(importlib.import_module("validrix.web_agent"))
