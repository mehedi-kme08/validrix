from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import validrix.web_agent.crawler as crawler_module
import validrix.web_agent.executor as executor_module
import validrix.web_agent.reporter as reporter_module
from validrix.core.config_manager import AIConfig, FrameworkConfig
from validrix.web_agent.crawler import (
    PageNotFoundError,
    PlaywrightError,
    WebCrawler,
    _extract_buttons,
    _extract_forms,
    _extract_headings,
    _extract_images,
    _extract_links,
    _extract_visible_text,
)
from validrix.web_agent.executor import TestExecutor, _parse_pytest_json, _write_conftest
from validrix.web_agent.models import (
    CrawlResult,
    GeneratedTest,
    GeneratedTestSuite,
    TestSuiteResult,
)
from validrix.web_agent.models import TestResult as WebTestResult
from validrix.web_agent.reporter import (
    WebReporter,
    _build_summary_prompt,
    _call_anthropic,
    _call_openai,
    _md_to_html,
)


class _EvalPage:
    def __init__(self, values: list[object]) -> None:
        self.values = list(values)

    def evaluate(self, script: str) -> object:
        return self.values.pop(0)


def test_crawler_extract_helpers_and_classify_errors() -> None:
    page = _EvalPage(
        [
            [
                {
                    "text": "Submit",
                    "aria_label": None,
                    "element_type": "button",
                    "selector": "#submit",
                    "is_visible": True,
                },
                {"bad": "shape"},
            ],
            [
                {
                    "action": "/submit",
                    "method": "post",
                    "fields": [{"name": "email", "field_type": "email"}],
                    "submit_text": "Go",
                    "selector": "form",
                },
                {"selector": "broken", "fields": ["x"]},
            ],
            [
                {"text": "Home", "href": "https://example.com/home", "aria_label": None},
                {"text": "Skip", "href": "#section", "aria_label": None},
                {"text": "External", "href": "https://other.com", "aria_label": None},
            ],
            [
                {"src": "/img.png", "alt": "", "is_decorative": True},
                {"broken": True},
            ],
            ["Heading 1", "Heading 2"],
            "visible text" * 100,
        ]
    )

    assert len(_extract_buttons(page, "https://example.com")) == 1
    assert len(_extract_forms(page)) == 1
    links = _extract_links(page, "https://example.com")
    assert len(links) == 2
    assert links[1].is_external is True
    assert len(_extract_images(page)) == 1
    assert _extract_headings(page) == ["Heading 1", "Heading 2"]
    assert len(_extract_visible_text(page, max_chars=20)) == 20

    assert "Page load timed out" in WebCrawler._classify_error(RuntimeError("timeout happened"))
    assert "SSL/TLS error" in WebCrawler._classify_error(RuntimeError("certificate invalid"))
    assert "DNS resolution failed" in WebCrawler._classify_error(RuntimeError("ERR_NAME_NOT_RESOLVED"))
    assert WebCrawler._classify_error(PageNotFoundError("HTTP 404")) == "HTTP 404"
    assert "Browser error" in WebCrawler._classify_error(PlaywrightError("browser boom"))
    assert "Unexpected crawl error" in WebCrawler._classify_error(ValueError("x"))


def test_web_crawler_crawl_and_do_crawl(monkeypatch: pytest.MonkeyPatch) -> None:
    crawler = WebCrawler(timeout_ms=1234, headless=False)
    invalid = crawler.crawl("example.com")
    assert invalid.succeeded is False

    monkeypatch.setattr(
        crawler,
        "_do_crawl",
        lambda url: CrawlResult(url=url, title="Title"),
    )
    success = crawler.crawl("https://example.com")
    assert success.succeeded is True
    assert success.crawl_duration >= 0

    monkeypatch.setattr(crawler, "_do_crawl", lambda url: (_ for _ in ()).throw(RuntimeError("timeout")))
    failed = crawler.crawl("https://example.com")
    assert failed.succeeded is False

    class FakeResponse:
        def __init__(self, status: int) -> None:
            self.status = status

    class FakePage:
        def goto(self, url: str, wait_until: str, timeout: int) -> FakeResponse:
            return FakeResponse(200)

        def title(self) -> str:
            return "Title"

        def evaluate(self, script: str) -> object:
            if 'meta[name="description"]' in script:
                return "desc"
            if "h1, h2, h3" in script:
                return ["Heading"]
            if "document.querySelectorAll('form')" in script:
                return []
            if "document.querySelectorAll('a[href]')" in script:
                return []
            if "document.querySelectorAll('img')" in script:
                return []
            if "selectors =" in script:
                return []
            return "text"

    class FakeContext:
        def new_page(self) -> FakePage:
            return FakePage()

        def close(self) -> None:
            pass

    class FakeBrowser:
        def new_context(self, **kwargs: object) -> FakeContext:
            return FakeContext()

        def close(self) -> None:
            pass

    class FakeChromium:
        def launch(self, headless: bool) -> FakeBrowser:
            assert headless is False
            return FakeBrowser()

    class FakePlaywrightManager:
        def __enter__(self) -> SimpleNamespace:
            return SimpleNamespace(chromium=FakeChromium())

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

    monkeypatch.setattr(
        crawler_module,
        "sync_playwright",
        lambda: FakePlaywrightManager(),
    )
    result = WebCrawler(timeout_ms=1234, headless=False)._do_crawl("https://example.com")
    assert result.title == "Title"
    assert result.meta_description == "desc"


def test_executor_helpers_and_run(monkeypatch: pytest.MonkeyPatch) -> None:
    writes: dict[str, str] = {}
    monkeypatch.setattr(
        Path,
        "write_text",
        lambda self, content, encoding="utf-8": writes.__setitem__(str(self), content) or len(content),
    )
    monkeypatch.setattr(Path, "mkdir", lambda self, parents=False, exist_ok=False: None)
    monkeypatch.setattr(Path, "exists", lambda self: str(self).endswith(".json") or str(self).endswith(".png"))
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda self, encoding="utf-8": json.dumps(
            {
                "tests": [
                    {
                        "nodeid": "t",
                        "outcome": "failed",
                        "duration": 0.1,
                        "call": {"longrepr": "tb", "crash": {"message": "msg"}},
                    }
                ]
            }
        ),
    )

    parsed = _parse_pytest_json(Path("report.json"), Path("screenshots"))
    assert parsed[0].status == "failed"
    assert parsed[0].screenshot_path is not None

    monkeypatch.setattr(Path, "exists", lambda self: False)
    assert _parse_pytest_json(Path("missing.json"), Path("screenshots")) == []

    monkeypatch.setattr(Path, "exists", lambda self: True)
    monkeypatch.setattr(Path, "read_text", lambda self, encoding="utf-8": "{bad")
    assert _parse_pytest_json(Path("bad.json"), Path("screenshots")) == []
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda self, encoding="utf-8": json.dumps({"tests": [{"nodeid": "e", "outcome": "error", "duration": 0.0}]}),
    )
    monkeypatch.setattr(Path, "exists", lambda self: self.name == "error.json")
    parsed_error = _parse_pytest_json(Path("error.json"), Path("screenshots"))
    assert parsed_error[0].status == "error"
    assert parsed_error[0].error_message is None
    assert parsed_error[0].screenshot_path is None

    _write_conftest(Path("."), "https://example.com", 1000, Path("shots"))
    assert any(path.endswith("conftest.py") for path in writes)

    suite = GeneratedTestSuite(
        url="https://example.com",
        prompt="goal",
        tests=[
            GeneratedTest(
                name="test_ok",
                description="ok",
                code="code",
                test_type="functional",
            )
        ],
        combined_code="print('x')",
    )
    executor = TestExecutor(timeout_seconds=3, headless=True)
    monkeypatch.setattr(
        executor_module.tempfile,
        "mkdtemp",
        lambda prefix="validrix_": "reportdir",
    )

    class TempDir:
        def __enter__(self) -> str:
            return "tmpdir"

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

    monkeypatch.setattr(
        executor_module.tempfile,
        "TemporaryDirectory",
        lambda prefix="validrix_exec_": TempDir(),
    )
    monkeypatch.setattr(
        Path,
        "write_text",
        lambda self, content, encoding="utf-8": writes.__setitem__(str(self), content) or len(content),
    )
    monkeypatch.setattr(Path, "mkdir", lambda self, parents=False, exist_ok=False: None)
    monkeypatch.setattr(
        executor_module,
        "_write_conftest",
        lambda directory, url, timeout_ms, screenshot_dir: None,
    )
    monkeypatch.setattr(executor, "_run_subprocess", lambda cmd, cwd: (0, "out", "err"))
    monkeypatch.setattr(
        executor_module,
        "_parse_pytest_json",
        lambda json_path, screenshot_dir: [WebTestResult(test_name="x", status="passed")],
    )

    result = executor.run(suite, report_dir=Path("reports"))
    assert result.passed == 1
    assert result.total_tests == 1

    monkeypatch.setattr(
        executor_module,
        "_parse_pytest_json",
        lambda json_path, screenshot_dir: [],
    )
    monkeypatch.setattr(
        executor,
        "_infer_from_exit_code",
        lambda exit_code, suite, output: [
            WebTestResult(
                test_name="y",
                status="failed",
                error_message="bad",
            )
        ],
    )
    fallback = executor.run(suite, report_dir=Path("reports"))
    assert fallback.failed == 1
    monkeypatch.setattr(executor, "_infer_from_exit_code", TestExecutor._infer_from_exit_code)

    failed_suite = executor.run(GeneratedTestSuite(url="x", prompt="p", error="boom"), report_dir=Path("reports"))
    assert "Test generation failed" in failed_suite.ai_summary

    cmd = executor._build_pytest_command(Path("test_file.py"), Path("report.json"))
    assert "--json-report" in cmd

    monkeypatch.setattr(
        executor_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="ok", stderr=""),
    )
    subprocess_executor = TestExecutor(timeout_seconds=3, headless=True)
    assert subprocess_executor._run_subprocess(["pytest"], ".") == (0, "ok", "")

    monkeypatch.setattr(
        executor_module.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(subprocess.TimeoutExpired(cmd=["pytest"], timeout=1)),
    )
    assert subprocess_executor._run_subprocess(["pytest"], ".")[2] == "pytest subprocess timed out"

    monkeypatch.setattr(
        executor_module.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("launch failed")),
    )
    assert "launch failed" in subprocess_executor._run_subprocess(["pytest"], ".")[2]

    infer_executor = TestExecutor(timeout_seconds=3, headless=True)
    inferred = infer_executor._infer_from_exit_code(
        1,
        GeneratedTestSuite(url="u", prompt="p", tests=[], combined_code=""),
        "boom",
    )
    assert inferred[0].test_name == "test_suite"
    passed = infer_executor._infer_from_exit_code(0, suite, "")
    assert passed[0].status == "passed"
    assert passed[0].error_message is None


def test_reporter_helpers_and_generate(monkeypatch: pytest.MonkeyPatch) -> None:
    assert "<h3>Title</h3>" in _md_to_html("## Title")
    assert "<ul>" in _md_to_html("- item")
    result = TestSuiteResult(
        url="https://example.com",
        prompt="goal",
        total_tests=2,
        passed=1,
        failed=1,
        tests=[
            WebTestResult(
                test_name="test_one",
                status="failed",
                error_message="boom",
                traceback="tb",
            )
        ],
        generated_code="print('x')",
        crawl_result=CrawlResult(url="https://example.com"),
    )
    assert "FAILED TESTS" in _build_summary_prompt(result)

    anthropic_calls: list[dict[str, object]] = []
    openai_calls: list[dict[str, object]] = []

    class FakeAnthropic:
        def __init__(self, api_key: str, timeout: int) -> None:
            self.messages = SimpleNamespace(create=self.create)

        def create(self, **kwargs: object) -> object:
            anthropic_calls.append(kwargs)
            return SimpleNamespace(content=[SimpleNamespace(text="anthropic summary")])

    class FakeOpenAI:
        def __init__(self, api_key: str, timeout: int) -> None:
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs: object) -> object:
            openai_calls.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="openai summary"))])

    monkeypatch.setattr(reporter_module.anthropic, "Anthropic", FakeAnthropic)
    monkeypatch.setattr(reporter_module.openai, "OpenAI", FakeOpenAI)

    assert _call_anthropic("sys", "user", AIConfig(anthropic_api_key="k")) == "anthropic summary"
    assert _call_openai("sys", "user", AIConfig(provider="openai", openai_api_key="k")) == "openai summary"
    assert anthropic_calls and openai_calls

    writes: dict[str, str] = {}
    monkeypatch.setattr(reporter_module.ConfigManager, "load", lambda: FrameworkConfig())
    monkeypatch.setattr(Path, "mkdir", lambda self, parents=False, exist_ok=False: None)
    monkeypatch.setattr(
        Path,
        "write_text",
        lambda self, content, encoding="utf-8": writes.__setitem__(str(self), content) or len(content),
    )
    reporter = WebReporter(ai_config=AIConfig())
    monkeypatch.setattr(reporter, "_generate_ai_summary", lambda result: "## Summary\n- item")

    report_path = reporter.generate(result, report_dir=Path("reports"))
    assert report_path == Path("reports") / "report.html"
    assert result.report_path == str(report_path)
    assert "report.html" in "".join(writes)

    cfg = AIConfig()
    reporter_no_key = WebReporter(ai_config=cfg)
    assert "No AI summary" in reporter_no_key._generate_ai_summary(result)

    reporter_openai = WebReporter(ai_config=AIConfig(provider="openai", openai_api_key="x"))
    monkeypatch.setattr(reporter_module, "_call_openai", lambda system, user, config: "openai path")
    assert reporter_openai._generate_ai_summary(result) == "openai path"

    monkeypatch.setattr(
        reporter_module,
        "_call_openai",
        lambda system, user, config: (_ for _ in ()).throw(RuntimeError("ai boom")),
    )
    assert "unavailable" in reporter_openai._generate_ai_summary(result)


def test_crawler_additional_failure_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    page = _EvalPage(
        [
            [
                {
                    "text": "Home",
                    "href": "https://example.com/home",
                    "aria_label": None,
                },
                {
                    "text": "Dup",
                    "href": "https://example.com/home",
                    "aria_label": None,
                },
                {"href": "https://example.com/bad"},
            ],
            [
                {"src": "/ok.png", "alt": "ok", "is_decorative": False},
                {"bad": True},
            ],
        ]
    )

    assert len(_extract_links(page, "https://example.com")) == 1
    assert len(_extract_images(page)) == 1

    class Response404:
        status = 404

    class FailingPage:
        def goto(self, url: str, wait_until: str, timeout: int) -> Response404:
            return Response404()

    class FakeContext:
        def new_page(self) -> FailingPage:
            return FailingPage()

        def close(self) -> None:
            pass

    class FakeBrowser:
        def new_context(self, **kwargs: object) -> FakeContext:
            return FakeContext()

        def close(self) -> None:
            pass

    class FakeChromium:
        def launch(self, headless: bool) -> FakeBrowser:
            return FakeBrowser()

    class FakePlaywrightManager:
        def __enter__(self) -> SimpleNamespace:
            return SimpleNamespace(chromium=FakeChromium())

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

    monkeypatch.setattr(crawler_module, "sync_playwright", lambda: FakePlaywrightManager())
    with pytest.raises(PageNotFoundError, match="HTTP 404"):
        WebCrawler()._do_crawl("https://example.com/missing")


def test_reporter_additional_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    html = _md_to_html("# Top\n\n### Mid\n#### Low\n1. **First**\nplain")
    assert "<h3>Top</h3>" in html
    assert "<h3>Mid</h3>" in html
    assert "<h4>Low</h4>" in html
    assert "<li><strong>First</strong></li>" in html
    assert "<p>plain</p>" in html
    list_then_header = _md_to_html("- item\n## Heading")
    assert "</ul>" in list_then_header
    assert "<h3>Heading</h3>" in list_then_header
    assert "<h4>Four</h4>" in _md_to_html("- item\n#### Four")
    assert "<h3>Three</h3>" in _md_to_html("- item\n### Three")
    assert "<h3>One</h3>" in _md_to_html("- item\n# One")
    assert "</ul>" in _md_to_html("- item\n\nplain")
    assert "</ul>" in _md_to_html("1. first\nplain")
    assert _md_to_html("- one\n- two").count("<li>") == 2
    assert _md_to_html("1. one\n2. two").count("<li>") == 2

    failed_with_details = TestSuiteResult(
        url="https://example.com",
        prompt="goal",
        total_tests=2,
        failed=2,
        tests=[
            WebTestResult(
                test_name="test_case",
                status="failed",
                error_message="boom",
                traceback="line1\nline2",
            ),
            WebTestResult(
                test_name="test_case_middle",
                status="failed",
            ),
            WebTestResult(
                test_name="test_case_two",
                status="failed",
                traceback="line3",
            ),
        ],
    )
    detailed_prompt = _build_summary_prompt(failed_with_details)
    assert "Error: boom" in detailed_prompt
    assert "Traceback" in detailed_prompt
    assert "test_case_middle" in detailed_prompt
    assert "test_case_two" in detailed_prompt

    all_pass = TestSuiteResult(
        url="https://example.com",
        prompt="goal",
        total_tests=1,
        passed=1,
    )
    assert "None" in _build_summary_prompt(all_pass)

    monkeypatch.setattr(reporter_module.ConfigManager, "load", lambda: FrameworkConfig())
    reporter = WebReporter()
    monkeypatch.setattr(
        reporter,
        "_render_html",
        lambda result, report_dir: report_dir / "report.html",
    )
    monkeypatch.setattr(reporter, "_write_json", lambda result, report_dir: None)
    monkeypatch.setattr(reporter, "_generate_ai_summary", lambda result: "summary")
    monkeypatch.setattr(Path, "mkdir", lambda self, parents=False, exist_ok=False: None)
    result = TestSuiteResult(url="https://example.com", prompt="goal")
    assert reporter.generate(result) == Path("validrix_reports") / "report.html"
    rendered: dict[str, str] = {}
    reporter_real = WebReporter(ai_config=AIConfig())
    monkeypatch.setattr(
        Path,
        "write_text",
        lambda self, content, encoding="utf-8": rendered.setdefault(str(self), content) or len(content),
    )
    with_crawl = TestSuiteResult(
        url="https://example.com",
        prompt="goal",
        ai_summary="summary",
        crawl_result=CrawlResult(url="https://example.com", visible_text_sample="hidden"),
    )
    assert reporter_real._render_html(result, Path("empty-report")) == Path("empty-report") / "report.html"
    assert reporter_real._render_html(with_crawl, Path("crawl-report")) == Path("crawl-report") / "report.html"
    assert rendered
    assert "visible_text_sample" not in next(iter(rendered.values()))

    anthropic_calls: list[str] = []
    reporter_anthropic = WebReporter(ai_config=AIConfig(anthropic_api_key="x", provider="anthropic"))
    monkeypatch.setattr(
        reporter_module,
        "_call_anthropic",
        lambda system, user, config: anthropic_calls.append(user) or "anthropic path",
    )
    assert reporter_anthropic._generate_ai_summary(result) == "anthropic path"
    assert anthropic_calls
