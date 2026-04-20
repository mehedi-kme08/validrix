from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import validrix.api
import validrix.api.routes as routes
from validrix.web_agent.models import (
    AnalysisOptions,
    AnalyzeRequest,
    CrawlResult,
    GeneratedTestSuite,
    JobStatus,
    TestSuiteResult,
)


def test_api_init_and_create_app(monkeypatch: pytest.MonkeyPatch) -> None:
    mounted: list[tuple[str, object, str]] = []

    class FakeApp:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            self.routes: list[object] = []

        def mount(self, path: str, app: object, name: str) -> None:
            mounted.append((path, app, name))

        def include_router(self, router: object) -> None:
            self.routes.append(router)

    monkeypatch.setattr(routes, "FastAPI", FakeApp)
    monkeypatch.setattr(Path, "exists", lambda self: True)
    monkeypatch.setattr(routes, "StaticFiles", lambda directory, html: SimpleNamespace(directory=directory, html=html))

    app = routes.create_app()
    assert mounted[0][0] == "/ui"
    assert routes._router in app.routes

    monkeypatch.setattr(Path, "exists", lambda self: False)
    app_without_ui = routes.create_app()
    assert app_without_ui.routes == [routes._router]

    importlib.reload(validrix.api)
    assert "app" in validrix.api.__all__


@pytest.mark.asyncio
async def test_route_helpers_status_and_report_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    routes._jobs.clear()
    assert await routes.root() == {"service": "Validrix Web Agent", "docs": "/docs"}

    with pytest.raises(HTTPException):
        await routes.get_status("missing")

    routes._jobs["1"] = {"status": "running", "progress": 10, "current_step": "step", "error": None}
    status = await routes.get_status("1")
    assert isinstance(status, JobStatus)
    assert status.status == "running"

    with pytest.raises(HTTPException):
        await routes.get_report("missing")

    routes._jobs["2"] = {"status": "running", "result": None}
    with pytest.raises(HTTPException):
        await routes.get_report("2")

    routes._jobs["3"] = {"status": "complete", "result": None}
    with pytest.raises(HTTPException):
        await routes.get_report("3")

    result = TestSuiteResult(url="u", prompt="p")
    routes._jobs["4"] = {"status": "complete", "result": result}
    assert await routes.get_report("4") is result

    with pytest.raises(HTTPException):
        await routes.get_report_html("missing")

    routes._jobs["5"] = {"status": "running", "report_dir": Path("reports")}
    with pytest.raises(HTTPException):
        await routes.get_report_html("5")

    monkeypatch.setattr(Path, "exists", lambda self: False)
    routes._jobs["6"] = {"status": "complete", "report_dir": Path("reports")}
    with pytest.raises(HTTPException):
        await routes.get_report_html("6")

    monkeypatch.setattr(Path, "exists", lambda self: True)
    monkeypatch.setattr(
        routes,
        "FileResponse",
        lambda path, media_type: SimpleNamespace(path=path, media_type=media_type),
    )
    response = await routes.get_report_html("6")
    assert response.media_type == "text/html"


@pytest.mark.asyncio
async def test_analyze_and_run_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    routes._jobs.clear()
    created_tasks: list[object] = []

    class FakeLoop:
        def create_task(self, coro: object) -> None:
            created_tasks.append(coro)

        async def run_in_executor(self, executor: object, func: object) -> object:
            return func()

        def is_closed(self) -> bool:
            return False

        def close(self) -> None:
            return None

    fake_loop = FakeLoop()
    monkeypatch.setattr(routes.asyncio, "get_event_loop", lambda: fake_loop)
    monkeypatch.setattr(routes.uuid, "uuid4", lambda: "job-1")
    monkeypatch.setattr(Path, "mkdir", lambda self, parents=False, exist_ok=False: None)

    request = AnalyzeRequest(
        url="https://example.com",
        prompt="check goal",
        options=AnalysisOptions(),
    )
    response = await routes.analyze(request)
    assert response == {"job_id": "job-1", "status": "queued"}
    assert routes._jobs["job-1"]["status"] == "queued"
    assert created_tasks
    created_tasks[0].close()

    crawl = CrawlResult(url="https://example.com")
    suite = GeneratedTestSuite(
        url="https://example.com",
        prompt="check goal",
        combined_code="code",
    )
    result = TestSuiteResult(
        url="https://example.com",
        prompt="check goal",
        total_tests=1,
        passed=1,
    )

    class FakeCrawler:
        def __init__(self, timeout_ms: int, headless: bool) -> None:
            pass

        def crawl(self, url: str) -> CrawlResult:
            return crawl

    class FakeGenerator:
        def generate(self, crawl: CrawlResult, prompt: str, max_tests: int, output_path: Path) -> GeneratedTestSuite:
            assert output_path.name == "test_generated.py"
            return suite

    class FakeExecutor:
        def __init__(self, timeout_seconds: int, headless: bool) -> None:
            pass

        def run(self, suite: GeneratedTestSuite, report_dir: Path) -> TestSuiteResult:
            return result

    class FakeReporter:
        def generate(self, suite_result: TestSuiteResult, report_dir: Path) -> None:
            return None

    monkeypatch.setattr(routes, "WebCrawler", FakeCrawler)
    monkeypatch.setattr(routes, "WebTestGenerator", FakeGenerator)
    monkeypatch.setattr(routes, "TestExecutor", FakeExecutor)
    monkeypatch.setattr(routes, "WebReporter", FakeReporter)

    routes._jobs["job-2"] = {
        "report_dir": Path("reports"),
        "status": "queued",
        "progress": 0,
        "current_step": "",
        "result": None,
        "error": None,
    }
    await routes._run_pipeline("job-2", request)
    assert routes._jobs["job-2"]["status"] == "complete"
    assert routes._jobs["job-2"]["result"] is result
    assert result.crawl_result is crawl


@pytest.mark.asyncio
async def test_run_pipeline_failure_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeLoop:
        async def run_in_executor(self, executor: object, func: object) -> object:
            return func()

        def is_closed(self) -> bool:
            return False

        def close(self) -> None:
            return None

    fake_loop = FakeLoop()
    monkeypatch.setattr(routes.asyncio, "get_event_loop", lambda: fake_loop)
    request = AnalyzeRequest(
        url="https://example.com",
        prompt="check goal",
        options=AnalysisOptions(),
    )

    routes._jobs["crawl_fail"] = {
        "report_dir": Path("reports"),
        "status": "queued",
        "progress": 0,
        "current_step": "",
        "result": None,
        "error": None,
    }
    monkeypatch.setattr(
        routes,
        "WebCrawler",
        lambda timeout_ms, headless: SimpleNamespace(crawl=lambda url: CrawlResult(url=url, error="crawl boom")),
    )
    await routes._run_pipeline("crawl_fail", request)
    assert routes._jobs["crawl_fail"]["status"] == "failed"

    routes._jobs["generate_fail"] = {
        "report_dir": Path("reports"),
        "status": "queued",
        "progress": 0,
        "current_step": "",
        "result": None,
        "error": None,
    }
    monkeypatch.setattr(
        routes,
        "WebCrawler",
        lambda timeout_ms, headless: SimpleNamespace(crawl=lambda url: CrawlResult(url=url)),
    )
    monkeypatch.setattr(
        routes,
        "WebTestGenerator",
        lambda: SimpleNamespace(
            generate=lambda **kwargs: GeneratedTestSuite(
                url="u",
                prompt="p",
                error="gen boom",
            )
        ),
    )
    await routes._run_pipeline("generate_fail", request)
    assert routes._jobs["generate_fail"]["status"] == "failed"

    routes._jobs["internal_fail"] = {
        "report_dir": Path("reports"),
        "status": "queued",
        "progress": 0,
        "current_step": "",
        "result": None,
        "error": None,
    }
    monkeypatch.setattr(
        routes,
        "WebTestGenerator",
        lambda: (_ for _ in ()).throw(RuntimeError("explode")),
    )
    await routes._run_pipeline("internal_fail", request)
    assert routes._jobs["internal_fail"]["status"] == "failed"
