"""
routes.py — FastAPI endpoints for the Validrix Web Agent.

Design decision: Background tasks via asyncio.create_task, not Celery.
  WHY: The Web Agent targets single-developer and small-team use cases where
       spinning up a Redis broker + Celery workers is disproportionate. Python's
       asyncio.create_task runs in the same process, requires zero extra
       infrastructure, and supports the polling pattern (POST → job_id →
       GET /status) that the UI needs.

  Alternatives considered:
    - Synchronous blocking endpoint: simple, but ties up the ASGI worker for
      30-120 seconds per request, preventing other requests from being served.
    - Celery + Redis: correct at scale; out-of-scope for v0.1, listed in roadmap.
    - FastAPI BackgroundTasks (starlette): runs after response is sent but in
      the same event loop. Works here; we use asyncio.create_task instead for
      finer control over task cancellation.

  Job storage:
    An in-memory dict keyed by UUID. Suitable for single-node development and
    demos. A persistent store (Redis, SQLite) is a planned v0.2 addition.
    We document this limitation prominently so users understand the trade-off.

  Pipeline execution:
    The pipeline (crawl → generate → execute → report) is CPU-bound and
    Playwright-bound, so we run it in a ThreadPoolExecutor via
    asyncio.get_event_loop().run_in_executor() to avoid blocking the event loop.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from validrix.web_agent.crawler import WebCrawler
from validrix.web_agent.executor import TestExecutor
from validrix.web_agent.models import (
    AnalyzeRequest,
    JobStatus,
    TestSuiteResult,
)
from validrix.web_agent.reporter import WebReporter
from validrix.web_agent.test_generator import WebTestGenerator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

# Module-level thread pool for running blocking pipeline stages
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="validrix_pipeline")

# In-memory job store: job_id → dict with status + result
# Key: str (UUID4)
# Value: {"status": str, "progress": int, "current_step": str,
#          "result": TestSuiteResult | None, "error": str | None,
#          "report_dir": Path, "created_at": datetime}
_jobs: dict[str, dict[str, Any]] = {}

_REPORT_ROOT = Path("validrix_reports")

# Pipeline step labels for the UI progress bar
_STEPS = [
    (10,  "Crawling website..."),
    (40,  "Generating test cases..."),
    (70,  "Executing tests..."),
    (95,  "Generating AI report..."),
    (100, "Complete"),
]


def create_app() -> FastAPI:
    """Application factory — returns the FastAPI instance."""
    app = FastAPI(
        title="Validrix Web Agent",
        version="0.1.0",
        description="AI-powered website test generation and execution.",
    )

    # Serve the single-page UI from validrix/ui/
    ui_dir = Path(__file__).parent.parent / "ui"
    if ui_dir.exists():
        app.mount("/ui", StaticFiles(directory=str(ui_dir), html=True), name="ui")

    app.include_router(_router)
    return app


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

from fastapi import APIRouter  # noqa: E402

_router = APIRouter(prefix="/api")


@_router.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    return {"service": "Validrix Web Agent", "docs": "/docs"}


@_router.post("/analyze", response_model=dict[str, str], status_code=202)
async def analyze(request: AnalyzeRequest) -> dict[str, str]:
    """
    Start an analysis pipeline for the given URL and prompt.

    Returns a job_id immediately; poll GET /api/status/{job_id} for progress.

    **Note**: Jobs are stored in-memory and lost on server restart.
    This is intentional for v0.1 — a persistent job store is planned for v0.2.
    """
    job_id = str(uuid.uuid4())
    report_dir = _REPORT_ROOT / job_id
    report_dir.mkdir(parents=True, exist_ok=True)

    _jobs[job_id] = {
        "status": "queued",
        "progress": 0,
        "current_step": "Queued...",
        "result": None,
        "error": None,
        "report_dir": report_dir,
        "created_at": datetime.now(UTC),
    }

    logger.info("Job %s queued for URL: %s", job_id, request.url)

    loop = asyncio.get_event_loop()
    loop.create_task(_run_pipeline(job_id, request))

    return {"job_id": job_id, "status": "queued"}


@_router.get("/status/{job_id}", response_model=JobStatus)
async def get_status(job_id: str) -> JobStatus:
    """Poll the status of a running analysis job."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found.")
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        current_step=job["current_step"],
        error=job.get("error"),
    )


@_router.get("/report/{job_id}")
async def get_report(job_id: str) -> TestSuiteResult:
    """Return the full TestSuiteResult JSON for a completed job."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found.")
    if job["status"] != "complete":
        raise HTTPException(
            status_code=409,
            detail=f"Job is not complete yet. Current status: {job['status']}",
        )
    result: TestSuiteResult | None = job.get("result")
    if result is None:
        raise HTTPException(status_code=500, detail="Job completed but result is missing.")
    return result


@_router.get("/report/{job_id}/html", response_class=HTMLResponse)
async def get_report_html(job_id: str) -> FileResponse:
    """Return the rendered HTML report file for a completed job."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found.")
    if job["status"] != "complete":
        raise HTTPException(status_code=409, detail=f"Job status: {job['status']}")

    report_path: Path = job["report_dir"] / "report.html"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="HTML report file not found.")
    return FileResponse(str(report_path), media_type="text/html")


# ---------------------------------------------------------------------------
# Pipeline runner (executed in thread pool to avoid blocking event loop)
# ---------------------------------------------------------------------------


async def _run_pipeline(job_id: str, request: AnalyzeRequest) -> None:
    """
    Orchestrate the full crawl → generate → execute → report pipeline.

    Updates _jobs[job_id] at each stage so the /status endpoint can report
    live progress to the UI.
    """
    loop = asyncio.get_event_loop()

    def _update(progress: int, step: str, status: str = "running") -> None:
        _jobs[job_id].update({"status": status, "progress": progress, "current_step": step})
        logger.info("Job %s [%d%%] %s", job_id, progress, step)

    try:
        _update(0, "Starting pipeline...")

        # ── Stage 1: Crawl ────────────────────────────────────────────
        _update(*_STEPS[0])
        crawl_result = await loop.run_in_executor(
            _executor,
            lambda: WebCrawler(
                timeout_ms=request.options.timeout_seconds * 1000,
                headless=request.options.headless,
            ).crawl(request.url),
        )
        if not crawl_result.succeeded:
            _jobs[job_id].update({
                "status": "failed",
                "current_step": "Crawl failed",
                "error": crawl_result.error,
            })
            return

        # ── Stage 2: Generate tests ───────────────────────────────────
        _update(*_STEPS[1])
        report_dir: Path = _jobs[job_id]["report_dir"]
        test_file = report_dir / "test_generated.py"
        suite = await loop.run_in_executor(
            _executor,
            lambda: WebTestGenerator().generate(
                crawl=crawl_result,
                prompt=request.prompt,
                max_tests=request.options.max_tests,
                output_path=test_file,
            ),
        )
        if not suite.succeeded:
            _jobs[job_id].update({
                "status": "failed",
                "current_step": "Test generation failed",
                "error": suite.error,
            })
            return

        # ── Stage 3: Execute tests ────────────────────────────────────
        _update(*_STEPS[2])
        suite_result = await loop.run_in_executor(
            _executor,
            lambda: TestExecutor(
                timeout_seconds=request.options.timeout_seconds,
                headless=request.options.headless,
            ).run(suite, report_dir=report_dir),
        )
        suite_result.crawl_result = crawl_result

        # ── Stage 4: Generate report ──────────────────────────────────
        _update(*_STEPS[3])
        await loop.run_in_executor(
            _executor,
            lambda: WebReporter().generate(suite_result, report_dir=report_dir),
        )

        # ── Done ──────────────────────────────────────────────────────
        _jobs[job_id].update({
            "status": "complete",
            "progress": 100,
            "current_step": "Complete",
            "result": suite_result,
        })
        logger.info(
            "Job %s complete: %d/%d tests passed",
            job_id,
            suite_result.passed,
            suite_result.total_tests,
        )

    except Exception as exc:
        logger.exception("Job %s failed with unexpected error: %s", job_id, exc)
        _jobs[job_id].update({
            "status": "failed",
            "current_step": "Internal error",
            "error": str(exc),
        })
