"""
models.py — Pydantic data contracts for the web_agent pipeline.

Design decision: Pydantic v2 BaseModel for all pipeline boundaries.
  WHY: Every hand-off between crawler → generator → executor → reporter
       is a serialisation boundary. Pydantic validates shape at that
       boundary, not halfway through execution. This catches crawl bugs
       before they corrupt test generation, and executor bugs before they
       corrupt the report.

  Alternatives considered:
    - dataclasses: no built-in validation, no JSON serialisation helpers.
    - TypedDict: no default values, no validators, no .model_dump().
    - attrs: viable but adds a dep; Pydantic is already in the tree.

  Tradeoffs:
    - All models are immutable by default (model_config frozen=True where
      appropriate) to prevent accidental mutation across pipeline stages.
    - Optional fields use `None` defaults rather than empty strings so that
      callers can distinguish "not found" from "empty string found".
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator


# ---------------------------------------------------------------------------
# Primitive element models — building blocks of CrawlResult
# ---------------------------------------------------------------------------


class ButtonElement(BaseModel):
    """A single interactive button discovered on the page."""

    text: str = Field(description="Visible button text, stripped of whitespace.")
    aria_label: str | None = Field(default=None, description="aria-label attribute, if present.")
    element_type: str = Field(default="button", description="Tag name (button, input[type=submit], a[role=button]).")
    selector: str = Field(description="CSS selector that uniquely identifies this element.")
    is_visible: bool = Field(default=True, description="Whether the element was visible in the viewport.")


class FormField(BaseModel):
    """A single field within a form."""

    name: str = Field(description="name or id attribute of the input.")
    field_type: str = Field(
        default="text",
        description="HTML input type: text, email, password, checkbox, select, textarea, …",
    )
    placeholder: str | None = Field(default=None, description="Placeholder text, if any.")
    is_required: bool = Field(default=False, description="Whether the field has the required attribute.")
    label: str | None = Field(default=None, description="Associated <label> text, if discoverable.")


class FormElement(BaseModel):
    """A complete <form> element and its constituent fields."""

    action: str | None = Field(default=None, description="Form action URL (relative or absolute).")
    method: str = Field(default="get", description="HTTP method: get or post.")
    fields: list[FormField] = Field(default_factory=list)
    submit_text: str | None = Field(default=None, description="Text of the submit button, if identifiable.")
    selector: str = Field(description="CSS selector for the form element itself.")


class LinkElement(BaseModel):
    """A navigation or content link found on the page."""

    text: str = Field(description="Visible anchor text.")
    href: str = Field(description="href attribute value (may be relative).")
    is_external: bool = Field(default=False, description="True if href points to a different domain.")
    aria_label: str | None = Field(default=None)


class ImageElement(BaseModel):
    """An <img> or background image discovered on the page."""

    src: str = Field(description="src attribute value.")
    alt: str = Field(default="", description="alt attribute — empty string if missing (accessibility concern).")
    is_decorative: bool = Field(
        default=False,
        description="True if alt='' (intentionally decorative per WCAG).",
    )


# ---------------------------------------------------------------------------
# CrawlResult — output of crawler.py
# ---------------------------------------------------------------------------


class CrawlResult(BaseModel):
    """
    Complete structural snapshot of a webpage produced by the crawler.

    All list fields are empty lists (not None) on success so that downstream
    code can iterate without null checks.
    """

    url: str = Field(description="The URL that was crawled.")
    title: str = Field(default="", description="Document <title> text.")
    meta_description: str = Field(default="", description="<meta name='description'> content.")
    headings: list[str] = Field(
        default_factory=list,
        description="All h1–h3 text nodes, in document order.",
    )
    buttons: list[ButtonElement] = Field(default_factory=list)
    forms: list[FormElement] = Field(default_factory=list)
    links: list[LinkElement] = Field(default_factory=list)
    images: list[ImageElement] = Field(default_factory=list)
    visible_text_sample: str = Field(
        default="",
        description="First 2000 chars of visible body text — gives AI context about page content.",
    )
    crawl_duration: float = Field(default=0.0, description="Seconds taken to crawl the page.")
    error: str | None = Field(
        default=None,
        description="Non-None if crawl failed. Downstream components must check this before using other fields.",
    )

    @property
    def succeeded(self) -> bool:
        """Convenience: True when crawl completed without error."""
        return self.error is None


# ---------------------------------------------------------------------------
# GeneratedTest — output of test_generator.py
# ---------------------------------------------------------------------------


class GeneratedTest(BaseModel):
    """A single pytest test function produced by the AI generator."""

    name: str = Field(description="Python function name, e.g. 'test_login_with_valid_credentials'.")
    description: str = Field(description="Plain-English summary of what this test verifies.")
    code: str = Field(description="Complete pytest function source code, including imports.")
    test_type: Literal["functional", "navigation", "form", "accessibility", "visual"] = Field(
        description="Broad category used for grouping in the HTML report.",
    )


class GeneratedTestSuite(BaseModel):
    """
    Collection of tests generated for a single URL + prompt pair.

    Includes the combined source file that the executor will run.
    """

    url: str
    prompt: str
    tests: list[GeneratedTest] = Field(default_factory=list)
    combined_code: str = Field(
        default="",
        description="Single Python file containing all generated test functions and shared fixtures.",
    )
    generation_duration: float = Field(default=0.0, description="Seconds taken for AI generation.")
    error: str | None = Field(default=None)

    @property
    def succeeded(self) -> bool:
        return self.error is None


# ---------------------------------------------------------------------------
# TestResult / TestSuiteResult — output of executor.py
# ---------------------------------------------------------------------------


class TestResult(BaseModel):
    """Execution outcome for a single pytest test function."""

    test_name: str = Field(description="Fully qualified test node ID, e.g. 'test_generated.py::test_login'.")
    status: Literal["passed", "failed", "skipped", "error"] = Field(
        description="'error' indicates a collection/setup error distinct from a test assertion failure.",
    )
    error_message: str | None = Field(
        default=None,
        description="The assertion error or exception message. None on pass.",
    )
    traceback: str | None = Field(default=None, description="Full traceback text for debugging.")
    screenshot_path: str | None = Field(
        default=None,
        description="Absolute path to the failure screenshot PNG, if captured.",
    )
    duration: float = Field(default=0.0, description="Wall-clock seconds from test start to end.")


class TestSuiteResult(BaseModel):
    """
    Complete output of a single web_agent analysis pipeline run.

    This is the top-level data contract returned to the API and used to
    render the HTML report. Every upstream model funnels into this one.
    """

    url: str = Field(description="The target URL that was tested.")
    prompt: str = Field(description="The user's original plain-English prompt.")
    total_tests: int = Field(default=0)
    passed: int = Field(default=0)
    failed: int = Field(default=0)
    skipped: int = Field(default=0)
    duration: float = Field(default=0.0, description="Total wall-clock time for the executor phase.")
    tests: list[TestResult] = Field(default_factory=list)
    ai_summary: str = Field(
        default="",
        description="AI-generated executive summary — plain English, suitable for non-technical readers.",
    )
    generated_code: str = Field(
        default="",
        description="The combined Python test file that was executed.",
    )
    crawl_result: CrawlResult | None = Field(
        default=None,
        description="Retained for report rendering — shows what the crawler found.",
    )
    report_path: str | None = Field(default=None, description="Absolute path to the generated HTML report.")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def pass_rate(self) -> float:
        """Fraction of tests that passed, 0.0 if total_tests == 0."""
        return self.passed / self.total_tests if self.total_tests else 0.0

    @property
    def health_label(self) -> Literal["healthy", "degraded", "critical", "untested"]:
        """Quick health signal for the dashboard badge."""
        if self.total_tests == 0:
            return "untested"
        if self.pass_rate == 1.0:
            return "healthy"
        if self.pass_rate >= 0.5:
            return "degraded"
        return "critical"


# ---------------------------------------------------------------------------
# API layer request / response models — consumed by routes.py
# ---------------------------------------------------------------------------


class AnalysisOptions(BaseModel):
    """Tuning knobs the API caller can override per request."""

    max_tests: int = Field(default=10, ge=1, le=30, description="Cap on the number of tests the AI may generate.")
    timeout_seconds: int = Field(default=60, ge=10, le=300, description="Per-test execution timeout.")
    screenshot_on_pass: bool = Field(
        default=False,
        description="If True, capture screenshots even for passing tests (increases storage use).",
    )
    headless: bool = Field(default=True, description="Run Playwright in headless mode.")


class AnalyzeRequest(BaseModel):
    """Body of POST /api/analyze."""

    url: str = Field(description="The live website URL to crawl and test.")
    prompt: str = Field(
        description="Plain-English description of what to test, e.g. 'Test the login flow'.",
        min_length=5,
    )
    options: AnalysisOptions = Field(default_factory=AnalysisOptions)

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Reject obviously invalid URLs before touching the network."""
        if not v.startswith(("http://", "https://")):
            raise ValueError("url must start with http:// or https://")
        return v.rstrip("/")


class JobStatus(BaseModel):
    """Response body for GET /api/status/{job_id}."""

    job_id: str
    status: Literal["queued", "running", "complete", "failed"]
    progress: int = Field(default=0, ge=0, le=100, description="Percentage complete, 0-100.")
    current_step: str = Field(default="", description="Human-readable description of the active pipeline stage.")
    error: str | None = Field(default=None)
