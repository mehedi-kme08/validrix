"""
test_generator.py — AI-powered test suite generator for crawled web pages.

Design decision: Structured crawl data as context, not raw HTML.
  WHY: Sending raw HTML to Claude is noisy — thousands of tokens of CSS
       class names, inline styles, and script tags that are irrelevant to
       test generation. Instead we send a compact JSON summary of the
       structural elements (buttons, forms, links, headings) that directly
       map to what a QA engineer would test. This reduces token usage by
       ~70% and produces more focused tests.

  Alternatives considered:
    - Send raw HTML: high token cost, low signal-to-noise, model struggles
      to ignore irrelevant markup.
    - Send a screenshot + vision model: powerful but slow and 10× more
      expensive per call; no concrete selectors to put in generated code.
    - Template-based generation: deterministic but cannot adapt to the
      user's plain-English prompt or infer test scenarios.

  Tradeoffs:
    - We re-use the same _LLMClient ABC from ai_generator.py indirectly by
      duplicating the thin Anthropic/OpenAI wrappers. This keeps web_agent
      self-contained without importing from plugins/ (avoiding coupling).
    - Generated tests may need minor tweaks for site-specific auth flows.
      We add a comment block at the top of every file calling this out.

  Prompt strategy:
    - System prompt sets the role (QA engineer + Playwright expert).
    - User prompt embeds crawl data as a structured text block, then states
      the user's plain-English goal. This separation lets the model
      distinguish "what exists on the page" from "what to test".
"""

from __future__ import annotations

import logging
import re
import textwrap
import time
from pathlib import Path
from typing import Final

import anthropic
import openai

from validrix.core.config_manager import AIConfig, ConfigManager
from validrix.web_agent.models import (
    CrawlResult,
    GeneratedTest,
    GeneratedTestSuite,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt — versioned here, tracked via git
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT: Final[str] = textwrap.dedent("""
    You are an expert QA automation engineer specialising in Python, pytest, and Playwright.
    Your task is to generate a complete, runnable pytest test file from a structured
    description of a live web page and a plain-English testing goal.

    Rules:
    1. Output ONLY valid Python code — no explanation, no markdown prose outside the code fence.
    2. Wrap all code in a single ```python ... ``` fence.
    3. Start the file with the required imports:
         import pytest
         from playwright.sync_api import Page, expect
    4. Each test function must:
         - Start with `test_`
         - Accept `page: Page` as its first parameter (Playwright fixture)
         - Begin with `page.goto(BASE_URL)` to navigate to the target URL
         - Have a one-line docstring explaining what it verifies
         - Use `expect()` for Playwright assertions (not assert statements)
         - Handle waits with `expect(...).to_be_visible(timeout=10000)`
    5. Define `BASE_URL` as a module-level constant at the top of the file.
    6. Cover what the user explicitly asked to test PLUS:
         - At least one navigation/page-load test (verify title or heading)
         - At least one positive test (happy path)
         - At least one negative/edge-case test where applicable
    7. Keep selectors resilient: prefer aria-label, role, and text over brittle CSS IDs.
    8. Use `pytest.mark.parametrize` for data-driven cases (e.g. valid/invalid inputs).
    9. Use `@pytest.mark.smoke` for critical path tests and `@pytest.mark.regression`
       for edge cases.
    10. Maximum tests: respect the MAX_TESTS limit specified in the user prompt.
    11. Do NOT generate tests for external links or third-party services.
    12. Do NOT add authentication unless the user's prompt explicitly requests it.
""").strip()


# ---------------------------------------------------------------------------
# LLM wrappers (self-contained, no import from plugins/)
# ---------------------------------------------------------------------------


def _call_anthropic(system: str, user: str, config: AIConfig) -> str:
    client = anthropic.Anthropic(
        api_key=config.anthropic_api_key,
        timeout=config.timeout_seconds,
    )
    message = client.messages.create(
        model=config.model,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return message.content[0].text  # type: ignore[union-attr]


def _call_openai(system: str, user: str, config: AIConfig) -> str:
    client = openai.OpenAI(
        api_key=config.openai_api_key,
        timeout=config.timeout_seconds,
    )
    response = client.chat.completions.create(
        model=config.model,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def _build_crawl_summary(crawl: CrawlResult) -> str:
    """Serialise CrawlResult into a compact, prompt-friendly text block."""
    parts: list[str] = [
        f"PAGE URL: {crawl.url}",
        f"TITLE: {crawl.title or '(no title)'}",
        f"META DESCRIPTION: {crawl.meta_description or '(none)'}",
    ]

    if crawl.headings:
        parts.append("HEADINGS (h1–h3):\n" + "\n".join(f"  - {h}" for h in crawl.headings[:15]))

    if crawl.buttons:
        btn_lines = []
        for b in crawl.buttons[:20]:
            btn_lines.append(
                f"  - [{b.element_type}] text={b.text!r} aria-label={b.aria_label!r} selector={b.selector!r}"
            )
        parts.append("BUTTONS:\n" + "\n".join(btn_lines))

    if crawl.forms:
        for i, form in enumerate(crawl.forms[:5], 1):
            field_lines = "\n".join(
                f"      field: name={f.name!r} type={f.field_type} required={f.is_required}" for f in form.fields
            )
            parts.append(
                f"FORM {i}: selector={form.selector!r} method={form.method}\n"
                f"  submit_text={form.submit_text!r}\n"
                f"{field_lines}"
            )

    if crawl.links:
        internal = [lk for lk in crawl.links if not lk.is_external][:15]
        if internal:
            parts.append("INTERNAL LINKS:\n" + "\n".join(f"  - {lk.text!r} → {lk.href}" for lk in internal))

    if crawl.visible_text_sample:
        parts.append(f"VISIBLE TEXT SAMPLE (first 500 chars):\n{crawl.visible_text_sample[:500]}")

    return "\n\n".join(parts)


def _build_user_prompt(crawl: CrawlResult, user_prompt: str, max_tests: int) -> str:
    crawl_summary = _build_crawl_summary(crawl)
    return (
        f"=== WEBSITE STRUCTURE ===\n{crawl_summary}\n\n"
        f"=== TESTING GOAL ===\n{user_prompt}\n\n"
        f"=== CONSTRAINTS ===\n"
        f"MAX_TESTS: {max_tests}\n"
        f"Generate at most {max_tests} test functions. Cover the most important "
        f"scenarios first.\n"
    )


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------


def _extract_code(raw: str) -> str:
    """Extract Python source from a markdown-fenced code block."""
    match = re.search(r"```(?:python)?\s*\n(.*?)```", raw, re.DOTALL)
    if match:
        return match.group(1).strip()
    stripped = raw.strip()
    if stripped:
        return stripped
    raise ValueError(f"AI response contained no extractable Python code. Raw response (first 200 chars): {raw[:200]!r}")


def _add_file_header(code: str, url: str, prompt: str) -> str:
    header = textwrap.dedent(f"""\
        # =============================================================================
        # AUTO-GENERATED by Validrix Web Agent
        # Target URL : {url}
        # User Prompt: {prompt}
        # Review generated selectors before merging into your permanent test suite.
        # =============================================================================

    """)
    return header + code


def _parse_tests_from_code(code: str) -> list[GeneratedTest]:
    """
    Parse individual test functions out of the combined code block.

    We extract metadata by inspecting the AST-level structure so that each
    GeneratedTest carries its docstring and can be displayed individually in
    the report. Falls back to a single catch-all entry if parsing fails.
    """
    import ast

    tests: list[GeneratedTest] = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Return the whole file as a single test entry rather than crashing
        return [
            GeneratedTest(
                name="test_generated_suite",
                description="Full generated test suite",
                code=code,
                test_type="functional",
            )
        ]

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if not node.name.startswith("test_"):
            continue

        docstring = ast.get_docstring(node) or ""
        test_type = _infer_test_type(node.name, docstring)

        # Extract the source lines for just this function
        func_lines = code.splitlines()
        start = node.lineno - 1
        end = node.end_lineno if hasattr(node, "end_lineno") else len(func_lines)  # type: ignore[attr-defined]
        func_code = "\n".join(func_lines[start:end])

        tests.append(
            GeneratedTest(
                name=node.name,
                description=docstring or node.name.replace("_", " ").title(),
                code=func_code,
                test_type=test_type,
            )
        )

    return tests or [
        GeneratedTest(
            name="test_generated_suite",
            description="Full generated test suite",
            code=code,
            test_type="functional",
        )
    ]


def _infer_test_type(name: str, docstring: str) -> str:
    """Heuristically classify a test function into a type bucket."""
    combined = (name + " " + docstring).lower()
    if any(kw in combined for kw in ("form", "submit", "input", "fill", "type")):
        return "form"
    if any(kw in combined for kw in ("nav", "link", "navigate", "click", "goto", "menu")):
        return "navigation"
    if any(kw in combined for kw in ("image", "alt", "visual", "screenshot")):
        return "visual"
    if any(kw in combined for kw in ("aria", "role", "a11y", "accessibility", "tab")):
        return "accessibility"
    return "functional"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class WebTestGenerator:
    """
    Generates a pytest + Playwright test suite from a CrawlResult and user prompt.

    Example::

        gen = WebTestGenerator()
        suite = gen.generate(
            crawl=crawler.crawl("https://example.com"),
            prompt="Test the contact form validation",
            max_tests=8,
        )
        if suite.succeeded:
            Path("tests/test_generated.py").write_text(suite.combined_code)
    """

    def __init__(self, ai_config: AIConfig | None = None) -> None:
        cfg = ConfigManager.load()
        self._ai_config = ai_config or cfg.ai

    def generate(
        self,
        crawl: CrawlResult,
        prompt: str,
        max_tests: int = 10,
        output_path: Path | None = None,
    ) -> GeneratedTestSuite:
        """
        Generate test functions for the crawled page.

        Args:
            crawl:       Structured output from WebCrawler.crawl().
            prompt:      Plain-English description of what to test.
            max_tests:   Upper bound on number of generated test functions.
            output_path: If provided, write the combined code to this file.

        Returns:
            GeneratedTestSuite — check ``.succeeded`` before using other fields.
        """
        if not crawl.succeeded:
            return GeneratedTestSuite(
                url=crawl.url,
                prompt=prompt,
                error=f"Crawl failed — cannot generate tests: {crawl.error}",
            )

        logger.info("Generating tests for %s (max=%d): %r", crawl.url, max_tests, prompt)
        start = time.monotonic()

        try:
            user_prompt = _build_user_prompt(crawl, prompt, max_tests)
            raw = self._call_ai(user_prompt)
            code = _extract_code(raw)
            code_with_header = _add_file_header(code, crawl.url, prompt)
            tests = _parse_tests_from_code(code)
        except Exception as exc:
            elapsed = time.monotonic() - start
            logger.error("Test generation failed: %s", exc)
            return GeneratedTestSuite(
                url=crawl.url,
                prompt=prompt,
                generation_duration=elapsed,
                error=str(exc),
            )

        elapsed = time.monotonic() - start
        logger.info("Generated %d test function(s) in %.1fs", len(tests), elapsed)

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(code_with_header, encoding="utf-8")
            logger.info("Generated tests written to: %s", output_path)

        return GeneratedTestSuite(
            url=crawl.url,
            prompt=prompt,
            tests=tests,
            combined_code=code_with_header,
            generation_duration=elapsed,
        )

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _call_ai(self, user_prompt: str) -> str:
        if self._ai_config.provider == "anthropic":
            return _call_anthropic(_SYSTEM_PROMPT, user_prompt, self._ai_config)
        return _call_openai(_SYSTEM_PROMPT, user_prompt, self._ai_config)
