"""
reporter.py — AI failure summariser and HTML report generator for web_agent.

Design decision: Inline Jinja2 template, not a separate .html file.
  WHY: The web_agent module ships as a Python package. Bundling the template
       as a string constant inside reporter.py means zero file-system
       dependency — it works whether the package is installed from PyPI,
       a wheel, or a git clone. External .html templates require either
       pkg_resources/importlib.resources machinery or a known file path,
       both of which break in certain deployment scenarios (e.g. zip imports).

  Alternatives considered:
    - External template file with importlib.resources: correct, but adds
      boilerplate and a MANIFEST.in entry. Template changes require remembering
      to update package_data in pyproject.toml.
    - Markdown-only report (no HTML): simpler, but screenshots can't be inlined
      and the non-technical stakeholder experience is poorer.
    - React/Vue SPA report: rich but requires a build step incompatible with
      a pure-Python library.

  Tradeoffs:
    - Inline template is harder to edit visually (no browser hot-reload).
      We mitigate by keeping the template clean and well-commented.
    - Screenshots are referenced by absolute path (file:// URL), which works
      in desktop browsers but not on remote servers. A future enhancement
      could base64-inline them.

  AI summary strategy:
    - We call the AI ONCE with all failures batched (same approach as
      ai_reporter.py) to get cross-failure pattern analysis.
    - If there are no failures, the AI receives a pass summary and returns
      a short health confirmation — avoids a wasted API call on 100% pass.
"""

from __future__ import annotations

import json
import logging
import textwrap
import time
from pathlib import Path
from typing import Final

import anthropic
import openai
from jinja2 import Environment, select_autoescape

from validrix.core.config_manager import AIConfig, ConfigManager
from validrix.web_agent.models import TestSuiteResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# AI system prompt for executive summary
# ---------------------------------------------------------------------------

_SUMMARY_SYSTEM_PROMPT: Final[str] = textwrap.dedent("""
    You are a QA lead writing an executive summary for a non-technical stakeholder.
    Analyse the provided automated test results for a live website and produce:

    1. **Overall website health assessment** (1-2 sentences)
    2. **Root cause analysis** of each failure (be specific about selectors, URLs, errors)
    3. **Priority fixes needed** (ordered: Critical → High → Medium)
    4. **Patterns in failures** (if multiple tests share the same root cause, call it out)
    5. **Recommended next steps** (actionable, specific)

    Tone: clear, concise, actionable. Avoid jargon. Write in plain English.
    Format: Markdown with headers. Be specific — reference actual test names and error messages.
    Maximum length: 400 words.
""").strip()

# ---------------------------------------------------------------------------
# HTML report template (inline Jinja2)
# ---------------------------------------------------------------------------

_HTML_TEMPLATE: Final[str] = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Validrix Web Agent Report — {{ result.url }}</title>
<style>
  :root {
    --pass:    #22c55e;
    --fail:    #ef4444;
    --skip:    #f59e0b;
    --neutral: #6b7280;
    --bg:      #0f172a;
    --surface: #1e293b;
    --border:  #334155;
    --text:    #e2e8f0;
    --muted:   #94a3b8;
    --accent:  #6366f1;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg); color: var(--text);
    line-height: 1.6; padding: 2rem;
  }
  h1 { font-size: 1.75rem; font-weight: 700; margin-bottom: .25rem; }
  h2 { font-size: 1.2rem; font-weight: 600; margin: 1.5rem 0 .75rem; color: var(--accent); }
  h3 { font-size: 1rem; font-weight: 600; margin-bottom: .5rem; }
  a  { color: var(--accent); }
  code, pre { font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: .82rem; }
  pre { background: #0d1117; border: 1px solid var(--border); border-radius: .5rem;
        padding: 1rem; overflow-x: auto; white-space: pre-wrap; word-break: break-word; }

  .header    { border-bottom: 1px solid var(--border); padding-bottom: 1.5rem; margin-bottom: 2rem; }
  .subtitle  { color: var(--muted); font-size: .9rem; margin-top: .25rem; }
  .meta      { display: flex; gap: 1.5rem; margin-top: .75rem; flex-wrap: wrap; }
  .meta span { font-size: .82rem; color: var(--muted); }
  .meta b    { color: var(--text); }

  /* Dashboard cards */
  .dashboard { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
               gap: 1rem; margin-bottom: 2rem; }
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: .75rem;
          padding: 1.25rem 1rem; text-align: center; }
  .card .value { font-size: 2.25rem; font-weight: 700; }
  .card .label { font-size: .75rem; color: var(--muted); text-transform: uppercase;
                 letter-spacing: .05em; margin-top: .25rem; }
  .card.pass   .value { color: var(--pass); }
  .card.fail   .value { color: var(--fail); }
  .card.skip   .value { color: var(--skip); }
  .card.total  .value { color: var(--accent); }
  .card.duration .value { font-size: 1.5rem; color: var(--text); }

  /* Health badge */
  .badge { display: inline-block; padding: .25rem .75rem; border-radius: 999px;
           font-size: .75rem; font-weight: 600; text-transform: uppercase; letter-spacing: .05em; }
  .badge.healthy  { background: #166534; color: #86efac; }
  .badge.degraded { background: #78350f; color: #fde68a; }
  .badge.critical { background: #7f1d1d; color: #fca5a5; }
  .badge.untested { background: #1e293b; color: #94a3b8; }

  /* AI summary */
  .ai-summary { background: var(--surface); border: 1px solid var(--border); border-left: 4px solid var(--accent);
                border-radius: .75rem; padding: 1.25rem 1.5rem; margin-bottom: 2rem; }
  .ai-summary h2 { margin-top: 0; }
  .ai-summary p, .ai-summary li { margin-bottom: .5rem; color: var(--muted); }
  .ai-summary strong { color: var(--text); }
  .ai-summary h3, .ai-summary h4 { color: var(--text); margin: .75rem 0 .4rem; }

  /* Test results */
  .test-list { display: flex; flex-direction: column; gap: .75rem; }
  .test-item { background: var(--surface); border: 1px solid var(--border); border-radius: .75rem;
               overflow: hidden; }
  .test-header { display: flex; align-items: center; gap: .75rem; padding: .875rem 1.25rem;
                 cursor: pointer; user-select: none; }
  .test-header:hover { background: rgba(255,255,255,.03); }
  .status-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
  .status-dot.passed  { background: var(--pass); }
  .status-dot.failed  { background: var(--fail); }
  .status-dot.skipped { background: var(--skip); }
  .status-dot.error   { background: var(--fail); }
  .test-name  { flex: 1; font-size: .875rem; font-family: monospace; color: var(--text); }
  .test-duration { font-size: .75rem; color: var(--muted); }
  .chevron    { font-size: .7rem; color: var(--muted); transition: transform .2s; }
  .test-body  { display: none; padding: 0 1.25rem 1.25rem; border-top: 1px solid var(--border); }
  .test-item.open .test-body { display: block; }
  .test-item.open .chevron  { transform: rotate(90deg); }
  .error-msg  { color: var(--fail); font-size: .85rem; margin-bottom: .75rem;
                background: rgba(239,68,68,.08); border-radius: .5rem; padding: .75rem 1rem; }
  .screenshot { margin-top: .75rem; }
  .screenshot img { max-width: 100%; border-radius: .5rem; border: 1px solid var(--border); }

  /* Generated code block */
  .code-section { margin-top: 2rem; }
  .code-section summary { cursor: pointer; color: var(--accent); font-size: .875rem;
                           margin-bottom: .5rem; }

  /* Footer */
  footer { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--border);
           font-size: .75rem; color: var(--muted); text-align: center; }
</style>
</head>
<body>

<div class="header">
  <h1>Validrix Web Agent Report
    <span class="badge {{ result.health_label }}">{{ result.health_label }}</span>
  </h1>
  <p class="subtitle">{{ result.url }}</p>
  <div class="meta">
    <span><b>Prompt:</b> {{ result.prompt }}</span>
    <span><b>Generated:</b> {{ result.timestamp.strftime('%Y-%m-%d %H:%M UTC') }}</span>
    <span><b>Duration:</b> {{ "%.1f"|format(result.duration) }}s</span>
  </div>
</div>

<!-- Dashboard -->
<div class="dashboard">
  <div class="card total">
    <div class="value">{{ result.total_tests }}</div>
    <div class="label">Total Tests</div>
  </div>
  <div class="card pass">
    <div class="value">{{ result.passed }}</div>
    <div class="label">Passed</div>
  </div>
  <div class="card fail">
    <div class="value">{{ result.failed }}</div>
    <div class="label">Failed</div>
  </div>
  <div class="card skip">
    <div class="value">{{ result.skipped }}</div>
    <div class="label">Skipped</div>
  </div>
  <div class="card duration">
    <div class="value">{{ "%.0f"|format(result.pass_rate * 100) }}%</div>
    <div class="label">Pass Rate</div>
  </div>
</div>

<!-- AI Summary -->
<div class="ai-summary">
  <h2>⚡ AI Executive Summary</h2>
  {{ ai_summary_html }}
</div>

<!-- Per-test results -->
<h2>Test Results</h2>
<div class="test-list">
{% for test in result.tests %}
  <div class="test-item {{ 'open' if test.status in ('failed', 'error') else '' }}" onclick="toggle(this)">
    <div class="test-header">
      <div class="status-dot {{ test.status }}"></div>
      <span class="test-name">{{ test.test_name.split('::')[-1] }}</span>
      <span class="test-duration">{{ "%.2f"|format(test.duration) }}s</span>
      <span class="chevron">▶</span>
    </div>
    <div class="test-body">
      <p style="font-size:.8rem;color:var(--muted);margin-bottom:.5rem">{{ test.test_name }}</p>
      {% if test.error_message %}
        <div class="error-msg">{{ test.error_message }}</div>
      {% endif %}
      {% if test.traceback %}
        <pre>{{ test.traceback }}</pre>
      {% endif %}
      {% if test.screenshot_path %}
        <div class="screenshot">
          <p style="font-size:.75rem;color:var(--muted);margin-bottom:.4rem">Screenshot on failure:</p>
          <img src="file://{{ test.screenshot_path }}" alt="Failure screenshot">
        </div>
      {% endif %}
      {% if not test.error_message and not test.traceback %}
        <p style="color:var(--pass);font-size:.875rem">✓ Test passed</p>
      {% endif %}
    </div>
  </div>
{% else %}
  <p style="color:var(--muted)">No tests were executed.</p>
{% endfor %}
</div>

<!-- Generated code -->
<div class="code-section">
  <details>
    <summary>View generated test code ({{ result.generated_code.splitlines()|length }} lines)</summary>
    <pre>{{ result.generated_code }}</pre>
  </details>
</div>

{% if result.crawl_result %}
<div class="code-section" style="margin-top:1rem">
  <details>
    <summary>View crawl data ({{ result.crawl_result.buttons|length }} buttons,
     {{ result.crawl_result.forms|length }} forms,
     {{ result.crawl_result.links|length }} links)</summary>
    <pre>{{ crawl_json }}</pre>
  </details>
</div>
{% endif %}

<footer>
  Generated by <strong>Validrix Web Agent</strong> &mdash;
  <a href="https://github.com/mehedi-kme08/validrix" target="_blank">github.com/mehedi-kme08/validrix</a>
</footer>

<script>
function toggle(el) { el.classList.toggle('open'); }
// Auto-open failed tests on load
document.querySelectorAll('.test-item').forEach(el => {
  const dot = el.querySelector('.status-dot');
  if (dot && (dot.classList.contains('failed') || dot.classList.contains('error'))) {
    el.classList.add('open');
  }
});
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Markdown → safe HTML conversion (no heavy dep like mistune)
# ---------------------------------------------------------------------------


def _md_to_html(md: str) -> str:
    """
    Convert a small subset of Markdown to HTML for the AI summary section.

    We handle only the constructs Claude actually emits (headers, bold, lists)
    without pulling in a full Markdown library. This keeps dependencies lean.
    """
    import re

    lines = md.splitlines()
    html_lines: list[str] = []
    in_list = False

    for line in lines:
        # h3/h4 headers
        if line.startswith("#### "):
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append(f"<h4>{line[5:]}</h4>")
        elif line.startswith("### "):
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append(f"<h3>{line[4:]}</h3>")
        elif line.startswith("## "):
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append(f"<h3>{line[3:]}</h3>")
        elif line.startswith("# "):
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append(f"<h3>{line[2:]}</h3>")
        # Bullet list items
        elif re.match(r"^[-*] ", line):
            if not in_list:
                html_lines.append("<ul>")
                in_list = True
            content = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", line[2:])
            html_lines.append(f"<li>{content}</li>")
        # Numbered list items
        elif re.match(r"^\d+\.\s", line):
            if not in_list:
                html_lines.append("<ul>")
                in_list = True
            content = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", re.sub(r"^\d+\.\s", "", line))
            html_lines.append(f"<li>{content}</li>")
        # Empty line
        elif not line.strip():
            if in_list:
                html_lines.append("</ul>")
                in_list = False
        # Regular paragraph
        else:
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            content = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", line)
            html_lines.append(f"<p>{content}</p>")

    if in_list:
        html_lines.append("</ul>")

    return "\n".join(html_lines)


# ---------------------------------------------------------------------------
# AI interaction helpers (mirror of test_generator.py approach)
# ---------------------------------------------------------------------------


def _call_anthropic(system: str, user: str, config: AIConfig) -> str:
    client = anthropic.Anthropic(api_key=config.anthropic_api_key, timeout=config.timeout_seconds)
    msg = client.messages.create(
        model=config.model,
        max_tokens=config.max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return msg.content[0].text  # type: ignore[union-attr]


def _call_openai(system: str, user: str, config: AIConfig) -> str:
    client = openai.OpenAI(api_key=config.openai_api_key, timeout=config.timeout_seconds)
    response = client.chat.completions.create(
        model=config.model,
        max_tokens=config.max_tokens,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return response.choices[0].message.content or ""


def _build_summary_prompt(result: TestSuiteResult) -> str:
    """Serialise the suite result into a compact prompt."""
    lines = [
        f"URL tested: {result.url}",
        f"User's testing goal: {result.prompt}",
        f"Total tests: {result.total_tests}",
        f"Passed: {result.passed} | Failed: {result.failed} | Skipped: {result.skipped}",
        f"Pass rate: {result.pass_rate:.0%}",
        "",
        "=== FAILED TESTS ===",
    ]
    failed = [t for t in result.tests if t.status in ("failed", "error")]
    if failed:
        for t in failed:
            lines.append(f"\nTest: {t.test_name}")
            if t.error_message:
                lines.append(f"Error: {t.error_message}")
            if t.traceback:
                lines.append("Traceback (last 10 lines):\n" + "\n".join((t.traceback or "").splitlines()[-10:]))
    else:
        lines.append("None — all tests passed!")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class WebReporter:
    """
    Generates an AI executive summary and an HTML report from a TestSuiteResult.

    Example::

        reporter = WebReporter()
        report_path = reporter.generate(result, report_dir=Path("validrix_reports"))
        print(f"Report saved to: {report_path}")
    """

    def __init__(self, ai_config: AIConfig | None = None) -> None:
        cfg = ConfigManager.load()
        self._ai_config = ai_config or cfg.ai

    def generate(
        self,
        result: TestSuiteResult,
        report_dir: Path | None = None,
    ) -> Path:
        """
        Attach an AI summary to ``result`` and write an HTML report.

        Args:
            result:     The TestSuiteResult from TestExecutor.run().
            report_dir: Directory where report.html will be written.
                        Defaults to ``./validrix_reports``.

        Returns:
            Path to the generated report.html file.
        """
        report_dir = report_dir or Path("validrix_reports")
        report_dir.mkdir(parents=True, exist_ok=True)

        # Generate AI summary (fail gracefully — report still renders without it)
        ai_summary = self._generate_ai_summary(result)

        # Mutate the result in place so the caller gets the full object back
        result.ai_summary = ai_summary

        report_path = self._render_html(result, report_dir)
        result.report_path = str(report_path)

        self._write_json(result, report_dir)

        logger.info("Report written to: %s", report_path)
        return report_path

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _generate_ai_summary(self, result: TestSuiteResult) -> str:
        api_key = self._ai_config.anthropic_api_key or self._ai_config.openai_api_key
        if not api_key:
            logger.warning("WebReporter: no AI API key — skipping AI summary.")
            return "_No AI summary generated (API key not configured)._"

        logger.info(
            "Generating AI summary with %s for %s",
            self._ai_config.provider,
            result.url,
        )
        start = time.monotonic()
        try:
            user_prompt = _build_summary_prompt(result)
            if self._ai_config.provider == "anthropic":
                summary = _call_anthropic(_SUMMARY_SYSTEM_PROMPT, user_prompt, self._ai_config)
            else:
                summary = _call_openai(_SUMMARY_SYSTEM_PROMPT, user_prompt, self._ai_config)
            logger.info("AI summary generated in %.1fs", time.monotonic() - start)
            return summary
        except Exception as exc:
            logger.error("AI summary generation failed: %s", exc)
            return f"_AI summary unavailable: {exc}_"

    def _render_html(self, result: TestSuiteResult, report_dir: Path) -> Path:
        """Render the Jinja2 template to report.html."""
        env = Environment(autoescape=select_autoescape(["html"]))

        # We disable autoescape for the AI summary because we convert it to
        # HTML ourselves via _md_to_html — the output is already escaped there.
        env.globals["ai_summary_html"] = _md_to_html(result.ai_summary)

        template = env.from_string(_HTML_TEMPLATE)

        crawl_json = ""
        if result.crawl_result:
            crawl_json = json.dumps(
                result.crawl_result.model_dump(exclude={"visible_text_sample"}),
                indent=2,
                default=str,
            )

        html = template.render(
            result=result,
            ai_summary_html=_md_to_html(result.ai_summary),
            crawl_json=crawl_json,
        )

        report_path = report_dir / "report.html"
        report_path.write_text(html, encoding="utf-8")
        return report_path

    @staticmethod
    def _write_json(result: TestSuiteResult, report_dir: Path) -> None:
        """Write a machine-readable JSON summary alongside the HTML report."""
        json_path = report_dir / "report.json"
        json_path.write_text(
            json.dumps(result.model_dump(), indent=2, default=str),
            encoding="utf-8",
        )
        logger.info("JSON report written to: %s", json_path)
