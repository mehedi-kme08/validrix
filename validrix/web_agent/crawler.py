"""
crawler.py — Playwright-based web crawler for the web_agent pipeline.

Design decision: Playwright over requests/BeautifulSoup.
  WHY: Modern websites are SPAs that render their DOM via JavaScript.
       requests + BS4 would see the pre-JS shell, missing most buttons,
       forms, and dynamic nav elements. Playwright drives a real Chromium
       instance, waits for networkidle, and inspects the fully rendered DOM.

  Alternatives considered:
    - Selenium: slower, heavier, requires separate driver binary management.
      Playwright's auto-install and async-first design win here.
    - Scrapy: excellent for link-following crawls, but its reactor conflicts
      with Playwright's event loop. Overkill for single-page analysis.
    - httpx + cssselect: fast but static-only; fails on SPAs.

  Tradeoffs:
    - Playwright adds ~100 MB of Chromium binaries. Acceptable because
      Playwright is already a core Validrix dependency for self_healing.py.
    - Headless Chromium is slower than raw HTTP (~2-5 s per page). We set
      a 30-second hard timeout to bound worst-case latency.
    - We use sync_api (not async) to match the rest of the codebase and
      allow straightforward pytest fixture integration.

  Error model:
    CrawlError subclasses let callers distinguish recoverable (timeout,
    SSL) from unrecoverable (invalid URL) failures without parsing strings.
"""

from __future__ import annotations

import logging
import time
from urllib.parse import urlparse

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Page, sync_playwright

from validrix.web_agent.models import (
    ButtonElement,
    CrawlResult,
    FormElement,
    FormField,
    ImageElement,
    LinkElement,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

_DEFAULT_TIMEOUT_MS: int = 30_000  # 30 seconds


class CrawlError(RuntimeError):
    """Base for all crawler failures."""


class InvalidURLError(CrawlError):
    """The provided URL is not a valid http/https URL."""


class PageNotFoundError(CrawlError):
    """The server returned a 4xx response."""


class CrawlTimeoutError(CrawlError):
    """Page did not reach networkidle within the configured timeout."""


class SSLError(CrawlError):
    """TLS/SSL certificate error encountered while loading the page."""


# ---------------------------------------------------------------------------
# Internal helpers — each maps one DOM query to one model
# ---------------------------------------------------------------------------


def _extract_buttons(page: Page, base_url: str) -> list[ButtonElement]:  # noqa: ARG001
    """Return all interactive buttons visible on the page."""
    buttons: list[ButtonElement] = []

    # Playwright's locator API finds elements matching any of these selectors.
    # We cast via evaluate so we get plain dicts, avoiding CDP round-trips.
    raw: list[dict[str, object]] = page.evaluate(
        """() => {
            const results = [];
            const selectors = [
                'button',
                'input[type="submit"]',
                'input[type="button"]',
                'input[type="reset"]',
                '[role="button"]',
                'a.btn, a.button, a[class*="btn"]',
            ];
            const seen = new Set();
            for (const sel of selectors) {
                for (const el of document.querySelectorAll(sel)) {
                    const text = (el.innerText || el.value || '').trim();
                    const aria = el.getAttribute('aria-label') || null;
                    const tag = el.tagName.toLowerCase();
                    const key = text + '|' + (aria || '') + '|' + tag;
                    if (seen.has(key)) continue;
                    seen.add(key);
                    // Build a simple but unique CSS selector for the element
                    let selector = tag;
                    if (el.id) selector = '#' + el.id;
                    else if (el.name) selector = tag + '[name="' + el.name + '"]';
                    else if (aria) selector = tag + '[aria-label="' + aria + '"]';
                    results.push({
                        text: text,
                        aria_label: aria,
                        element_type: tag,
                        selector: selector,
                        is_visible: el.offsetParent !== null
                    });
                }
            }
            return results;
        }"""
    )

    for item in raw:
        try:
            buttons.append(ButtonElement(**item))  # type: ignore[arg-type]
        except Exception as exc:
            logger.debug("Skipping malformed button element: %s — %s", item, exc)

    return buttons


def _extract_forms(page: Page) -> list[FormElement]:
    """Return all <form> elements with their fields."""
    raw: list[dict[str, object]] = page.evaluate(
        """() => {
            return Array.from(document.querySelectorAll('form')).map((form, fi) => {
                const fields = Array.from(
                    form.querySelectorAll('input, select, textarea')
                ).map(el => {
                    const label = document.querySelector('label[for="' + el.id + '"]');
                    return {
                        name: el.name || el.id || ('field_' + Math.random().toString(36).slice(2,6)),
                        field_type: el.type || el.tagName.toLowerCase(),
                        placeholder: el.placeholder || null,
                        is_required: el.required || false,
                        label: label ? label.innerText.trim() : null,
                    };
                }).filter(f => f.field_type !== 'hidden' && f.field_type !== 'submit');

                const submitBtn = form.querySelector('[type="submit"], button:not([type="button"])');
                let selector = 'form';
                if (form.id) selector = '#' + form.id;
                else if (form.name) selector = 'form[name="' + form.name + '"]';
                else selector = 'form:nth-of-type(' + (fi + 1) + ')';

                return {
                    action: form.action || null,
                    method: (form.method || 'get').toLowerCase(),
                    fields: fields,
                    submit_text: submitBtn ? (submitBtn.innerText || submitBtn.value || null) : null,
                    selector: selector,
                };
            });
        }"""
    )

    forms: list[FormElement] = []
    for item in raw:
        try:
            fields = [FormField(**f) for f in (item.pop("fields", []) or [])]  # type: ignore[union-attr]
            forms.append(FormElement(fields=fields, **item))  # type: ignore[arg-type]
        except Exception as exc:
            logger.debug("Skipping malformed form element: %s — %s", item, exc)

    return forms


def _extract_links(page: Page, base_url: str) -> list[LinkElement]:
    """Return all <a> elements — de-duplicated by href."""
    parsed_base = urlparse(base_url)
    raw: list[dict[str, object]] = page.evaluate(
        """() => Array.from(document.querySelectorAll('a[href]')).map(a => ({
            text: (a.innerText || a.textContent || '').trim().slice(0, 200),
            href: a.href,
            aria_label: a.getAttribute('aria-label') || null,
        }))"""
    )

    seen_hrefs: set[str] = set()
    links: list[LinkElement] = []
    for item in raw:
        href = str(item.get("href", ""))
        if not href or href.startswith(("javascript:", "mailto:", "tel:", "#")):
            continue
        if href in seen_hrefs:
            continue
        seen_hrefs.add(href)
        is_external = urlparse(href).netloc not in ("", parsed_base.netloc)
        try:
            links.append(LinkElement(is_external=is_external, **item))  # type: ignore[arg-type]
        except Exception as exc:
            logger.debug("Skipping malformed link: %s — %s", item, exc)

    return links


def _extract_images(page: Page) -> list[ImageElement]:
    """Return all <img> elements."""
    raw: list[dict[str, object]] = page.evaluate(
        """() => Array.from(document.querySelectorAll('img')).map(img => ({
            src: img.src,
            alt: img.alt || '',
            is_decorative: img.alt === '',
        }))"""
    )
    images: list[ImageElement] = []
    for item in raw:
        try:
            images.append(ImageElement(**item))  # type: ignore[arg-type]
        except Exception as exc:
            logger.debug("Skipping malformed image: %s — %s", item, exc)
    return images


def _extract_headings(page: Page) -> list[str]:
    """Return h1–h3 text nodes in document order."""
    return page.evaluate(
        """() => Array.from(document.querySelectorAll('h1, h2, h3'))
               .map(h => h.innerText.trim())
               .filter(t => t.length > 0)"""
    )


def _extract_visible_text(page: Page, max_chars: int = 2000) -> str:
    """Return a sample of visible body text for AI context."""
    text: str = page.evaluate(
        """() => (document.body && document.body.innerText) ? document.body.innerText.trim() : ''"""
    )
    return text[:max_chars]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class WebCrawler:
    """
    Playwright-powered web crawler that produces a structured CrawlResult.

    Designed to be called once per URL per pipeline run. Internally manages
    a browser context lifetime (open → crawl → close) to avoid resource leaks.

    Example::

        crawler = WebCrawler(timeout_ms=20_000, headless=True)
        result = crawler.crawl("https://example.com")
        if result.succeeded:
            print(result.title, len(result.buttons), "buttons found")
    """

    def __init__(
        self,
        timeout_ms: int = _DEFAULT_TIMEOUT_MS,
        headless: bool = True,
    ) -> None:
        """
        Args:
            timeout_ms: Hard timeout for page load (milliseconds).
            headless:   Run Chromium without a visible window.
        """
        self._timeout_ms = timeout_ms
        self._headless = headless

    def crawl(self, url: str) -> CrawlResult:
        """
        Navigate to ``url``, extract structural data, and return a CrawlResult.

        Never raises — all errors are captured into CrawlResult.error so that
        the pipeline can continue and the reporter can describe what went wrong.

        Args:
            url: The fully-qualified https/http URL to crawl.

        Returns:
            CrawlResult — check ``.succeeded`` before using other fields.
        """
        if not url.startswith(("http://", "https://")):
            logger.error("Invalid URL (missing scheme): %r", url)
            return CrawlResult(url=url, error=f"Invalid URL — must start with http:// or https://: {url!r}")

        logger.info("Crawling: %s", url)
        start = time.monotonic()

        try:
            result = self._do_crawl(url)
        except Exception as exc:
            elapsed = time.monotonic() - start
            error_msg = self._classify_error(exc)
            logger.error("Crawl failed for %s after %.1fs: %s", url, elapsed, error_msg)
            return CrawlResult(url=url, crawl_duration=elapsed, error=error_msg)

        result.crawl_duration = time.monotonic() - start
        logger.info(
            "Crawl complete: %s — %.1fs, %d buttons, %d forms, %d links",
            url,
            result.crawl_duration,
            len(result.buttons),
            len(result.forms),
            len(result.links),
        )
        return result

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _do_crawl(self, url: str) -> CrawlResult:
        """Core crawl logic — may raise; errors are caught by crawl()."""
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=self._headless)
            context = browser.new_context(
                # Mimic a realistic desktop viewport so responsive sites render fully
                viewport={"width": 1280, "height": 800},
                user_agent=(
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 Validrix/0.1"
                ),
                ignore_https_errors=False,  # SSL errors are surfaced, not swallowed
            )
            page = context.new_page()

            try:
                response = page.goto(
                    url,
                    wait_until="networkidle",
                    timeout=self._timeout_ms,
                )

                # Surface 4xx/5xx as structured errors rather than empty crawls
                if response and response.status >= 400:
                    raise PageNotFoundError(f"HTTP {response.status} from {url}")

                return CrawlResult(
                    url=url,
                    title=page.title(),
                    meta_description=self._get_meta_description(page),
                    headings=_extract_headings(page),
                    buttons=_extract_buttons(page, url),
                    forms=_extract_forms(page),
                    links=_extract_links(page, url),
                    images=_extract_images(page),
                    visible_text_sample=_extract_visible_text(page),
                )
            finally:
                # Always close to avoid Chromium zombie processes
                context.close()
                browser.close()

    @staticmethod
    def _get_meta_description(page: Page) -> str:
        return page.evaluate(
            """() => {
                const el = document.querySelector('meta[name="description"]');
                return el ? el.getAttribute('content') || '' : '';
            }"""
        )

    @staticmethod
    def _classify_error(exc: Exception) -> str:
        """Map raw exceptions to user-friendly error strings."""
        msg = str(exc)
        if "Timeout" in type(exc).__name__ or "timeout" in msg.lower():
            return f"Page load timed out: {msg}"
        if "SSL" in msg or "certificate" in msg.lower() or "ERR_CERT" in msg:
            return f"SSL/TLS error: {msg}"
        if "ERR_NAME_NOT_RESOLVED" in msg or "Name or service not known" in msg:
            return f"DNS resolution failed — check the URL is reachable: {msg}"
        if isinstance(exc, (PageNotFoundError, InvalidURLError, CrawlTimeoutError, SSLError)):
            return msg
        if isinstance(exc, PlaywrightError):
            return f"Browser error: {msg}"
        return f"Unexpected crawl error ({type(exc).__name__}): {msg}"
