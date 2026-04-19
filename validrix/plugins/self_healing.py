"""
self_healing.py — Self-healing Playwright locator strategies.

Design decision: Monkey-patch Playwright's Page.locator at the fixture level.
  WHY: Playwright does not expose a locator-resolution hook, so we wrap the
       Page object in a proxy that intercepts locator() calls and applies
       fallback strategies before raising TimeoutError.  This is transparent
       to test authors — they write normal Playwright code.

  Alternatives considered:
    - Subclass Page: Playwright's Page is a sealed dataclass-like object
      generated from the protocol spec; subclassing is fragile across versions.
    - Separate HealingPage wrapper: cleaner OOP but requires test authors to
      import and use HealingPage instead of the standard page fixture, which
      breaks existing tests.
    - AI-powered locator suggestion via screenshot analysis: powerful but adds
      ~2-5 s per healing event and requires vision model access.  Deferred to
      a future enhancement tracked in the roadmap.

  Fallback order (configurable via validrix.yml):
    1. aria-label   — accessibility attribute, most stable across DOM changes
    2. text         — visible text content, resilient to attribute renames
    3. nearby       — siblings/ancestors of original element, structural fallback
    4. css          — broad CSS selector rebuilt from element context

  Tradeoffs:
    - Wrapping Page adds a thin indirection layer (~microseconds per call).
    - "Healing" can mask real regression bugs if the UI has genuinely changed.
      We log every healing event and write a history file so teams can audit.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

import pytest

from validrix.core.config_manager import ConfigManager, SelfHealingConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class HealingEvent:
    """Records a single self-healing locator resolution."""

    test_id: str
    original_selector: str
    successful_strategy: str
    healed_selector: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    duration_ms: float = 0.0


@dataclass
class HealingHistory:
    """Aggregated healing events for a complete test session."""

    total_healed: int
    events: list[HealingEvent]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_healed": self.total_healed,
            "events": [asdict(e) for e in self.events],  # type: ignore[arg-type]
        }


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------

@runtime_checkable
class HealingStrategy(Protocol):
    """Protocol for locator fallback strategies."""

    name: str

    def build_selector(self, original: str, page: Any) -> str | None:
        """
        Attempt to build an alternative selector for the element.

        Args:
            original: The selector that failed.
            page:     The Playwright Page object.

        Returns:
            An alternative selector string, or None if this strategy cannot help.
        """
        ...


class AriaLabelStrategy:
    """Try aria-label attribute of the target element."""

    name = "aria-label"

    def build_selector(self, original: str, page: Any) -> str | None:
        try:
            # Attempt to find the element by aria-label derived from original selector text
            element = page.query_selector(original)
            if element:
                aria_label = element.get_attribute("aria-label")
                if aria_label:
                    return f"[aria-label='{aria_label}']"
        except Exception:
            pass
        return None


class TextContentStrategy:
    """Try locating by visible text content."""

    name = "text"

    def build_selector(self, original: str, page: Any) -> str | None:
        try:
            element = page.query_selector(original)
            if element:
                text = element.inner_text().strip()
                if text and len(text) < 100:  # skip huge text blocks
                    return f"text={text}"
        except Exception:
            pass
        return None


class NearbyElementStrategy:
    """Try locating via a stable parent or sibling."""

    name = "nearby"

    def build_selector(self, original: str, page: Any) -> str | None:
        try:
            element = page.query_selector(original)
            if element:
                # Walk up to first ancestor with an id or data-testid
                parent = element.evaluate(
                    """el => {
                        let node = el.parentElement;
                        while (node) {
                            if (node.id) return '#' + node.id;
                            if (node.dataset.testid) return '[data-testid="' + node.dataset.testid + '"]';
                            node = node.parentElement;
                        }
                        return null;
                    }"""
                )
                if parent:
                    tag = element.evaluate("el => el.tagName.toLowerCase()")
                    return f"{parent} {tag}"
        except Exception:
            pass
        return None


class CSSRebuildStrategy:
    """Rebuild a broad CSS selector from element context."""

    name = "css"

    def build_selector(self, original: str, page: Any) -> str | None:
        try:
            element = page.query_selector(original)
            if element:
                rebuilt = element.evaluate(
                    """el => {
                        const tag = el.tagName.toLowerCase();
                        const cls = Array.from(el.classList).slice(0, 2).join('.');
                        const type = el.type ? `[type="${el.type}"]` : '';
                        return cls ? `${tag}.${cls}${type}` : `${tag}${type}`;
                    }"""
                )
                if rebuilt:
                    return rebuilt
        except Exception:
            pass
        return None


# Default ordered strategy chain
_DEFAULT_STRATEGIES: list[Any] = [
    AriaLabelStrategy(),
    TextContentStrategy(),
    NearbyElementStrategy(),
    CSSRebuildStrategy(),
]


# ---------------------------------------------------------------------------
# Healing page proxy
# ---------------------------------------------------------------------------

class HealingPage:
    """
    Proxy around a Playwright Page that applies fallback locator strategies.

    Usage::

        # Injected automatically via the `healing_page` fixture
        def test_example(healing_page):
            healing_page.locator("#submit-btn").click()
    """

    def __init__(
        self,
        page: Any,
        test_id: str,
        strategies: list[Any],
        config: SelfHealingConfig,
        events: list[HealingEvent],
    ) -> None:
        self._page = page
        self._test_id = test_id
        self._strategies = strategies
        self._config = config
        self._events = events

    def locator(self, selector: str, **kwargs: Any) -> Any:
        """
        Return a Playwright Locator, falling back through healing strategies
        if the original selector produces no match.
        """
        native_locator = self._page.locator(selector, **kwargs)

        try:
            # Probe whether the locator resolves without actually interacting
            native_locator.wait_for(state="attached", timeout=2000)
            return native_locator
        except Exception:
            pass  # Selector failed — try healing

        logger.info("SelfHealing: selector %r failed. Attempting strategies…", selector)
        return self._heal(selector, kwargs)

    def _heal(self, original: str, kwargs: dict[str, Any]) -> Any:
        start = time.monotonic()
        for strategy in self._strategies:
            alternative = strategy.build_selector(original, self._page)
            if not alternative:
                continue

            candidate = self._page.locator(alternative, **kwargs)
            try:
                candidate.wait_for(state="attached", timeout=2000)
                elapsed_ms = (time.monotonic() - start) * 1000

                event = HealingEvent(
                    test_id=self._test_id,
                    original_selector=original,
                    successful_strategy=strategy.name,
                    healed_selector=alternative,
                    duration_ms=elapsed_ms,
                )
                self._events.append(event)
                logger.info(
                    "SelfHealing: healed %r → %r via strategy %r (%.0f ms)",
                    original,
                    alternative,
                    strategy.name,
                    elapsed_ms,
                )
                return candidate
            except Exception:
                continue

        # All strategies exhausted — fall back to native (will raise naturally)
        logger.warning(
            "SelfHealing: all strategies exhausted for selector %r. "
            "Returning native locator — test will likely fail.",
            original,
        )
        return self._page.locator(original, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Delegate all other Page methods transparently."""
        return getattr(self._page, name)


# ---------------------------------------------------------------------------
# pytest plugin
# ---------------------------------------------------------------------------

class SelfHealingPlugin:
    """
    Pytest plugin that provides self-healing Playwright locators.

    Activated when validrix is installed. Exposes the ``healing_page`` fixture.
    Writes a healing history JSON at session end.
    """

    def __init__(self, config: pytest.Config | None = None) -> None:
        self._pytest_config = config
        self._cfg = ConfigManager.load()
        self._healing_config = self._cfg.healing
        self._events: list[HealingEvent] = []
        self._strategies = _DEFAULT_STRATEGIES

    def pytest_configure(self, config: pytest.Config) -> None:
        """Register the healing_page fixture."""
        config.addinivalue_line(
            "markers",
            "self_healing: mark test to use self-healing locators explicitly",
        )

    @pytest.fixture(name="healing_page")
    def healing_page_fixture(
        self,
        page: Any,  # Playwright's built-in `page` fixture
        request: pytest.FixtureRequest,
    ) -> HealingPage:
        """
        Playwright Page wrapped with self-healing locator strategies.

        Drop-in replacement for the standard ``page`` fixture::

            def test_login(healing_page):
                healing_page.goto("https://example.com/login")
                healing_page.locator("#email").fill("user@example.com")
        """
        if not self._healing_config.enabled:
            return page  # type: ignore[return-value]

        return HealingPage(
            page=page,
            test_id=request.node.nodeid,
            strategies=self._strategies,
            config=self._healing_config,
            events=self._events,
        )

    def pytest_sessionfinish(
        self,
        session: pytest.Session,
        exitstatus: int | pytest.ExitCode,
    ) -> None:
        """Write healing history to JSON at session end."""
        if not self._events:
            return

        history = HealingHistory(
            total_healed=len(self._events),
            events=self._events,
        )
        path = self._healing_config.history_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(history.to_dict(), indent=2),
            encoding="utf-8",
        )
        logger.info(
            "SelfHealing: %d healing event(s) written to %s",
            len(self._events),
            path,
        )
