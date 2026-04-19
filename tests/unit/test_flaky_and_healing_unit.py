from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from validrix.core.config_manager import FrameworkConfig
from validrix.plugins.flaky_detector import (
    FlakinessLabel,
    FlakinessMetric,
    FlakyDetectorPlugin,
    RunResult,
)
from validrix.plugins.self_healing import (
    AriaLabelStrategy,
    CSSRebuildStrategy,
    HealingEvent,
    HealingHistory,
    HealingPage,
    NearbyElementStrategy,
    SelfHealingPlugin,
    TextContentStrategy,
)


def _capture_writes(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    writes: dict[str, str] = {}

    monkeypatch.setattr(Path, "mkdir", lambda self, parents=False, exist_ok=False: None)
    monkeypatch.setattr(
        Path,
        "write_text",
        lambda self, content, encoding="utf-8": writes.__setitem__(str(self), content) or len(content),
    )
    return writes


class _FakeElement:
    def __init__(
        self,
        aria_label: str | None = None,
        text: str = "",
        parent: str | None = None,
        tag: str | None = None,
        rebuilt: str | None = None,
    ) -> None:
        self._aria_label = aria_label
        self._text = text
        self._parent = parent
        self._tag = tag
        self._rebuilt = rebuilt

    def get_attribute(self, name: str) -> str | None:
        return self._aria_label if name == "aria-label" else None

    def inner_text(self) -> str:
        return self._text

    def evaluate(self, expression: str) -> str | None:
        if "parentElement" in expression:
            return self._parent
        if "tagName" in expression and "classList" not in expression:
            return self._tag
        if "classList" in expression:
            return self._rebuilt
        return None


class _FakeLocator:
    def __init__(self, should_resolve: bool) -> None:
        self.should_resolve = should_resolve

    def wait_for(self, *, state: str, timeout: int) -> None:
        if not self.should_resolve:
            raise RuntimeError("not found")


class _FakePage:
    def __init__(self, element: _FakeElement | None, resolvable: dict[str, bool], extra: object = "value") -> None:
        self.element = element
        self.resolvable = resolvable
        self.extra = extra

    def query_selector(self, selector: str) -> _FakeElement | None:
        return self.element

    def locator(self, selector: str, **kwargs: object) -> _FakeLocator:
        return _FakeLocator(self.resolvable.get(selector, False))


def test_flakiness_metric_covers_all_labels_and_to_dict() -> None:
    stable = FlakinessMetric.compute("x::stable", [RunResult(1, True, 0.1)])
    failing = FlakinessMetric.compute("x::failing", [RunResult(1, False, 0.1)])
    flaky = FlakinessMetric.compute(
        "x::flaky",
        [RunResult(1, True, 0.1), RunResult(2, False, 0.2)],
    )

    assert stable.label == FlakinessLabel.STABLE
    assert failing.label == FlakinessLabel.FAILING
    assert flaky.label == FlakinessLabel.FLAKY
    assert flaky.to_dict()["label"] == "FLAKY"


def test_flaky_detector_addoption_configure_and_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = FrameworkConfig()
    cfg.flaky.report_path = Path("reports/flaky.json")
    monkeypatch.setattr("validrix.plugins.flaky_detector.ConfigManager.load", lambda: cfg)
    plugin = FlakyDetectorPlugin()
    seen: list[tuple[str, object]] = []

    class Group:
        def addoption(self, *args: object, **kwargs: object) -> None:
            seen.append(("addoption", args))

    class Parser:
        def getgroup(self, name: str, description: str) -> Group:
            return Group()

    class Config:
        def __init__(self, detect: bool, runs: int | None) -> None:
            self.detect = detect
            self.runs = runs
            self.lines: list[tuple[str, str]] = []

        def addinivalue_line(self, name: str, value: str) -> None:
            self.lines.append((name, value))

        def getoption(self, name: str, default: object = None) -> object:
            if name == "--detect-flaky":
                return self.detect
            if name == "--flaky-runs":
                return self.runs
            return default

    plugin.pytest_addoption(Parser())
    plugin.pytest_configure(Config(True, 5))
    assert seen
    assert plugin._detect_all is True
    assert plugin._flaky_cfg.runs == 5

    item = SimpleNamespace(nodeid="tests/test_one.py::test_it", get_closest_marker=lambda name: True)
    monkeypatch.setattr(
        plugin,
        "_run_n_times",
        lambda item, n: [RunResult(1, True, 0.1), RunResult(2, False, 0.2)],
    )
    assert plugin.pytest_runtest_protocol(item, None) is True
    assert plugin._run_counts[item.nodeid][0].passed is True


def test_flaky_detector_configure_without_override(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = FrameworkConfig()
    monkeypatch.setattr("validrix.plugins.flaky_detector.ConfigManager.load", lambda: cfg)
    plugin = FlakyDetectorPlugin()

    class Config:
        def addinivalue_line(self, name: str, value: str) -> None:
            pass

        def getoption(self, name: str, default: object = None) -> object:
            return False if name == "--detect-flaky" else None

    plugin.pytest_configure(Config())

    assert plugin._flaky_cfg.runs == cfg.flaky.runs


def test_flaky_detector_protocol_returns_none_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = FrameworkConfig()
    cfg.flaky.enabled = False
    monkeypatch.setattr("validrix.plugins.flaky_detector.ConfigManager.load", lambda: cfg)
    plugin = FlakyDetectorPlugin()
    item = SimpleNamespace(nodeid="x", get_closest_marker=lambda name: None)

    assert plugin.pytest_runtest_protocol(item, None) is None


def test_flaky_detector_protocol_logs_non_flaky_result(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = FrameworkConfig()
    monkeypatch.setattr("validrix.plugins.flaky_detector.ConfigManager.load", lambda: cfg)
    plugin = FlakyDetectorPlugin()
    plugin._detect_all = True
    item = SimpleNamespace(nodeid="tests/test_one.py::test_it", get_closest_marker=lambda name: None)
    monkeypatch.setattr(plugin, "_run_n_times", lambda item, n: [RunResult(1, True, 0.1)])

    assert plugin.pytest_runtest_protocol(item, None) is True


def test_flaky_detector_run_n_times_and_execute_item(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = FrameworkConfig()
    monkeypatch.setattr("validrix.plugins.flaky_detector.ConfigManager.load", lambda: cfg)
    plugin = FlakyDetectorPlugin()
    reports = [
        SimpleNamespace(when="setup", passed=True, failed=False, duration=0.0),
        SimpleNamespace(when="call", passed=False, failed=True, duration=0.3, longreprtext="trace", longrepr="trace"),
        SimpleNamespace(when="teardown", passed=True, failed=False, duration=0.0),
    ]
    monkeypatch.setattr(plugin, "_execute_item", lambda item: iter(reports))

    result = plugin._run_n_times(SimpleNamespace(nodeid="node"), 1)

    assert result[0].passed is False
    assert result[0].error_message == "trace"

    monkeypatch.setattr(
        "validrix.plugins.flaky_detector.runtestprotocol",
        lambda item, log: [SimpleNamespace(when="call", passed=True, failed=False, duration=0.1)],
    )
    assert list(FlakyDetectorPlugin._execute_item(SimpleNamespace()))[0].passed is True


def test_flaky_detector_run_n_times_with_successful_call(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = FrameworkConfig()
    monkeypatch.setattr("validrix.plugins.flaky_detector.ConfigManager.load", lambda: cfg)
    plugin = FlakyDetectorPlugin()
    monkeypatch.setattr(
        plugin,
        "_execute_item",
        lambda item: iter([SimpleNamespace(when="call", passed=True, failed=False, duration=0.2)]),
    )

    result = plugin._run_n_times(SimpleNamespace(nodeid="node"), 1)

    assert result[0].passed is True
    assert result[0].error_message == ""


def test_flaky_detector_writes_reports(monkeypatch: pytest.MonkeyPatch) -> None:
    writes = _capture_writes(monkeypatch)
    tmp_path = Path("reports")
    cfg = FrameworkConfig()
    cfg.flaky.report_path = tmp_path / "report.json"
    monkeypatch.setattr("validrix.plugins.flaky_detector.ConfigManager.load", lambda: cfg)
    plugin = FlakyDetectorPlugin()
    plugin._run_counts = {
        "tests/test_demo.py::test_a": [RunResult(1, True, 0.1)],
        "tests/test_demo.py::test_b": [RunResult(1, False, 0.1)],
    }

    plugin.pytest_sessionfinish(SimpleNamespace(), 0)

    html = writes[str(tmp_path / "report.html")]
    assert "Validrix Flaky Test Report" in html
    assert "test_a" in html


def test_flaky_detector_sessionfinish_no_results(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = FrameworkConfig()
    monkeypatch.setattr("validrix.plugins.flaky_detector.ConfigManager.load", lambda: cfg)
    plugin = FlakyDetectorPlugin()

    plugin.pytest_sessionfinish(SimpleNamespace(), 0)


def test_healing_strategies_and_history() -> None:
    element = _FakeElement(
        aria_label="Submit",
        text="Click me",
        parent="#form",
        tag="button",
        rebuilt='button.primary[type="submit"]',
    )
    page = _FakePage(element, {})

    assert AriaLabelStrategy().build_selector("#btn", page) == "[aria-label='Submit']"
    assert TextContentStrategy().build_selector("#btn", page) == "text=Click me"
    assert NearbyElementStrategy().build_selector("#btn", page) == "#form button"
    assert CSSRebuildStrategy().build_selector("#btn", page) == 'button.primary[type="submit"]'

    history = HealingHistory(total_healed=1, events=[HealingEvent("t", "#a", "css", ".b")])
    assert history.to_dict()["total_healed"] == 1


def test_healing_page_locator_success_heal_and_fallback() -> None:
    events: list[HealingEvent] = []
    config = FrameworkConfig().healing

    native_page = _FakePage(None, {"#ok": True})
    healing_page = HealingPage(native_page, "test", [], config, events)
    assert healing_page.locator("#ok").should_resolve is True

    strategy = SimpleNamespace(name="aria", build_selector=lambda original, page: "#healed")
    heal_page = _FakePage(None, {"#missing": False, "#healed": True}, extra="delegated")
    healing_page = HealingPage(heal_page, "test", [strategy], config, events)
    assert healing_page.locator("#missing").should_resolve is True
    assert events[0].successful_strategy == "aria"
    assert healing_page.extra == "delegated"

    bad_strategy = SimpleNamespace(name="none", build_selector=lambda original, page: "#still-missing")
    fallback_page = _FakePage(None, {"#missing": False, "#still-missing": False}, extra="x")
    fallback = HealingPage(fallback_page, "test", [bad_strategy], config, [])
    assert fallback.locator("#missing").should_resolve is False


def test_self_healing_plugin_fixture_and_sessionfinish(monkeypatch: pytest.MonkeyPatch) -> None:
    writes = _capture_writes(monkeypatch)
    tmp_path = Path("reports")
    cfg = FrameworkConfig()
    cfg.healing.history_path = tmp_path / "healing.json"
    monkeypatch.setattr("validrix.plugins.self_healing.ConfigManager.load", lambda: cfg)
    plugin = SelfHealingPlugin()

    config = SimpleNamespace(lines=[])
    config.addinivalue_line = lambda name, value: config.lines.append((name, value))
    plugin.pytest_configure(config)
    assert config.lines

    page = _FakePage(None, {"#ok": True})
    request = SimpleNamespace(node=SimpleNamespace(nodeid="tests/test_demo.py::test_case"))
    wrapped = plugin.healing_page_fixture.__wrapped__(plugin, page, request)
    assert isinstance(wrapped, HealingPage)

    plugin._healing_config.enabled = False
    assert plugin.healing_page_fixture.__wrapped__(plugin, page, request) is page

    plugin._events = [HealingEvent("t", "#a", "css", ".b")]
    plugin.pytest_sessionfinish(SimpleNamespace(), 0)

    data = json.loads(writes[str(tmp_path / "healing.json")])
    assert data["total_healed"] == 1


def test_self_healing_strategy_exceptions_and_empty_session(monkeypatch: pytest.MonkeyPatch) -> None:
    class ExplodingPage:
        def query_selector(self, selector: str) -> None:
            raise RuntimeError("boom")

        def locator(self, selector: str, **kwargs: object) -> _FakeLocator:
            return _FakeLocator(False)

    page = ExplodingPage()

    assert AriaLabelStrategy().build_selector("#x", page) is None
    assert TextContentStrategy().build_selector("#x", page) is None
    assert NearbyElementStrategy().build_selector("#x", page) is None
    assert CSSRebuildStrategy().build_selector("#x", page) is None

    cfg = FrameworkConfig()
    monkeypatch.setattr("validrix.plugins.self_healing.ConfigManager.load", lambda: cfg)
    plugin = SelfHealingPlugin()
    plugin.pytest_sessionfinish(SimpleNamespace(), 0)


def test_self_healing_strategies_return_none_when_values_missing() -> None:
    element = _FakeElement(aria_label=None, text="x" * 120, parent=None, tag="button", rebuilt=None)
    page = _FakePage(element, {})

    assert AriaLabelStrategy().build_selector("#x", page) is None
    assert TextContentStrategy().build_selector("#x", page) is None
    assert NearbyElementStrategy().build_selector("#x", page) is None
    assert CSSRebuildStrategy().build_selector("#x", page) is None


def test_self_healing_strategies_return_none_when_element_missing() -> None:
    page = _FakePage(None, {})

    assert AriaLabelStrategy().build_selector("#x", page) is None
    assert TextContentStrategy().build_selector("#x", page) is None
    assert NearbyElementStrategy().build_selector("#x", page) is None
    assert CSSRebuildStrategy().build_selector("#x", page) is None


def test_healing_page_skips_empty_alternative_before_success() -> None:
    events: list[HealingEvent] = []
    config = FrameworkConfig().healing
    page = _FakePage(None, {"#healed": True})
    strategies = [
        SimpleNamespace(name="empty", build_selector=lambda original, current_page: None),
        SimpleNamespace(name="ok", build_selector=lambda original, current_page: "#healed"),
    ]

    healed = HealingPage(page, "test", strategies, config, events).locator("#missing")

    assert healed.should_resolve is True
    assert events[0].healed_selector == "#healed"
