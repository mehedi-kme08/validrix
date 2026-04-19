from __future__ import annotations

from pathlib import Path

import pytest
from _pytest.outcomes import Failed

from validrix.core.base_test import BaseTest, _method_name
from validrix.core.config_manager import ConfigManager, FrameworkConfig, _deep_merge, _load_config


class _SampleTest(BaseTest):
    pass


_CONFIG_FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "sample_validrix.yml"


def test_deep_merge_overwrites_nested_keys() -> None:
    base = {"outer": {"keep": 1, "change": 2}, "plain": "a"}
    override = {"outer": {"change": 3, "new": 4}, "plain": "b"}

    _deep_merge(base, override)

    assert base == {"outer": {"keep": 1, "change": 3, "new": 4}, "plain": "b"}


def test_config_manager_loads_yaml_and_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    ConfigManager.reset()
    monkeypatch.setenv("VALIDRIX_RETRY__MAX_ATTEMPTS", "6")
    monkeypatch.setenv("VALIDRIX_ENVIRONMENT", "staging")

    cfg = ConfigManager.load(config_path=_CONFIG_FIXTURE)

    assert cfg.environment == "staging"
    assert cfg.retry.max_attempts == 6
    assert cfg.retry.delay_seconds == 0.25
    assert cfg.env.base_url == "https://staging.example.com"
    assert cfg.env.name == "staging"
    assert cfg.report_dir == Path("reports")


def test_config_manager_uses_defaults_without_file(monkeypatch: pytest.MonkeyPatch) -> None:
    ConfigManager.reset()
    monkeypatch.delenv("VALIDRIX_ENVIRONMENT", raising=False)
    monkeypatch.delenv("VALIDRIX_RETRY__MAX_ATTEMPTS", raising=False)
    monkeypatch.chdir(Path(__file__).parent)

    cfg = ConfigManager.load()

    assert cfg.environment == "dev"
    assert cfg.env.name == "dev"
    assert cfg.report_dir == Path("validrix_reports")


def test_config_manager_raises_for_missing_explicit_file() -> None:
    ConfigManager.reset()

    with pytest.raises(FileNotFoundError):
        ConfigManager.load(config_path=Path("tests/fixtures/missing.yml"))


def test_load_config_is_cached_and_reset_clears_cache() -> None:
    ConfigManager.reset()

    first = ConfigManager.load(config_path=_CONFIG_FIXTURE)
    second = ConfigManager.load(config_path=_CONFIG_FIXTURE)
    assert first is second

    ConfigManager.reset()
    third = ConfigManager.load(config_path=_CONFIG_FIXTURE)
    assert third is not first


def test_framework_config_coerces_report_dir_and_syncs_env_name() -> None:
    cfg = FrameworkConfig(report_dir="custom_reports", environment="prod")

    assert cfg.report_dir == Path("custom_reports")
    assert cfg.env.name == "prod"


def test_load_config_with_environment_argument_overrides_yaml_target() -> None:
    _load_config.cache_clear()

    cfg = ConfigManager.load(config_path=_CONFIG_FIXTURE, environment="staging")

    assert cfg.env.base_url == "https://staging.example.com"


def test_base_test_setup_and_teardown_use_config(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = FrameworkConfig()
    monkeypatch.setattr("validrix.core.base_test.ConfigManager.load", lambda: cfg)
    test_case = _SampleTest()

    test_case.setup_method("test_example")

    assert test_case.config is cfg
    assert test_case.retry.config.max_attempts == cfg.retry.max_attempts
    assert test_case._soft_failures == []
    assert test_case.log.name.endswith("_SampleTest")

    test_case.teardown_method("test_example")


def test_method_name_prefers_dunder_name() -> None:
    class Named:
        __name__ = "named"

    assert _method_name(Named()) == "named"
    assert _method_name(123) == "123"


def test_base_test_soft_assert_collects_and_reports_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("validrix.core.base_test.ConfigManager.load", lambda: FrameworkConfig())
    test_case = _SampleTest()
    test_case.setup_method("soft")
    test_case.soft_assert(True, "ok")
    test_case.soft_assert(False, "broken")

    with pytest.raises(Failed, match="1 soft assertion"):
        test_case.assert_soft_failures()


def test_base_test_assert_soft_failures_noop_when_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("validrix.core.base_test.ConfigManager.load", lambda: FrameworkConfig())
    test_case = _SampleTest()
    test_case.setup_method("none")

    test_case.assert_soft_failures()


def test_base_test_assert_eventually_accepts_boolean(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("validrix.core.base_test.ConfigManager.load", lambda: FrameworkConfig())
    test_case = _SampleTest()
    test_case.setup_method("bool")

    test_case.assert_eventually(True, timeout_seconds=0.01, poll_interval=0.001)


def test_base_test_assert_eventually_polls_callable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("validrix.core.base_test.ConfigManager.load", lambda: FrameworkConfig())
    test_case = _SampleTest()
    test_case.setup_method("callable")
    calls: list[int] = []

    def condition() -> bool:
        calls.append(1)
        return len(calls) >= 2

    test_case.assert_eventually(condition, timeout_seconds=0.1, poll_interval=0.001)

    assert len(calls) >= 2


def test_base_test_assert_eventually_fails_after_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("validrix.core.base_test.ConfigManager.load", lambda: FrameworkConfig())
    test_case = _SampleTest()
    test_case.setup_method("timeout")

    with pytest.raises(Failed, match="Condition not met"):
        test_case.assert_eventually(False, timeout_seconds=0.01, poll_interval=0.001)
