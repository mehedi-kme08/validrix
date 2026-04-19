from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from validrix.core.config_manager import AIConfig, FrameworkConfig
from validrix.plugins.ai_generator import (
    AIGeneratorPlugin,
    AITestGenerator,
    GenerationResult,
    _AnthropicClient,
    _make_client,
    _OpenAIClient,
)
from validrix.plugins.ai_reporter import AIReport, AIReporterPlugin, FailureRecord


def _capture_writes(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    writes: dict[str, str] = {}

    monkeypatch.setattr(Path, "mkdir", lambda self, parents=False, exist_ok=False: None)
    monkeypatch.setattr(
        Path,
        "write_text",
        lambda self, content, encoding="utf-8": writes.__setitem__(str(self), content) or len(content),
    )
    return writes


def test_ai_generator_factory_and_helpers() -> None:
    anthropic_cfg = AIConfig(provider="anthropic")
    openai_cfg = AIConfig(provider="openai")

    assert isinstance(_make_client(anthropic_cfg), _AnthropicClient)
    assert isinstance(_make_client(openai_cfg), _OpenAIClient)
    with pytest.raises(ValueError, match="Unknown AI provider"):
        _make_client(AIConfig.model_construct(provider="other"))  # type: ignore[arg-type]

    prompt = AITestGenerator._build_user_prompt("Login page", "Extra context")
    assert "FEATURE: Login page" in prompt
    assert "Extra context" in prompt
    assert AITestGenerator._extract_code("```python\nprint('x')\n```") == "print('x')"
    assert AITestGenerator._extract_code("raw code") == "raw code"
    with pytest.raises(ValueError, match="no extractable"):
        AITestGenerator._extract_code("   ")

    header = AITestGenerator._add_file_header("print('x')", "Feature")
    assert "AUTO-GENERATED" in header
    path = Path("generated/test_gen.py")
    writes: dict[str, str] = {}

    original_mkdir = Path.mkdir
    original_write = Path.write_text
    try:
        Path.mkdir = lambda self, parents=False, exist_ok=False: None  # type: ignore[method-assign]
        Path.write_text = lambda self, content, encoding="utf-8": writes.__setitem__(str(self), content) or len(content)  # type: ignore[method-assign]
        saved = AITestGenerator._write_to_file("print('x')", path)
    finally:
        Path.mkdir = original_mkdir  # type: ignore[method-assign]
        Path.write_text = original_write  # type: ignore[method-assign]

    assert writes[str(path)] == "print('x')"
    assert saved == path


def test_ai_generator_generate_uses_client_and_saves(monkeypatch: pytest.MonkeyPatch) -> None:
    writes = _capture_writes(monkeypatch)

    class FakeClient:
        def complete(self, system: str, user: str, config: AIConfig) -> str:
            return "```python\ndef test_ok() -> None:\n    assert True\n```"

    monkeypatch.setattr("validrix.core.config_manager.ConfigManager.load", lambda: FrameworkConfig())
    monkeypatch.setattr("validrix.plugins.ai_generator._make_client", lambda config: FakeClient())
    generator = AITestGenerator()
    output_path = Path("generated/test_feature.py")

    result = generator.generate("Feature", output_path=output_path, extra_context="ctx")

    assert isinstance(result, GenerationResult)
    assert result.saved_path == output_path
    assert str(output_path) in writes
    assert "Feature: Feature" in result.code


def test_ai_generator_generate_without_output_and_without_extra_context(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeClient:
        def complete(self, system: str, user: str, config: AIConfig) -> str:
            assert "Additional context" not in user
            return "def test_ok() -> None:\n    assert True\n"

    monkeypatch.setattr("validrix.core.config_manager.ConfigManager.load", lambda: FrameworkConfig())
    monkeypatch.setattr("validrix.plugins.ai_generator._make_client", lambda config: FakeClient())
    generator = AITestGenerator()

    result = generator.generate("Feature")

    assert result.saved_path is None
    assert result.description == "Feature"


def test_ai_generator_plugin_pytest_configure_is_noop() -> None:
    plugin = AIGeneratorPlugin()
    assert plugin.create_generator is AITestGenerator
    plugin.pytest_configure(object())


def test_ai_generator_provider_clients_call_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    anthropic_calls: list[dict[str, object]] = []
    openai_calls: list[dict[str, object]] = []

    class FakeAnthropic:
        def __init__(self, api_key: str, timeout: int) -> None:
            self.messages = SimpleNamespace(create=self.create)

        def create(self, **kwargs: object) -> object:
            anthropic_calls.append(kwargs)
            return SimpleNamespace(content=[SimpleNamespace(text="anthropic result")])

    class FakeOpenAI:
        def __init__(self, api_key: str, timeout: int) -> None:
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs: object) -> object:
            openai_calls.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="openai result"))])

    monkeypatch.setattr("validrix.plugins.ai_generator.anthropic.Anthropic", FakeAnthropic)
    monkeypatch.setattr("validrix.plugins.ai_generator.openai.OpenAI", FakeOpenAI)

    anthropic_result = _AnthropicClient().complete("sys", "user", AIConfig(anthropic_api_key="k"))
    openai_result = _OpenAIClient().complete("sys", "user", AIConfig(provider="openai", openai_api_key="k"))

    assert anthropic_result == "anthropic result"
    assert openai_result == "openai result"
    assert anthropic_calls[0]["system"] == "sys"
    assert openai_calls[0]["messages"][0]["role"] == "system"


def _failure_record() -> FailureRecord:
    return FailureRecord(
        test_id="tests/test_demo.py::test_case",
        test_name="test_case",
        error_type="AssertionError",
        error_message="broken",
        traceback="Traceback...\nAssertionError: broken",
        duration_seconds=0.2,
    )


def test_ai_report_to_dict_and_prompt_format() -> None:
    failure = _failure_record()
    report = AIReport(
        session_id="1",
        total_failures=1,
        generated_at="now",
        summary_markdown="summary",
        failures=[failure],
    )

    assert report.to_dict()["session_id"] == "1"
    prompt = AIReporterPlugin._format_failures_for_prompt([failure])
    assert "Total failures: 1" in prompt
    assert "AssertionError" in prompt


def test_ai_reporter_logreport_and_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    writes = _capture_writes(monkeypatch)
    tmp_path = Path("reports/checkpoint")
    cfg = FrameworkConfig()
    cfg.report_dir = tmp_path
    monkeypatch.setattr("validrix.plugins.ai_reporter.ConfigManager.load", lambda: cfg)
    plugin = AIReporterPlugin()
    report = SimpleNamespace(
        when="call",
        failed=True,
        nodeid="tests/test_file.py::test_name",
        longreprtext="Traceback\nValueError: nope",
        longrepr="Traceback\nValueError: nope",
        duration=0.5,
    )

    plugin.pytest_runtest_logreport(report)

    checkpoint = tmp_path / "_failures_checkpoint.json"
    payload = json.loads(writes[str(checkpoint)])
    assert payload[0]["error_type"] == "ValueError"


def test_ai_reporter_logreport_ignores_non_call_or_non_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    tmp_path = Path("reports/ignore")
    cfg = FrameworkConfig()
    cfg.report_dir = tmp_path
    monkeypatch.setattr("validrix.plugins.ai_reporter.ConfigManager.load", lambda: cfg)
    plugin = AIReporterPlugin()

    plugin.pytest_runtest_logreport(SimpleNamespace(when="setup", failed=True))
    plugin.pytest_runtest_logreport(SimpleNamespace(when="call", failed=False))

    assert plugin._failures == []


def test_ai_reporter_logreport_uses_unknown_error_when_no_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    writes = _capture_writes(monkeypatch)
    cfg = FrameworkConfig()
    cfg.report_dir = Path("reports/unknown")
    monkeypatch.setattr("validrix.plugins.ai_reporter.ConfigManager.load", lambda: cfg)
    plugin = AIReporterPlugin()
    report = SimpleNamespace(
        when="call",
        failed=True,
        nodeid="tests/test_file.py::test_name",
        longreprtext="Traceback only",
        longrepr="Traceback only",
        duration=0.5,
    )

    plugin.pytest_runtest_logreport(report)

    payload = json.loads(writes[str(Path("reports/unknown") / "_failures_checkpoint.json")])
    assert payload[0]["error_type"] == "UnknownError"


def test_ai_reporter_analyse_and_write_reports(monkeypatch: pytest.MonkeyPatch) -> None:
    writes = _capture_writes(monkeypatch)
    tmp_path = Path("reports/write")
    cfg = FrameworkConfig()
    cfg.report_dir = tmp_path
    cfg.ai.anthropic_api_key = "secret"
    monkeypatch.setattr("validrix.plugins.ai_reporter.ConfigManager.load", lambda: cfg)
    plugin = AIReporterPlugin()
    plugin._failures = [_failure_record()]
    monkeypatch.setattr(plugin, "_analyse_failures", lambda failures: "# Summary")

    plugin.pytest_sessionfinish(SimpleNamespace(), 1)

    assert writes[str(tmp_path / "report.md")] == "# Summary"
    data = json.loads(writes[str(tmp_path / "report.json")])
    assert data["total_failures"] == 1


def test_ai_reporter_analyse_failures_uses_anthropic_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = FrameworkConfig()
    cfg.report_dir = Path("reports/anthropic")
    monkeypatch.setattr("validrix.plugins.ai_reporter.ConfigManager.load", lambda: cfg)
    plugin = AIReporterPlugin()
    monkeypatch.setattr(plugin, "_call_anthropic", lambda payload: "anthropic")

    assert plugin._analyse_failures([_failure_record()]) == "anthropic"


def test_ai_reporter_sessionfinish_no_failures_or_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    writes = _capture_writes(monkeypatch)
    tmp_path = Path("reports/no_key")
    cfg = FrameworkConfig()
    cfg.report_dir = tmp_path
    monkeypatch.setattr("validrix.plugins.ai_reporter.ConfigManager.load", lambda: cfg)
    plugin = AIReporterPlugin()
    plugin.pytest_sessionfinish(SimpleNamespace(), 0)
    plugin._failures = [_failure_record()]
    plugin.pytest_sessionfinish(SimpleNamespace(), 1)

    assert str(tmp_path / "report.md") not in writes


def test_ai_reporter_sessionfinish_handles_analysis_error(monkeypatch: pytest.MonkeyPatch) -> None:
    writes = _capture_writes(monkeypatch)
    tmp_path = Path("reports/error")
    cfg = FrameworkConfig()
    cfg.report_dir = tmp_path
    cfg.ai.anthropic_api_key = "secret"
    monkeypatch.setattr("validrix.plugins.ai_reporter.ConfigManager.load", lambda: cfg)
    plugin = AIReporterPlugin()
    plugin._failures = [_failure_record()]
    monkeypatch.setattr(plugin, "_analyse_failures", lambda failures: (_ for _ in ()).throw(RuntimeError("boom")))

    plugin.pytest_sessionfinish(SimpleNamespace(), 1)

    assert str(tmp_path / "report.md") not in writes


def test_ai_reporter_generate_from_checkpoint_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    writes = _capture_writes(monkeypatch)
    tmp_path = Path("reports/from_checkpoint")
    checkpoint = tmp_path / "_failures_checkpoint.json"
    cfg = FrameworkConfig()
    cfg.report_dir = tmp_path
    monkeypatch.setattr("validrix.plugins.ai_reporter.ConfigManager.load", lambda: cfg)
    plugin = AIReporterPlugin()
    monkeypatch.setattr(Path, "exists", lambda self: self == checkpoint)
    monkeypatch.setattr(Path, "read_text", lambda self, encoding="utf-8": json.dumps([_failure_record().__dict__]))

    assert plugin.generate_from_checkpoint() is None

    assert plugin.generate_from_checkpoint() is None

    plugin._ai_config.anthropic_api_key = "secret"
    monkeypatch.setattr(plugin, "_analyse_failures", lambda failures: "done")
    result = plugin.generate_from_checkpoint()

    assert result == tmp_path / "report.md"
    assert writes[str(tmp_path / "report.md")] == "done"


def test_ai_reporter_generate_from_checkpoint_missing_file(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = FrameworkConfig()
    cfg.report_dir = Path("reports/missing_checkpoint")
    monkeypatch.setattr("validrix.plugins.ai_reporter.ConfigManager.load", lambda: cfg)
    plugin = AIReporterPlugin()
    monkeypatch.setattr(Path, "exists", lambda self: False)

    assert plugin.generate_from_checkpoint() is None


def test_ai_reporter_provider_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    anthropic_calls: list[dict[str, object]] = []
    openai_calls: list[dict[str, object]] = []

    class FakeAnthropic:
        def __init__(self, api_key: str, timeout: int) -> None:
            self.messages = SimpleNamespace(create=self.create)

        def create(self, **kwargs: object) -> object:
            anthropic_calls.append(kwargs)
            return SimpleNamespace(content=[SimpleNamespace(text="anthropic markdown")])

    class FakeOpenAI:
        def __init__(self, api_key: str, timeout: int) -> None:
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs: object) -> object:
            openai_calls.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="openai markdown"))])

    cfg = FrameworkConfig()
    cfg.report_dir = Path("reports/provider_calls")
    cfg.ai.anthropic_api_key = "secret"
    monkeypatch.setattr("validrix.plugins.ai_reporter.ConfigManager.load", lambda: cfg)
    monkeypatch.setattr("validrix.plugins.ai_reporter.anthropic.Anthropic", FakeAnthropic)
    monkeypatch.setattr("validrix.plugins.ai_reporter.openai.OpenAI", FakeOpenAI)
    plugin = AIReporterPlugin()

    assert plugin._call_anthropic("payload") == "anthropic markdown"
    plugin._ai_config.provider = "openai"
    plugin._ai_config.openai_api_key = "secret"
    assert plugin._call_openai("payload") == "openai markdown"
    assert plugin._analyse_failures([_failure_record()]) == "openai markdown"
    assert anthropic_calls[0]["messages"][0]["content"] == "payload"
    assert openai_calls[0]["messages"][1]["content"] == "payload"
