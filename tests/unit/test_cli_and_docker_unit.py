from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from validrix.cli.framework_cli import (
    _run_in_docker,
    _run_locally,
    cli,
)
from validrix.core.config_manager import FrameworkConfig
from validrix.integrations.docker_runner import DockerRunner


def _capture_paths(monkeypatch: pytest.MonkeyPatch) -> tuple[list[str], list[str]]:
    created_dirs: list[str] = []
    written_files: list[str] = []

    monkeypatch.setattr(
        Path,
        "mkdir",
        lambda self, parents=False, exist_ok=False: created_dirs.append(str(self)),
    )
    monkeypatch.setattr(
        Path,
        "write_text",
        lambda self, content, encoding="utf-8": written_files.append(str(self)) or len(content),
    )
    return created_dirs, written_files


class _FakeGenerator:
    def __init__(self, ai_config: object) -> None:
        self.ai_config = ai_config

    def generate(self, description: str, output_path: Path | None, extra_context: str) -> object:
        return SimpleNamespace(code="def test_generated() -> None:\n    assert True\n", saved_path=output_path)


def test_cli_without_subcommand_shows_panel() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, [])

    assert result.exit_code == 0
    assert "Validrix AI Test Framework" in result.output


def test_generate_command_dry_run_uses_generator(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = FrameworkConfig()
    monkeypatch.setattr("validrix.core.config_manager.ConfigManager.load", lambda: cfg)
    monkeypatch.setattr("validrix.plugins.ai_generator.AITestGenerator", _FakeGenerator)

    runner = CliRunner()
    result = runner.invoke(cli, ["generate", "Login page", "--provider", "openai", "--dry-run", "-c", "More"])

    assert result.exit_code == 0
    assert cfg.ai.provider == "openai"
    assert "Generated Tests" in result.output
    assert "--dry-run" in result.output


def test_generate_command_prints_save_path_and_tip(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = FrameworkConfig()
    monkeypatch.setattr("validrix.core.config_manager.ConfigManager.load", lambda: cfg)
    monkeypatch.setattr("validrix.plugins.ai_generator.AITestGenerator", _FakeGenerator)
    runner = CliRunner()

    saved = runner.invoke(cli, ["generate", "Login page", "--output", "tests/test_login.py"])
    tip = runner.invoke(cli, ["generate", "Login page"])

    assert saved.exit_code == 0
    assert "Tests written to" in saved.output
    assert tip.exit_code == 0
    assert "Use -o <path>" in tip.output


def test_generate_command_handles_generator_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class BrokenGenerator:
        def __init__(self, ai_config: object) -> None:
            self.ai_config = ai_config

        def generate(self, description: str, output_path: Path | None, extra_context: str) -> object:
            raise RuntimeError("boom")

    monkeypatch.setattr("validrix.core.config_manager.ConfigManager.load", lambda: FrameworkConfig())
    monkeypatch.setattr("validrix.plugins.ai_generator.AITestGenerator", BrokenGenerator)

    runner = CliRunner()
    result = runner.invoke(cli, ["generate", "Broken"])

    assert result.exit_code == 1
    assert "Error:" in result.output


def test_run_locally_builds_pytest_command(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str]] = []

    def fake_run(cmd: list[str], check: bool) -> object:
        seen.append(cmd)
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr("validrix.cli.framework_cli.subprocess.run", fake_run)

    with pytest.raises(SystemExit) as exc:
        _run_locally("staging", True, "smoke", ("-k", "login"))

    assert exc.value.code == 7
    assert "--detect-flaky" in seen[0]
    assert seen[0][-2:] == ["-k", "login"]


def test_run_locally_without_optional_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str]] = []

    def fake_run(cmd: list[str], check: bool) -> object:
        seen.append(cmd)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("validrix.cli.framework_cli.subprocess.run", fake_run)

    with pytest.raises(SystemExit):
        _run_locally("dev", False, None, ())

    assert seen[0][1:] == ["-m", "pytest"]
    assert "--detect-flaky" not in seen[0]
    assert len(seen[0]) == 3


def test_run_in_docker_builds_compose_command(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str]] = []

    def fake_run(cmd: list[str], check: bool) -> object:
        seen.append(cmd)
        return SimpleNamespace(returncode=3)

    monkeypatch.setattr("validrix.cli.framework_cli.subprocess.run", fake_run)

    with pytest.raises(SystemExit) as exc:
        _run_in_docker("prod", False, None, ("-q",))

    assert exc.value.code == 3
    assert seen[0][:5] == ["docker", "compose", "run", "--rm", "-e"]
    assert "VALIDRIX_ENVIRONMENT=prod" in seen[0]


def test_run_in_docker_includes_detect_flaky_and_marker(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str]] = []

    def fake_run(cmd: list[str], check: bool) -> object:
        seen.append(cmd)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("validrix.cli.framework_cli.subprocess.run", fake_run)

    with pytest.raises(SystemExit):
        _run_in_docker("dev", True, "smoke", ())

    assert "--detect-flaky" in seen[0]
    assert "-m" in seen[0]


def test_run_command_selects_local_or_docker(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str]] = []
    runner = CliRunner()
    monkeypatch.setattr(
        "validrix.cli.framework_cli._run_locally",
        lambda env, detect_flaky, marker, args: calls.append(("local", env)),
    )
    monkeypatch.setattr(
        "validrix.cli.framework_cli._run_in_docker",
        lambda env, detect_flaky, marker, args: calls.append(("docker", env)),
    )

    assert runner.invoke(cli, ["run"]).exit_code == 0
    assert runner.invoke(cli, ["run", "--docker", "--env", "prod"]).exit_code == 0
    assert calls == [("local", "dev"), ("docker", "prod")]


def test_report_command_generates_or_skips(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeReporter:
        def __init__(self) -> None:
            self._report_dir = Path(".")

        def generate_from_checkpoint(self) -> Path | None:
            return self._report_dir / "report.md"

    monkeypatch.setattr("validrix.core.config_manager.ConfigManager.load", lambda: FrameworkConfig())
    monkeypatch.setattr("validrix.plugins.ai_reporter.AIReporterPlugin", FakeReporter)
    runner = CliRunner()

    result = runner.invoke(cli, ["report", "--output-dir", "custom"])

    assert result.exit_code == 0
    assert "Report saved" in result.output


def test_report_command_handles_missing_report(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeReporter:
        def __init__(self) -> None:
            self._report_dir = Path(".")

        def generate_from_checkpoint(self) -> Path | None:
            return None

    monkeypatch.setattr("validrix.core.config_manager.ConfigManager.load", lambda: FrameworkConfig())
    monkeypatch.setattr("validrix.plugins.ai_reporter.AIReporterPlugin", FakeReporter)
    runner = CliRunner()

    result = runner.invoke(cli, ["report"])

    assert result.exit_code == 0
    assert "No failures to report" in result.output


def test_scaffold_creates_project_files() -> None:
    monkeypatch = pytest.MonkeyPatch()
    created_dirs, written_files = _capture_paths(monkeypatch)
    destination = Path("scaffold-output")
    runner = CliRunner()
    try:
        result = runner.invoke(cli, ["scaffold", "demo", "--destination", str(destination)])
    finally:
        monkeypatch.undo()

    assert result.exit_code == 0
    assert created_dirs
    assert any(path.endswith("demo\\validrix.yml") or path.endswith("demo/validrix.yml") for path in written_files)
    assert "validrix.yml" in result.output
    assert "tests\\conftest.py" in result.output or "tests/conftest.py" in result.output
    assert "Next steps" in result.output


def test_docker_runner_requires_docker_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("validrix.integrations.docker_runner.shutil.which", lambda name: None)

    with pytest.raises(OSError, match="docker is not on PATH"):
        DockerRunner()


def test_docker_runner_builds_and_runs_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("validrix.integrations.docker_runner.shutil.which", lambda name: "docker")
    seen: list[list[str]] = []

    def fake_run(cmd: list[str], check: bool) -> object:
        seen.append(cmd)
        return SimpleNamespace(returncode=len(seen))

    monkeypatch.setattr("validrix.integrations.docker_runner.subprocess.run", fake_run)
    runner = DockerRunner(compose_file=Path("compose.yml"))

    run_code = runner.run(environment="staging", pytest_args=["-m", "smoke"], env_overrides={"FOO": "bar"})
    build_code = runner.build(no_cache=True)

    assert run_code == 1
    assert build_code == 2
    assert seen[0][-3:] == ["pytest", "-m", "smoke"]
    assert "FOO=bar" in seen[0]
    assert seen[1][-1] == "--no-cache"


def test_docker_runner_build_without_no_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("validrix.integrations.docker_runner.shutil.which", lambda name: "docker")
    seen: list[list[str]] = []
    monkeypatch.setattr(
        "validrix.integrations.docker_runner.subprocess.run",
        lambda cmd, check: seen.append(cmd) or SimpleNamespace(returncode=0),
    )
    runner = DockerRunner(compose_file=Path("compose.yml"))

    runner.build()

    assert seen[0][-1] == "build"
