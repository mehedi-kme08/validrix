"""
framework_cli.py — validrix CLI entry point.

Design decision: Click over argparse / Typer.
  WHY: Click's decorator API co-locates command definition with its handler,
       making individual commands easy to read, test, and extend. It handles
       --help generation, error formatting, and shell completion out of the box.
       Typer is a Click wrapper with auto-typing — we skip it to avoid another
       abstraction layer and keep direct access to Click's Context.

  Alternatives considered:
    - argparse: verbose, no help-text formatting, no command groups.
    - Typer: adds auto-completion sugar but hides Click's Context, which we
      need for pass_context propagation.
    - fire: magic, hard to test, difficult to add subcommand groups.

  Tradeoffs:
    - Click requires explicit type declarations on options (Typer infers them).
    - We use Rich for output so the CLI looks polished in any terminal that
      supports ANSI; Click handles plain text fallback automatically.

Commands:
  validrix generate "<description>"   → AI test generation
  validrix run [--env ENV]            → run tests via Docker
  validrix report                     → generate AI failure summary
  validrix scaffold <name>            → scaffold a new test project
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

console = Console()


# ---------------------------------------------------------------------------
# Root group
# ---------------------------------------------------------------------------


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    invoke_without_command=True,
)
@click.version_option(package_name="validrix", prog_name="validrix")
@click.pass_context
def cli(ctx: click.Context) -> None:
    """
    validrix — Validrix AI-Powered Test Framework CLI.

    \b
    Run `validrix COMMAND --help` for detailed help on any command.
    """
    if ctx.invoked_subcommand is None:
        console.print(
            Panel.fit(
                "[bold cyan]validrix[/] — Validrix AI Test Framework\n\n"
                "Run [bold]validrix --help[/] to see available commands.",
                border_style="cyan",
            )
        )


# ---------------------------------------------------------------------------
# generate command
# ---------------------------------------------------------------------------


@cli.command("generate")
@click.argument("description")
@click.option(
    "--output",
    "-o",
    default=None,
    help="Output file path for generated tests (e.g., tests/test_login.py).",
)
@click.option(
    "--provider",
    type=click.Choice(["anthropic", "openai"]),
    default=None,
    help="AI provider to use (overrides config).",
)
@click.option(
    "--context",
    "-c",
    default="",
    help="Additional context or instructions for the AI.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print generated code without writing to file.",
)
def generate(
    description: str,
    output: str | None,
    provider: str | None,
    context: str,
    dry_run: bool,
) -> None:
    """
    Generate pytest test cases from a plain-English DESCRIPTION.

    \b
    Examples:
      validrix generate "Login page with email and password fields"
      validrix generate "REST API /users endpoint" -o tests/test_users.py
      validrix generate "Checkout flow" --provider openai --dry-run
    """
    from validrix.core.config_manager import ConfigManager
    from validrix.plugins.ai_generator import AITestGenerator

    cfg = ConfigManager.load()

    if provider:
        cfg.ai.provider = provider  # type: ignore[assignment]

    console.print(f"\n[bold cyan]Generating tests for:[/] {description}\n")

    output_path = Path(output) if output else None

    with console.status("[bold green]Calling AI…"):
        try:
            gen = AITestGenerator(ai_config=cfg.ai)
            result = gen.generate(
                description=description,
                output_path=None if dry_run else output_path,
                extra_context=context,
            )
        except Exception as exc:
            console.print(f"[bold red]Error:[/] {exc}")
            sys.exit(1)

    syntax = Syntax(result.code, "python", theme="monokai", line_numbers=True)
    console.print(Panel(syntax, title="Generated Tests", border_style="green"))

    if dry_run:
        console.print("\n[yellow]--dry-run:[/] code not written to file.")
    elif output_path:
        console.print(f"\n[green]✓[/] Tests written to [bold]{output_path}[/]")
    else:
        console.print("\n[yellow]Tip:[/] Use -o <path> to save the generated tests.")


# ---------------------------------------------------------------------------
# run command
# ---------------------------------------------------------------------------


@cli.command("run")
@click.option(
    "--env",
    "-e",
    default="dev",
    show_default=True,
    help="Target environment (dev | staging | prod).",
)
@click.option(
    "--docker/--no-docker",
    default=False,
    help="Run tests inside Docker container.",
)
@click.option(
    "--detect-flaky",
    is_flag=True,
    default=False,
    help="Enable flaky test detection (runs each test multiple times).",
)
@click.option(
    "--marker",
    "-m",
    default=None,
    help="pytest -m marker expression (e.g., 'smoke and not slow').",
)
@click.argument("pytest_args", nargs=-1, type=click.UNPROCESSED)
def run(
    env: str,
    docker: bool,
    detect_flaky: bool,
    marker: str | None,
    pytest_args: tuple[str, ...],
) -> None:
    """
    Run the test suite, optionally inside Docker.

    \b
    Examples:
      validrix run
      validrix run --env staging
      validrix run --docker --env prod
      validrix run --detect-flaky -m smoke
      validrix run -- -k test_login -v
    """
    import os

    os.environ["VALIDRIX_ENVIRONMENT"] = env

    if docker:
        _run_in_docker(env, detect_flaky, marker, pytest_args)
    else:
        _run_locally(env, detect_flaky, marker, pytest_args)


def _run_locally(
    env: str,
    detect_flaky: bool,
    marker: str | None,
    extra_args: tuple[str, ...],
) -> None:
    cmd = [sys.executable, "-m", "pytest"]
    if detect_flaky:
        cmd.append("--detect-flaky")
    if marker:
        cmd += ["-m", marker]
    cmd += list(extra_args)

    console.print(f"[cyan]Environment:[/] {env}")
    console.print(f"[cyan]Command:[/] {' '.join(cmd)}\n")

    result = subprocess.run(cmd, check=False)
    sys.exit(result.returncode)


def _run_in_docker(
    env: str,
    detect_flaky: bool,
    marker: str | None,
    extra_args: tuple[str, ...],
) -> None:
    pytest_flags = []
    if detect_flaky:
        pytest_flags.append("--detect-flaky")
    if marker:
        pytest_flags += ["-m", marker]
    pytest_flags += list(extra_args)

    cmd = [
        "docker",
        "compose",
        "run",
        "--rm",
        "-e",
        f"VALIDRIX_ENVIRONMENT={env}",
        "validrix",
        "pytest",
        *pytest_flags,
    ]

    console.print(f"[cyan]Docker command:[/] {' '.join(cmd)}\n")
    result = subprocess.run(cmd, check=False)
    sys.exit(result.returncode)


# ---------------------------------------------------------------------------
# report command
# ---------------------------------------------------------------------------


@cli.command("report")
@click.option(
    "--output-dir",
    "-d",
    default="validrix_reports",
    show_default=True,
    help="Directory containing the failure checkpoint.",
)
def report(output_dir: str) -> None:
    """
    Generate an AI failure summary from the last test run.

    Reads the failure checkpoint written during the session and sends it
    to the configured AI provider for root-cause analysis.

    \b
    Example:
      validrix report
      validrix report --output-dir custom_reports/
    """
    from validrix.core.config_manager import ConfigManager
    from validrix.plugins.ai_reporter import AIReporterPlugin

    cfg = ConfigManager.load()
    cfg.report_dir = Path(output_dir)

    reporter = AIReporterPlugin()
    reporter._report_dir = Path(output_dir)

    console.print("[bold cyan]Generating AI failure report…[/]\n")

    with console.status("[bold green]Analysing failures…"):
        result_path = reporter.generate_from_checkpoint()

    if result_path:
        console.print(f"[green]✓[/] Report saved to [bold]{result_path}[/]")
    else:
        console.print("[yellow]No failures to report or missing API key.[/]")


# ---------------------------------------------------------------------------
# scaffold command
# ---------------------------------------------------------------------------


@cli.command("scaffold")
@click.argument("project_name")
@click.option(
    "--destination",
    "-d",
    default=".",
    help="Parent directory for the new project.",
)
def scaffold(project_name: str, destination: str) -> None:
    """
    Scaffold a new Validrix test project structure.

    \b
    Example:
      validrix scaffold my_test_suite
      validrix scaffold my_tests --destination ~/projects/
    """
    dest = Path(destination) / project_name

    structure = {
        dest / "tests" / "unit" / "__init__.py": "# Unit tests\n",
        dest / "tests" / "e2e" / "__init__.py": "# E2E tests\n",
        dest / "tests" / "conftest.py": _CONFTEST_TEMPLATE.format(project=project_name),
        dest / "validrix.yml": _CONFIG_TEMPLATE.format(project=project_name),
        dest / "pyproject.toml": _PYPROJECT_TEMPLATE.format(project=project_name),
        dest / ".gitignore": _GITIGNORE_CONTENT,
    }

    for path, content in structure.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    # Show what was created
    table = Table(title=f"Scaffolded: {project_name}", show_header=True)
    table.add_column("File", style="cyan")
    table.add_column("Status", style="green")
    for path in structure:
        table.add_row(str(path.relative_to(dest.parent)), "created")

    console.print(table)
    console.print(f"\n[green]✓[/] Project [bold]{project_name}[/] created at [bold]{dest}[/]")
    console.print(
        f"\n[cyan]Next steps:[/]\n"
        f"  cd {dest}\n"
        f"  pip install -e '.[dev]'\n"
        f"  validrix generate 'Your first feature description'\n"
    )


# ---------------------------------------------------------------------------
# Scaffold templates
# ---------------------------------------------------------------------------

_CONFTEST_TEMPLATE = textwrap.dedent("""\
    \"\"\"Root conftest for {project}.\"\"\"
    import pytest

    @pytest.fixture(scope="session")
    def base_url() -> str:
        from validrix.core.config_manager import ConfigManager
        return ConfigManager.load().env.base_url
""")

_CONFIG_TEMPLATE = textwrap.dedent("""\
    # Validrix configuration for {project}
    environment: dev

    ai:
      provider: anthropic
      model: claude-sonnet-4-20250514

    environments:
      dev:
        base_url: http://localhost:3000
      staging:
        base_url: https://staging.{project}.example.com
      prod:
        base_url: https://{project}.example.com
""")

_PYPROJECT_TEMPLATE = textwrap.dedent("""\
    [build-system]
    requires = ["hatchling"]
    build-backend = "hatchling.build"

    [project]
    name = "{project}"
    version = "0.1.0"
    requires-python = ">=3.11"
    dependencies = [
        "validrix>=0.1",
        "pytest>=8.2",
        "playwright>=1.44",
    ]

    [tool.pytest.ini_options]
    testpaths = ["tests"]
""")

_GITIGNORE_CONTENT = textwrap.dedent("""\
    __pycache__/
    *.py[cod]
    .venv/
    .env
    validrix_reports/
    .pytest_cache/
""")
