"""
docker_runner.py — Programmatic Docker Compose runner for test execution.

Design decision: Subprocess delegation, not Docker SDK.
  WHY: The Docker Python SDK is a 30 MB transitive dependency. Subprocess
       calls to `docker compose` are simpler, keep the dependency tree lean,
       and produce output that streams directly to the user's terminal.

  Alternatives considered:
    - docker-py (Docker SDK): full API access but heavyweight.
    - Testcontainers-python: great for per-test containers but overkill for
      running the full suite inside a pre-built Compose service.

  Tradeoffs:
    - Subprocess approach requires `docker` to be on PATH.  We validate this
      at DockerRunner construction time with a clear error message.
    - Output streaming works naturally via subprocess's inherited stdio.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class DockerRunner:
    """
    Runs the Validrix test suite inside a Docker Compose service.

    Usage::

        runner = DockerRunner()
        exit_code = runner.run(environment="staging", pytest_args=["-m", "smoke"])
        sys.exit(exit_code)
    """

    SERVICE_NAME = "validrix"

    def __init__(self, compose_file: Path | None = None) -> None:
        """
        Initialise the runner.

        Args:
            compose_file: Path to docker-compose.yml. Defaults to the file in
                          the current working directory.

        Raises:
            OSError: If `docker` is not on PATH.
        """
        if not shutil.which("docker"):
            raise OSError("docker is not on PATH. Install Docker Desktop and retry.")
        self._compose_file = compose_file or Path("docker-compose.yml")

    def run(
        self,
        environment: str = "dev",
        pytest_args: list[str] | None = None,
        env_overrides: dict[str, str] | None = None,
    ) -> int:
        """
        Execute pytest inside the Compose service.

        Args:
            environment:   Target environment name (sets VALIDRIX_ENVIRONMENT).
            pytest_args:   Additional pytest flags (e.g., ["-m", "smoke"]).
            env_overrides: Extra environment variables passed to the container.

        Returns:
            pytest exit code (0 = all passed).
        """
        cmd = self._build_command(environment, pytest_args or [], env_overrides or {})
        logger.info("DockerRunner: %s", " ".join(cmd))

        result = subprocess.run(cmd, check=False)
        return result.returncode

    def _build_command(
        self,
        environment: str,
        pytest_args: list[str],
        env_overrides: dict[str, str],
    ) -> list[str]:
        env_flags: list[str] = ["-e", f"VALIDRIX_ENVIRONMENT={environment}"]
        for key, value in env_overrides.items():
            env_flags += ["-e", f"{key}={value}"]

        return [
            "docker",
            "compose",
            "-f",
            str(self._compose_file),
            "run",
            "--rm",
            *env_flags,
            self.SERVICE_NAME,
            "pytest",
            *pytest_args,
        ]

    def build(self, no_cache: bool = False) -> int:
        """
        Build (or rebuild) the Docker image.

        Args:
            no_cache: If True, force a full rebuild ignoring the layer cache.

        Returns:
            docker compose build exit code.
        """
        cmd = [
            "docker",
            "compose",
            "-f",
            str(self._compose_file),
            "build",
        ]
        if no_cache:
            cmd.append("--no-cache")

        logger.info("DockerRunner.build: %s", " ".join(cmd))
        result = subprocess.run(cmd, check=False)
        return result.returncode
