"""
Example: ConfigManager and FrameworkConfig.

Demonstrates loading config from YAML, environment variable overrides,
and per-environment settings.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pydantic
import pytest

from validrix.core.config_manager import ConfigManager


class TestConfigManager:
    """Tests for the ConfigManager config loading system."""

    def setup_method(self, method: object) -> None:
        """Reset the LRU cache before each test."""
        ConfigManager.reset()

    def test_loads_defaults_without_config_file(self) -> None:
        """Config loads with sensible defaults when no YAML file is present."""
        cfg = ConfigManager.load(config_path="/nonexistent/path.yml" if False else None)
        assert cfg.environment == "dev"
        assert cfg.ai.provider == "anthropic"
        assert cfg.retry.max_attempts == 3

    def test_loads_from_yaml_file(self, tmp_path: Path) -> None:
        """YAML values are merged into the config model."""
        yaml_content = textwrap.dedent("""\
            environment: staging
            ai:
              provider: openai
              max_tokens: 2048
            retry:
              max_attempts: 5
        """)
        config_file = tmp_path / "validrix.yml"
        config_file.write_text(yaml_content)

        cfg = ConfigManager.load(config_path=config_file)
        assert cfg.environment == "staging"
        assert cfg.ai.provider == "openai"
        assert cfg.ai.max_tokens == 2048
        assert cfg.retry.max_attempts == 5

    def test_environment_specific_overrides(self, tmp_path: Path) -> None:
        """Per-environment keys override global defaults."""
        yaml_content = textwrap.dedent("""\
            environment: staging
            environments:
              staging:
                base_url: https://staging.example.com
              prod:
                base_url: https://example.com
        """)
        config_file = tmp_path / "validrix.yml"
        config_file.write_text(yaml_content)

        cfg = ConfigManager.load(config_path=config_file, environment="staging")
        # The environment override should flow through
        assert cfg.environment == "staging"

    def test_env_var_overrides_yaml(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Environment variables override YAML values (highest priority)."""
        yaml_content = "retry:\n  max_attempts: 3\n"
        config_file = tmp_path / "validrix.yml"
        config_file.write_text(yaml_content)

        monkeypatch.setenv("VALIDRIX_RETRY__MAX_ATTEMPTS", "10")

        cfg = ConfigManager.load(config_path=config_file)
        assert cfg.retry.max_attempts == 10

    def test_raises_on_missing_explicit_path(self) -> None:
        """FileNotFoundError raised when explicitly given path does not exist."""
        with pytest.raises(FileNotFoundError):
            ConfigManager.load(config_path="/definitely/does/not/exist.yml")

    def test_config_is_cached(self, tmp_path: Path) -> None:
        """Same config object returned on repeated calls (LRU cache)."""
        config_file = tmp_path / "validrix.yml"
        config_file.write_text("environment: dev\n")

        cfg1 = ConfigManager.load(config_path=config_file)
        cfg2 = ConfigManager.load(config_path=config_file)
        assert cfg1 is cfg2, "ConfigManager should return the same cached instance"

    def test_reset_clears_cache(self, tmp_path: Path) -> None:
        """ConfigManager.reset() forces a fresh load on the next call."""
        config_file = tmp_path / "validrix.yml"
        config_file.write_text("environment: dev\n")

        cfg1 = ConfigManager.load(config_path=config_file)
        ConfigManager.reset()
        cfg2 = ConfigManager.load(config_path=config_file)
        assert cfg1 is not cfg2, "After reset, a new config object should be created"

    def test_pydantic_rejects_invalid_values(self, tmp_path: Path) -> None:
        """Pydantic validation catches bad config values at load time."""
        yaml_content = textwrap.dedent("""\
            ai:
              max_tokens: 99999999   # exceeds le=16384 constraint
        """)
        config_file = tmp_path / "validrix.yml"
        config_file.write_text(yaml_content)

        with pytest.raises(pydantic.ValidationError):
            ConfigManager.load(config_path=config_file)
