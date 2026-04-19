"""
config_manager.py — Centralised, type-safe framework configuration.

Design decision: Pydantic v2 BaseSettings.
  WHY: Pydantic v2 gives us compile-time type validation, automatic env-var
       overlay, YAML deserialization in one model, and IDE auto-complete for
       free.  It eliminates an entire class of "config key typo" bugs that
       plague string-dict config systems.

  Alternatives considered:
    - dynaconf: powerful but heavy; introduces another DSL to learn.
    - configparser (INI): no type safety, no env overlay, no nested models.
    - plain dict from PyYAML: no validation, no IDE support.

  Tradeoffs:
    - Pydantic v2 is a compiled Rust extension; adds ~6 MB to the wheel.
    - Config is validated eagerly at import time, which surfaces mistakes
      before any test runs (deliberate: fail fast).

Layer order (later layers override earlier ones):
  1. Hardcoded defaults (in model)
  2. validrix.yml  (project-level file)
  3. Environment variables  (CI/CD secrets)
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

# ---------------------------------------------------------------------------
# Sub-models — plain BaseModel (not BaseSettings).
#
# Design decision: Only FrameworkConfig is a BaseSettings.  Sub-models are
# plain BaseModel so that env-var reading happens in exactly ONE place with
# ONE set of rules (env_prefix + env_nested_delimiter on FrameworkConfig).
# Having each sub-model read its own env vars with its own prefix created
# dual-lookup ambiguity: VALIDRIX_RETRY_MAX_ATTEMPTS (sub-model prefix) vs.
# VALIDRIX_RETRY__MAX_ATTEMPTS (nested delimiter).  Single-source wins.
# ---------------------------------------------------------------------------


class AIConfig(BaseModel):
    """Configuration for AI provider credentials and behaviour."""

    provider: Literal["anthropic", "openai"] = Field(
        default="anthropic",
        description="Which AI provider to use for test generation and reporting.",
    )
    anthropic_api_key: str = Field(
        default="",
        description="Anthropic API key. Set via VALIDRIX_AI__ANTHROPIC_API_KEY.",
    )
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key. Set via VALIDRIX_AI__OPENAI_API_KEY.",
    )
    # claude-sonnet-4-20250514 is the flagship reasoning model as of mid-2025
    model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Model identifier passed to the chosen provider.",
    )
    max_tokens: int = Field(default=4096, ge=256, le=16384)
    temperature: float = Field(default=0.2, ge=0.0, le=1.0)
    timeout_seconds: int = Field(default=60, ge=5, le=300)


class RetryConfig(BaseModel):
    """Global retry defaults shared by all plugins and base test classes."""

    max_attempts: int = Field(default=3, ge=1, le=10)
    delay_seconds: float = Field(default=1.0, ge=0.1)
    backoff_multiplier: float = Field(default=2.0, ge=1.0)
    # Jitter prevents thundering-herd when many tests retry simultaneously
    jitter: bool = Field(default=True)


class FlakyConfig(BaseModel):
    """Flaky test detector settings."""

    enabled: bool = Field(default=True)
    runs: int = Field(default=3, ge=2, le=20, description="How many times to run each test.")
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Pass rate below this value → test marked FLAKY.",
    )
    report_path: Path = Field(default=Path("validrix_reports/flaky_report.json"))


class SelfHealingConfig(BaseModel):
    """Self-healing Playwright locator settings."""

    enabled: bool = Field(default=True)
    history_path: Path = Field(default=Path("validrix_reports/healing_history.json"))
    # Order in which fallback strategies are attempted
    fallback_order: list[str] = Field(
        default=["aria-label", "text", "nearby", "css"],
    )


class EnvironmentConfig(BaseModel):
    """Per-environment overrides (base_url, credentials, etc.)."""

    model_config = {"extra": "allow"}

    name: str = Field(default="dev")
    base_url: str = Field(default="http://localhost:8080")
    headless: bool = Field(default=True)
    slow_mo: int = Field(default=0, ge=0, description="Playwright slowMo in ms.")
    timeout_ms: int = Field(default=30_000, ge=1000)


# ---------------------------------------------------------------------------
# Root config model
# ---------------------------------------------------------------------------


class FrameworkConfig(BaseSettings):
    """
    Top-level configuration model for the entire Validrix framework.

    Merges YAML file + environment variables into a single typed object.
    """

    model_config = SettingsConfigDict(
        env_prefix="VALIDRIX_",
        env_nested_delimiter="__",  # VALIDRIX_AI__MODEL=... maps to ai.model
        extra="ignore",
    )

    environment: str = Field(default="dev")
    report_dir: Path = Field(default=Path("validrix_reports"))

    ai: AIConfig = Field(default_factory=AIConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    flaky: FlakyConfig = Field(default_factory=FlakyConfig)
    healing: SelfHealingConfig = Field(default_factory=SelfHealingConfig)
    env: EnvironmentConfig = Field(default_factory=EnvironmentConfig)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # env_settings before init_settings so env vars override YAML-sourced init data.
        # This is the correct "env beats config file" priority for a CI-friendly framework.
        return (env_settings, init_settings, dotenv_settings, file_secret_settings)

    @field_validator("report_dir", mode="before")
    @classmethod
    def _coerce_report_dir(cls, v: str | Path) -> Path:
        return Path(v)

    @model_validator(mode="after")
    def _sync_env_name(self) -> FrameworkConfig:
        """Keep EnvironmentConfig.name consistent with root environment field."""
        self.env.name = self.environment
        return self


# ---------------------------------------------------------------------------
# ConfigManager — singleton loader
# ---------------------------------------------------------------------------


class ConfigManager:
    """
    Loads and caches the merged framework configuration.

    Usage::

        cfg = ConfigManager.load()
        print(cfg.ai.model)

    Design decision: @lru_cache on a module-level function rather than a
    Singleton class.  lru_cache is thread-safe and requires less boilerplate.
    Calling ConfigManager.load() is idempotent for the lifetime of the process.
    """

    _DEFAULT_CONFIG_FILE = "validrix.yml"

    @staticmethod
    def load(
        config_path: str | Path | None = None,
        environment: str | None = None,
    ) -> FrameworkConfig:
        """
        Return the fully merged FrameworkConfig.

        Args:
            config_path: Path to YAML config file. Defaults to ``validrix.yml``
                         in the current working directory.
            environment: Target environment name (dev / staging / prod).
                         Overrides ``VALIDRIX_ENVIRONMENT`` env var.

        Returns:
            Validated FrameworkConfig instance.

        Raises:
            FileNotFoundError: If a non-default config_path is given but missing.
            pydantic.ValidationError: If config values fail type/range checks.
        """
        return _load_config(
            config_path=str(config_path) if config_path else None,
            environment=environment,
        )

    @staticmethod
    def reset() -> None:
        """Bust the cache — useful in tests that mutate config."""
        _load_config.cache_clear()


@lru_cache(maxsize=1)
def _load_config(
    config_path: str | None,
    environment: str | None,
) -> FrameworkConfig:
    """Cached config loader. Separated so cache_clear() targets this function."""
    yaml_data: dict[str, Any] = {}

    # Resolve config file path
    resolved = Path(config_path or ConfigManager._DEFAULT_CONFIG_FILE)

    if resolved.exists():
        with resolved.open() as fh:
            raw = yaml.safe_load(fh) or {}

        # Support per-environment overrides in YAML:
        # environments:
        #   staging:
        #     base_url: https://staging.example.com
        target_env = environment or os.getenv("VALIDRIX_ENVIRONMENT", "dev")
        yaml_data = raw.get("default", raw)  # top-level keys are defaults
        env_overrides: dict[str, Any] = raw.get("environments", {}).get(target_env, {})
        _deep_merge(yaml_data, env_overrides)

    elif config_path:
        # Only raise if user explicitly requested a path that doesn't exist
        raise FileNotFoundError(f"Validrix config not found: {resolved}")

    # Pydantic will further overlay env vars on top of YAML data
    return FrameworkConfig(**yaml_data)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> None:
    """In-place recursive merge of override into base."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
