"""
Root conftest.py — auto-loaded by pytest for any test session using validrix.

Design decision: Register all built-in plugins here via pytest11 entry point
rather than in individual conftest files. This ensures plugins are always
available without users having to import them manually, matching pytest's own
conventions (e.g., pytest-cov, pytest-xdist).

Plugins that need configuration accept it through pytest.ini / pyproject.toml
options declared in their respective register() hooks.
"""

from __future__ import annotations

import pytest

from validrix.plugins.flaky_detector import FlakyDetectorPlugin
from validrix.plugins.ai_reporter import AIReporterPlugin
from validrix.plugins.self_healing import SelfHealingPlugin


def pytest_configure(config: pytest.Config) -> None:
    """Register all built-in Validrix plugins with the pytest plugin manager."""
    config.pluginmanager.register(FlakyDetectorPlugin(config), "validrix-flaky")
    config.pluginmanager.register(AIReporterPlugin(config), "validrix-ai-reporter")
    config.pluginmanager.register(SelfHealingPlugin(config), "validrix-self-healing")
