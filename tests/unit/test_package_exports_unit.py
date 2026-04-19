from __future__ import annotations

import importlib
import importlib.metadata
import sys

from validrix.conftest import pytest_configure


def test_validrix_version_uses_installed_distribution(monkeypatch: object) -> None:
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "9.9.9")
    sys.modules.pop("validrix", None)

    module = importlib.import_module("validrix")

    assert module.__version__ == "9.9.9"
    assert module.__all__ == ["__version__"]


def test_validrix_version_falls_back_when_package_missing(monkeypatch: object) -> None:
    def raise_not_found(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(importlib.metadata, "version", raise_not_found)
    sys.modules.pop("validrix", None)

    module = importlib.import_module("validrix")

    assert module.__version__ == "0.0.0-dev"


def test_public_package_exports() -> None:
    import validrix.cli as cli_package
    import validrix.core as core_package
    import validrix.integrations as integrations_package
    import validrix.plugins as plugins_package

    assert "cli" in cli_package.__all__
    assert "ConfigManager" in core_package.__all__
    assert "DockerRunner" in integrations_package.__all__
    assert "AIReporterPlugin" in plugins_package.__all__


class _PluginManager:
    def __init__(self) -> None:
        self._registered: dict[str, object] = {}

    def hasplugin(self, name: str) -> bool:
        return name in self._registered

    def register(self, plugin: object, name: str) -> None:
        self._registered[name] = plugin


class _Config:
    def __init__(self) -> None:
        self.pluginmanager = _PluginManager()


def test_root_conftest_registers_plugins_once() -> None:
    config = _Config()

    pytest_configure(config)
    first_plugins = dict(config.pluginmanager._registered)
    pytest_configure(config)

    assert set(first_plugins) == {
        "validrix-flaky",
        "validrix-ai-reporter",
        "validrix-self-healing",
    }
    assert config.pluginmanager._registered == first_plugins


def test_reload_import_surfaces_for_coverage() -> None:
    module_names = [
        "validrix.cli",
        "validrix.core",
        "validrix.integrations",
        "validrix.plugins",
        "validrix.conftest",
        "validrix.core.base_test",
        "validrix.core.config_manager",
        "validrix.core.retry_manager",
        "validrix.cli.framework_cli",
        "validrix.integrations.docker_runner",
        "validrix.plugins.ai_generator",
        "validrix.plugins.ai_reporter",
        "validrix.plugins.flaky_detector",
        "validrix.plugins.self_healing",
    ]

    for name in module_names:
        importlib.reload(importlib.import_module(name))
