"""
Validrix — AI-Powered PyTest Plugin Framework.

Design decision: Keep __init__.py minimal. Expose only the public API surface
that downstream consumers need. Internal modules are imported lazily to avoid
circular imports and keep startup time fast.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("validrix")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

__all__ = ["__version__", "web_agent"]

# web_agent is imported lazily — `from validrix import web_agent` triggers it
# without paying the cost of Playwright + FastAPI imports for CLI-only usage.
