"""
Validrix — AI-Powered PyTest Plugin Framework.

Design decision: Keep __init__.py minimal. Expose only the public API surface
that downstream consumers need. Internal modules are imported lazily to avoid
circular imports and keep startup time fast.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("validrix")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

__all__ = ["__version__"]
