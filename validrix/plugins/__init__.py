"""
Plugin registry for Validrix.

Design decision: Plugins are discovered via Python entry_points (PEP 517/518).
This mirrors pytest's own plugin system, letting third parties ship plugins as
independent pip packages without modifying the core framework.

Alternatives considered:
- Explicit import list: simple but not extensible by third parties.
- File-system scan: fragile and order-dependent.

Entry point group: "validrix.plugins"
"""

from validrix.plugins.ai_generator import AIGeneratorPlugin, AITestGenerator
from validrix.plugins.ai_reporter import AIReporterPlugin
from validrix.plugins.flaky_detector import FlakyDetectorPlugin
from validrix.plugins.self_healing import SelfHealingPlugin

__all__ = [
    "AIGeneratorPlugin",
    "AITestGenerator",
    "AIReporterPlugin",
    "SelfHealingPlugin",
    "FlakyDetectorPlugin",
]
