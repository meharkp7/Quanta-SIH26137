"""Demo-facing front door for the SIH26137 platform.

The research modules stay in ``src/data``, ``src/routing``, ``src/optim``,
and ``src/sim``. This package is the only layer the UI and runtime loop
should import.
"""

from src.platform.catalog import PROJECT_ROOT, list_scenarios, load_scenario
from src.platform.service import PlatformService

__all__ = [
    "PROJECT_ROOT",
    "PlatformService",
    "list_scenarios",
    "load_scenario",
]
