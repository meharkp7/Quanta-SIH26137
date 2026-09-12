"""Operating loop for the live demo.

Forecast and DRL slots are present but labelled as baselines until
``src/learning`` is trained.
"""

from src.runtime.loop import DemoLoop, LoopState

__all__ = ["DemoLoop", "LoopState"]
