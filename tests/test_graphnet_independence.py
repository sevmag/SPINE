"""Core-independence gate: importing all of spine must not pull in graphnet.

The graphnet integration lives in the separate spine_graphnet package; the
core must stay importable in envs without graphnet. Importing every spine
submodule and then inspecting sys.modules catches any accidental coupling,
whether graphnet is installed (it would load) or not (it would raise).
"""

from __future__ import annotations

import importlib
import pkgutil
import sys


def test_core_is_graphnet_free():
    """Import every spine submodule; assert no graphnet module was loaded."""
    import spine

    for m in pkgutil.walk_packages(spine.__path__, "spine."):
        importlib.import_module(m.name)
    loaded = [m for m in sys.modules if m.split(".")[0] == "graphnet"]
    assert not loaded, f"spine core imported graphnet modules: {loaded}"
