"""graphnet integration: DeepIce backbone + reader adapters for SPINE.

Requires graphnet on the import path (env-provided checkout, not on PyPI).
The spine core never imports this package -- the dependency is strictly
one-way, spine_graphnet -> spine.
"""

# fail here, not deep inside hydra.instantiate, when the env lacks graphnet
try:
    import graphnet  # noqa: F401
except ImportError as e:
    raise ImportError(
        "spine_graphnet requires graphnet on the import path "
        "(env-provided checkout, not on PyPI); core spine works without it"
    ) from e
