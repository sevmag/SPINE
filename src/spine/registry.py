"""Tiny name -> factory registries so backbones/tasks are picked by config."""

from __future__ import annotations

from typing import Callable, Dict


class Registry:
    """Maps a string name to a factory (class or callable)."""

    def __init__(self, kind: str):
        self.kind = kind
        self._m: Dict[str, Callable] = {}

    def register(self, name: str):
        def deco(obj):
            if name in self._m:
                raise KeyError(f"{self.kind} {name!r} already registered")
            self._m[name] = obj
            return obj

        return deco

    def build(self, name: str, **kwargs):
        if name not in self._m:
            raise KeyError(
                f"unknown {self.kind} {name!r}; registered: {sorted(self._m)}"
            )
        return self._m[name](**kwargs)

    def available(self):
        return sorted(self._m)
