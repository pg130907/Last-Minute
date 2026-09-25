"""Instance registry — the staged progression from spec §5c.

Each entry is a factory that returns (graph, K, capacity). We keep the
registry small and self-contained: no network fetches, no giant files.
Solomon / GH files, when checked in, plug into `load_solomon`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Tuple

from engine_numpy.graph import Graph
from engine_numpy.loaders import load_8node, load_solomon, make_synthetic


InstanceFactory = Callable[[], Tuple[Graph, int, float]]

_INSTANCES: Dict[str, InstanceFactory] = {}


def register(name: str, factory: InstanceFactory) -> None:
    _INSTANCES[name] = factory


def get(name: str) -> Tuple[Graph, int, float]:
    if name not in _INSTANCES:
        raise KeyError(f"unknown instance {name}; known: {list(_INSTANCES)}")
    return _INSTANCES[name]()


def names() -> list[str]:
    return list(_INSTANCES)


# ── Built-in registrations ──────────────────────────────────────────────────
def _f_8node():
    g = load_8node()
    return g, 2, 8.0


def _f_synth(N: int, K: int, capacity: float, seed: int):
    def _f():
        return make_synthetic(n=N, seed=seed, capacity=capacity), K, capacity
    return _f


register("8node", _f_8node)
register("synth_25_k3",   _f_synth(25, 3, 40.0, seed=17))
register("synth_50_k5",   _f_synth(50, 5, 60.0, seed=23))
register("synth_100_k10", _f_synth(100, 10, 80.0, seed=31))

# Solomon / GH files, when present.
_INSTANCE_DIR = Path(__file__).resolve().parent.parent / "data" / "instances"
for _name in ("c101_25", "r101_25", "rc101_25", "c101_100"):
    p = _INSTANCE_DIR / f"{_name}.txt"
    if p.exists():
        register(f"solomon_{_name}", (lambda pp=p: lambda: (load_solomon(pp), 25, 200.0))())
