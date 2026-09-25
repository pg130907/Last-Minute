"""Common shim for baselines — normalize result shape and route validation.

Every baseline returns a `BaselineResult`. The benchmark harness aggregates
these across runs and reports mean±std alongside our engine's numbers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class BaselineResult:
    name: str
    best_cost: float          # Z_report — the un-penalized, judge-facing cost
    best_cost_search: float   # Z_search  — the penalized objective (for infeasible baselines)
    best_routes: List[List[int]]
    wall_time_s: float
    feasible: bool
    seed: int = 0
    iterations: int = 0
    convergence: List[float] = field(default_factory=list)
    notes: str = ""
