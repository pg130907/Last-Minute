"""Module 5a — Warm start on live traffic update.

Spec §5a (verbatim intent):

    on traffic_update(t):
        graph.update_weights(t)             # refresh tau/speed/emis
        re-evaluate fitness of ALL particles # positions unchanged
        RESET pbest VALUES to current fitness  # stale bests invalid
        KEEP pbest POSITIONS
        alpha ← min(1.0, alpha * 1.5)        # MULTIPLICATIVE boost (REVISED)
        continue

Reset-values-keep-positions is the whole point: after a traffic shock the
old pbest costs no longer describe the same landscape. If you leave them
alone the swarm keeps chasing a phantom optimum. If you reset positions
too, you throw away the swarm's structural learning. Resetting only the
scalar values re-anchors the search in the new landscape while preserving
every angle vector.

Multiplicative α × 1.5 (spec REVISED): the additive `+0.25` variant was
accidentally time-dependent — clamped early, strong late. Multiplicative
gives a consistent proportional exploration kick regardless of when the
event fires.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from common.bootstrap import get_logger
from engine_numpy.encoding import decode
from engine_numpy.fitness import (
    DEFAULT_R_CAP,
    DEFAULT_R_TW,
    FitnessResult,
    default_tau_lookup,
    evaluate,
)
from engine_numpy.graph import Graph

_log = get_logger("warmstart")


@dataclass
class SwarmState:
    """Mutable QPSO state — passed by reference so warm_start can rewrite it
    in place. Held by run_qpso across iterations.
    """
    theta: np.ndarray                    # (M, 2, Nc)
    pbest_theta: np.ndarray              # (M, 2, Nc)
    pbest_val: np.ndarray                # (M,) — Z_search
    pbest_report: np.ndarray             # (M,) — Z_report
    pbest_routes: List[Optional[list]]
    pbest_fitness: List[Optional[FitnessResult]]
    gbest_theta: np.ndarray              # (2, Nc)
    gbest_val: float
    gbest_report: float
    gbest_routes: list
    gbest_fitness: Optional[FitnessResult]
    alpha: float
    K: int
    depot: int


def apply_warm_start(
    state: SwarmState,
    g: Graph,
    at_time_min: float,
    *,
    weights: Optional[dict] = None,
    R_cap: float = DEFAULT_R_CAP,
    R_tw: float = DEFAULT_R_TW,
    fleet_capacity: Optional[float] = None,
    tau_lookup: Callable = default_tau_lookup,
    alpha_boost: float = 1.5,
) -> SwarmState:
    """Refresh weights (caller-installed schedule / event / BPR), re-score all
    particles in place, reset pbest VALUES, keep pbest POSITIONS, and multiply
    α by `alpha_boost` (clamped to `alpha_cap`).
    """
    g.update_weights(t=at_time_min)

    M = state.theta.shape[0]
    new_gbest_val = float("inf")
    new_gbest_idx = 0
    for i in range(M):
        routes, _, _ = decode(state.theta[i, 0], state.theta[i, 1], state.K, depot=state.depot)
        res = evaluate(
            g, routes, weights=weights, R_cap=R_cap, R_tw=R_tw,
            tau_lookup=tau_lookup, fleet_capacity=fleet_capacity,
        )
        # RESET pbest_val to current — stale bests from the old landscape are
        # no longer credible.
        state.pbest_val[i] = res.Z_search
        state.pbest_report[i] = res.Z_report
        state.pbest_routes[i] = routes
        state.pbest_fitness[i] = res
        # pbest_theta is UNCHANGED — position knowledge kept.
        if res.Z_search < new_gbest_val:
            new_gbest_val = res.Z_search
            new_gbest_idx = i

    state.gbest_val = float(new_gbest_val)
    state.gbest_report = float(state.pbest_report[new_gbest_idx])
    state.gbest_theta = state.pbest_theta[new_gbest_idx].copy()
    state.gbest_routes = list(state.pbest_routes[new_gbest_idx])
    state.gbest_fitness = state.pbest_fitness[new_gbest_idx]

    # Multiplicative boost on the state's alpha carrier (spec §5a REVISED).
    # The QPSO loop applies the effective-α cap when composing base_alpha *
    # this carrier — capping here would collapse the boost when the carrier
    # is already 1.0, defeating the whole mechanism.
    old_alpha = state.alpha
    state.alpha = old_alpha * alpha_boost
    _log.info(
        "warm-start applied",
        extra={
            "at_time_min": at_time_min,
            "alpha_before": round(old_alpha, 4),
            "alpha_after": round(state.alpha, 4),
            "gbest_after": round(state.gbest_val, 4),
        },
    )
    return state
