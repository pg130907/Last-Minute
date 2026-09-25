"""Module 4a — Diversity control: chaos escape + phase rescatter.

Triggered on stall: mean pairwise angle distance AND fitness variance both
below thresholds for `L_stall` consecutive iters. When that fires:

  - Chaos escape: map gbest angles → [0,1] fraction, iterate the logistic
    map y ← 4y(1−y) `T_CHAOS` times, map back, pick the best decoded state,
    overwrite ONE RANDOM particle with it.
  - Phase rescatter (per-angle border-mutation, always running): any angle
    with sin²θ < eps or > 1−eps for `L_sat` consecutive iters is redrawn
    from U(0, 2π). Fights boundary pile-up in the sin²-decode.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from common.bootstrap import get_logger

_log = get_logger("diversity")
_TWO_PI = 2.0 * np.pi


@dataclass
class DiversityState:
    """Held across iterations by the QPSO loop.

    `sat_counters` — per-angle consecutive-saturated count, shape matches theta.
    `stall_count` — consecutive iters both dispersion and fitness were flat.
    `last_gbest` — most recent gbest value, for the fitness-flat check.
    """
    sat_counters: np.ndarray
    stall_count: int = 0
    last_gbest: float = float("inf")
    n_chaos_fired: int = 0
    n_rescatter_fired: int = 0

    @classmethod
    def new(cls, theta_shape: tuple) -> "DiversityState":
        return cls(sat_counters=np.zeros(theta_shape, dtype=np.int32))


def _mean_pairwise_distance(theta: np.ndarray) -> float:
    """Mean L2 distance in angle space, sampled cheaply.

    For small M (~20) exact O(M²) is fine; we compute against the swarm mean
    and use var-times-2 as a proxy (equivalent up to constants for centered
    Gaussians and cheap to compute).
    """
    flat = theta.reshape(theta.shape[0], -1)
    center = flat.mean(axis=0, keepdims=True)
    return float(np.sqrt(((flat - center) ** 2).sum(axis=1).mean()))


def stalled(
    state: DiversityState,
    theta_swarm: np.ndarray,
    gbest_val: float,
    L_stall: int = 15,
    dist_eps: float = 0.05,
    fit_eps: float = 1e-4,
) -> bool:
    """Increment/reset stall counter, return True when threshold crossed."""
    dispersion = _mean_pairwise_distance(theta_swarm)
    fit_flat = abs(gbest_val - state.last_gbest) < fit_eps
    if dispersion < dist_eps and fit_flat:
        state.stall_count += 1
    else:
        state.stall_count = 0
    state.last_gbest = gbest_val
    return state.stall_count >= L_stall


def chaos_escape(gbest_theta: np.ndarray, T_chaos: int = 20) -> list[np.ndarray]:
    """Run the logistic map T_chaos steps on gbest, in angle-fraction space.

    Returns a list of candidate theta arrays — caller decodes each, keeps the
    best, and installs it into one random particle. Doing the evaluation in
    the caller keeps this module free of graph/fitness dependencies.
    """
    y = (np.mod(gbest_theta, _TWO_PI) / _TWO_PI).astype(np.float64)
    # Avoid the logistic map's fixed points {0, 0.5, 0.75}.
    y = np.clip(y, 1e-3, 1.0 - 1e-3)
    candidates: list[np.ndarray] = []
    for _ in range(T_chaos):
        y = 4.0 * y * (1.0 - y)
        y = np.clip(y, 1e-3, 1.0 - 1e-3)
        candidates.append((y * _TWO_PI).astype(np.float32).copy())
    return candidates


def apply_chaos_and_rescatter(
    theta: np.ndarray,
    state: DiversityState,
    rng: np.random.Generator,
    *,
    eps_sat: float = 1e-3,
    L_sat: int = 10,
) -> np.ndarray:
    """Per-angle phase rescatter (border mutation). Chaos is applied in the
    caller because it needs decode/evaluate.
    """
    sin_sq = np.sin(theta) ** 2
    saturated = (sin_sq < eps_sat) | (sin_sq > 1.0 - eps_sat)
    state.sat_counters = np.where(saturated, state.sat_counters + 1, 0)
    mask = state.sat_counters >= L_sat
    if mask.any():
        state.n_rescatter_fired += int(mask.sum())
        new_theta = rng.uniform(0.0, _TWO_PI, size=theta.shape).astype(theta.dtype)
        theta = np.where(mask, new_theta, theta)
        state.sat_counters = np.where(mask, 0, state.sat_counters)
    return theta
