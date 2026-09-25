"""Synthetic 24h volume generator (spec §5b default training source).

Bimodal traffic — morning peak around 08:00, evening around 18:00 — plus
low-amplitude noise. Deterministic in `seed` so the predictor is testable
out of the box. Time index is minutes-since-midnight; output values are
veh/hr (background flow on an edge, to be plugged into BPR).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


DEFAULT_SLOT_MIN = 15         # 96 samples per day
MIN_PER_DAY = 24 * 60


@dataclass
class VolumeSeries:
    times_min: np.ndarray     # (T,) — bucket-start times, minutes
    volumes: np.ndarray       # (n_edges, T) — veh/hr per edge per bucket
    slot_min: int             # bucket width in minutes


def _bimodal_shape(times_min: np.ndarray, morning: float, evening: float,
                   sigma_min: float) -> np.ndarray:
    """Two Gaussians centred at rush hours. Unit-max shape ∈ [0, 1]."""
    def _g(mu):
        return np.exp(-((times_min - mu) ** 2) / (2 * sigma_min ** 2))
    y = _g(morning) + _g(evening)
    return y / y.max()


def generate_series(
    n_edges: int,
    seed: int = 0,
    slot_min: int = DEFAULT_SLOT_MIN,
    base_vol: float = 200.0,
    peak_vol: float = 1400.0,
    sigma_min: float = 75.0,
    morning: float = 8 * 60,
    evening: float = 18 * 60,
    noise_frac: float = 0.10,
) -> VolumeSeries:
    """Generate a synthetic 24h volume series for `n_edges` edges.

    Returns a VolumeSeries; caller aligns edges → matrix positions.
    """
    rng = np.random.default_rng(seed)
    n_slots = MIN_PER_DAY // slot_min
    times = (np.arange(n_slots) * slot_min).astype(np.float32)

    shape = _bimodal_shape(times, morning, evening, sigma_min)   # (T,)
    per_edge_bias = rng.uniform(0.6, 1.4, size=n_edges).astype(np.float32)
    vols = base_vol + (peak_vol - base_vol) * shape[None, :] * per_edge_bias[:, None]
    # Per-slot multiplicative noise.
    noise = rng.normal(1.0, noise_frac, size=vols.shape).clip(0.5, 1.5)
    vols = np.maximum(vols * noise, 0.0).astype(np.float32)
    return VolumeSeries(times_min=times, volumes=vols, slot_min=slot_min)
