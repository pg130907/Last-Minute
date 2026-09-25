"""Module 5b — Predictive / TDVRP.

Three concerns bundled here:
  1. `TrafficSchedule` — per-edge, per-time-bucket tau plus a FIFO-safe
     `tau_lookup(g, i, j, t_depart)` implementing spec §5b's schedule walk.
  2. `EventShock` (spec §4c schema) — schedules extra volume on named edges
     over a time interval; TDVRP then routes around the future jam.
  3. Predictors — `arima` (statsmodels), `none` (identity/last-value). STGCN
     is stretch; not built here per user instructions.

FIFO enforcement (spec §5b test): tau_lookup(t) is post-processed so
`arrive(t) = t + tau_lookup(t)` is non-decreasing in `t`. Concretely we
build a monotone envelope of arrivals over the discrete bucket grid and
interpolate on that envelope, so leaving later never lands earlier — the
exact invariant `test_fifo.py` asserts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np

from common.bootstrap import get_logger
from engine_numpy.graph import BIG_FINITE, BPR_ALPHA, BPR_BETA, Graph

_log = get_logger("predictor")

MIN_PER_DAY = 24 * 60


# ── Event shocks (spec §4c/§5b) ─────────────────────────────────────────────
@dataclass
class EventShock:
    name: str
    edges: List[tuple[int, int]]     # (i, j) pairs; each shocks tau on that edge
    start_min: float                 # minutes since midnight
    end_min: float
    extra_vol: float                 # added to background vol on those edges

    def active_at(self, t_min: float) -> bool:
        return self.start_min <= t_min <= self.end_min


# ── Traffic schedule with FIFO-safe tau_lookup ──────────────────────────────
@dataclass
class TrafficSchedule:
    """Time-bucketed (bucket_min-wide) tau slices per edge.

    tau_slices[s, i, j] = travel time (min) on edge i→j when *entering* it
    within time bucket s. Non-edges hold BIG_FINITE.

    Enforces FIFO by construction: the internal `_arrive_grid[s, i, j]` is
    the cumulative-max of `(bucket_start + tau_slices)` across s per edge,
    and `tau_lookup` interpolates against that envelope.
    """
    tau_slices: np.ndarray            # (S, N, N) float32 — minutes
    bucket_min: int
    n_buckets: int
    dist: np.ndarray                  # (N, N) — needed for detour computations
    _arrive_grid: np.ndarray = field(init=False, repr=False)
    _bucket_starts: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self._bucket_starts = (np.arange(self.n_buckets) * self.bucket_min).astype(np.float32)
        # arrive[s, i, j] = start_s + tau_slices[s, i, j]
        arrive = self._bucket_starts[:, None, None] + self.tau_slices
        # FIFO envelope: monotone-non-decreasing cumulative max down the time axis.
        # This is the tightest FIFO envelope; the caller pays at most one
        # bucket's worth of "extra tau" at rare non-monotone junctions.
        self._arrive_grid = np.maximum.accumulate(arrive, axis=0).astype(np.float32)

    @classmethod
    def from_bpr(
        cls,
        g: Graph,
        vol_series: np.ndarray,       # (S, N, N) or (S, n_edge) mapped to (N,N)
        bucket_min: int,
    ) -> "TrafficSchedule":
        """Given a time-bucketed background volume schedule, run BPR per slice.

        vol_series[s, i, j] = background volume on edge i→j at bucket s.
        Non-edges remain at BIG_FINITE tau.
        """
        S, N, _ = vol_series.shape
        tau0 = g.tau0
        cap = g.cap
        adj = g.adj_mask
        tau_slices = np.zeros((S, N, N), dtype=np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            for s in range(S):
                ratio = np.where(cap > 0, vol_series[s] / cap, 0.0)
                tau = tau0 * (1.0 + BPR_ALPHA * ratio ** BPR_BETA)
                tau = np.where(adj & np.isfinite(tau), tau, BIG_FINITE)
                tau_slices[s] = tau.astype(np.float32)
        return cls(tau_slices=tau_slices, bucket_min=bucket_min, n_buckets=S,
                   dist=g.dist.copy())

    def apply_events(self, g: Graph, events: Sequence[EventShock]) -> None:
        """Re-derive `tau_slices` on the specified edges within [start, end]
        buckets, adding `extra_vol` to the background before re-applying BPR.
        Rebuild the FIFO envelope.
        """
        if not events:
            return
        tau0 = g.tau0
        cap = g.cap
        adj = g.adj_mask
        for ev in events:
            s_lo = max(0, int(ev.start_min // self.bucket_min))
            s_hi = min(self.n_buckets - 1, int(ev.end_min // self.bucket_min))
            for i, j in ev.edges:
                if not adj[i, j]:
                    _log.warning("event edge not present", extra={"i": i, "j": j, "ev": ev.name})
                    continue
                for s in range(s_lo, s_hi + 1):
                    # Recover base_vol via inversion of BPR on the current slice.
                    ratio_cur = ((self.tau_slices[s, i, j] / tau0[i, j]) - 1.0) / BPR_ALPHA
                    ratio_cur = max(0.0, float(ratio_cur))
                    base_vol = (ratio_cur ** (1.0 / BPR_BETA)) * cap[i, j]
                    new_vol = base_vol + ev.extra_vol
                    ratio_new = new_vol / cap[i, j] if cap[i, j] > 0 else 0.0
                    tau_new = tau0[i, j] * (1.0 + BPR_ALPHA * ratio_new ** BPR_BETA)
                    self.tau_slices[s, i, j] = float(tau_new)
        # Rebuild the FIFO envelope after the mutation.
        arrive = self._bucket_starts[:, None, None] + self.tau_slices
        self._arrive_grid = np.maximum.accumulate(arrive, axis=0).astype(np.float32)

    def tau_lookup(self, g: Graph, i: int, j: int, t_depart: float) -> float:
        """FIFO-safe lookup — arrival(t_depart) = t_depart + tau."""
        # Interpolate against the arrive-grid envelope.
        t = float(t_depart)
        if not np.isfinite(t):
            return BIG_FINITE
        if t <= self._bucket_starts[0]:
            arrival = float(self._arrive_grid[0, i, j])
        elif t >= self._bucket_starts[-1]:
            arrival = float(self._arrive_grid[-1, i, j])
        else:
            k = int(np.searchsorted(self._bucket_starts, t, side="right") - 1)
            k = max(0, min(k, self.n_buckets - 2))
            t_lo = float(self._bucket_starts[k])
            t_hi = float(self._bucket_starts[k + 1])
            a_lo = float(self._arrive_grid[k, i, j])
            a_hi = float(self._arrive_grid[k + 1, i, j])
            frac = (t - t_lo) / (t_hi - t_lo)
            # Linearly interpolate arrival, but also enforce arrival >= t + eps
            # (can never arrive before you left).
            arrival = a_lo + frac * (a_hi - a_lo)
        tau = max(arrival - t, 0.0)   # arrival - t = travel time
        if not np.isfinite(tau) or tau >= BIG_FINITE:
            return BIG_FINITE
        return tau


def make_tau_lookup(schedule: TrafficSchedule) -> Callable:
    """Adapter — matches the `TauLookup` signature expected by fitness.walk_route."""
    def _lookup(g, i, j, t_depart):
        return schedule.tau_lookup(g, i, j, t_depart)
    return _lookup


# ── ARIMA predictor (statsmodels) ───────────────────────────────────────────
def arima_forecast(series: np.ndarray, horizon: int, order=(2, 0, 1)) -> np.ndarray:
    """Fit ARIMA to a 1D series and forecast `horizon` steps ahead.

    Statsmodels raises on constant / near-constant series; guard by falling
    back to the last observed value.
    """
    from statsmodels.tsa.arima.model import ARIMA

    series = np.asarray(series, dtype=np.float64)
    if np.std(series) < 1e-6:
        return np.full(horizon, float(series[-1]), dtype=np.float32)
    try:
        model = ARIMA(series, order=order).fit(method_kwargs={"warn_convergence": False})
        fcst = model.forecast(steps=horizon)
        return np.clip(np.asarray(fcst, dtype=np.float32), 0.0, None)
    except Exception as e:
        _log.warning("arima failed; using last-value", extra={"err": type(e).__name__})
        return np.full(horizon, float(series[-1]), dtype=np.float32)


def forecast_volumes(vol_history: np.ndarray, horizon: int, order=(2, 0, 1)) -> np.ndarray:
    """Forecast volumes for every edge in `vol_history` (shape (T_hist, N_edges)).

    Returns (horizon, N_edges) forecasts.
    """
    T, E = vol_history.shape
    out = np.zeros((horizon, E), dtype=np.float32)
    for e in range(E):
        out[:, e] = arima_forecast(vol_history[:, e], horizon, order=order)
    return out
