"""Module 1 — Graph: dense float32 matrices, BPR dynamic weights, emissions, LCC.

Layer 1 of the pipeline. Every downstream module reads dist/tau/emis off this
object; only `update_weights(t)` mutates its state.

Scope declaration: background `vol` is EXOGENOUS. The fleet is a price-taker
against city-wide traffic. Optional `feedback_iterations>0` will add fleet
flow back into `vol` and re-solve (fixed-point) — default OFF.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import networkx as nx
import numpy as np

from common.bootstrap import get_logger

_log = get_logger("graph")

# ── Constants pulled from spec §2, §6 ───────────────────────────────────────
BPR_ALPHA = 0.15
BPR_BETA = 4.0
EMIS_A = 100.0    # g CO2/km, idle-ish baseline
EMIS_B = 800.0    # g CO2 · (km/h)/km — dominates at low speed
EMIS_C = 0.015    # g CO2 / (km · (km/h)²) — drag at highway speed
FLAT_FACTOR = 150.0
BIG_FINITE = 1.0e9       # non-edge sentinel used by fitness (§2.5)
DEFAULT_FREE_SPEED = 50.0  # km/h — for benchmark instances that only give dist


def emission_rate(speed_kmh: np.ndarray) -> np.ndarray:
    """COPERT-style g CO2/km curve. Vectorized; safe on zero speed → returns inf."""
    v = np.asarray(speed_kmh, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = EMIS_A + EMIS_B / v + EMIS_C * v * v
    return rate.astype(np.float32)


@dataclass
class Graph:
    """Dense, symmetric-or-directed graph shared by both engines.

    Non-edges: `adj_mask=False`, `dist=inf`, `tau0=inf`. The fitness finite-guard
    (§2.5) converts any inf lookup into `BIG_FINITE` so mbest = mean(pbest)
    stays finite even when a particle transiently crosses a non-edge.
    """
    dist: np.ndarray            # (N,N) km
    tau0: np.ndarray            # (N,N) min, free-flow
    cap: np.ndarray             # (N,N) veh/hr
    vol: np.ndarray             # (N,N) veh/hr (exogenous background)
    adj_mask: np.ndarray        # (N,N) bool
    speed0: Optional[np.ndarray] = None   # (N,N) km/h — presence flips emissions to speed-dependent
    depot: int = 0

    # VRPTW customer attrs (optional; None for pure CVRP like 8-node)
    demand: Optional[np.ndarray] = None       # (N,) load per stop
    tw_early: Optional[np.ndarray] = None     # (N,) minutes
    tw_late: Optional[np.ndarray] = None      # (N,)
    service_time: Optional[np.ndarray] = None # (N,)
    capacity: Optional[float] = None          # per-vehicle Q_k (uniform fleet)

    # Live state (populated by update_weights)
    tau: np.ndarray = field(init=False)
    speed: np.ndarray = field(init=False)
    emis: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        # Cast to float32 as per §0 precision rule.
        self.dist = np.ascontiguousarray(self.dist, dtype=np.float32)
        self.tau0 = np.ascontiguousarray(self.tau0, dtype=np.float32)
        self.cap = np.ascontiguousarray(self.cap, dtype=np.float32)
        self.vol = np.ascontiguousarray(self.vol, dtype=np.float32)
        self.adj_mask = np.ascontiguousarray(self.adj_mask, dtype=bool)
        if self.speed0 is not None:
            self.speed0 = np.ascontiguousarray(self.speed0, dtype=np.float32)
        self.update_weights(t=0.0)

    @property
    def n(self) -> int:
        return int(self.dist.shape[0])

    def update_weights(self, t: float = 0.0) -> None:
        """BPR refresh — tau, speed, emis derived from current vol.

        `t` is currently informational (single-slice BPR). The time-dependent
        predictor (Module 5b) is what actually swaps in a schedule of vol/tau
        slices; this method just applies the BPR curve to whatever vol is
        installed right now.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(self.cap > 0, self.vol / self.cap, 0.0).astype(np.float32)
            self.tau = (self.tau0 * (1.0 + BPR_ALPHA * ratio ** BPR_BETA)).astype(np.float32)
            # speed[km/h] = dist[km] / tau[min] * 60. tau=inf on non-edges → speed=0.
            self.speed = np.where(
                (self.tau > 0) & np.isfinite(self.tau) & (self.dist > 0),
                self.dist / np.where(self.tau > 0, self.tau, 1.0) * 60.0,
                0.0,
            ).astype(np.float32)

        if self.speed0 is not None:
            # Speed-dependent COPERT. Zero-speed / non-edge / diagonal are
            # handled by the finite guard below; suppress the transient
            # inf/NaN warnings from a·∞ or ∞·0 in intermediate arithmetic.
            with np.errstate(divide="ignore", invalid="ignore"):
                rate = emission_rate(self.speed)
                emis = rate * self.dist
        else:
            emis = self.dist * FLAT_FACTOR

        emis = np.where(self.adj_mask & np.isfinite(emis), emis, BIG_FINITE)
        self.emis = emis.astype(np.float32)

    # ── Convenience --------------------------------------------------------
    def add_fleet_volume(self, fleet_vol: np.ndarray) -> None:
        """Fixed-point feedback (feature-flagged, default OFF)."""
        self.vol = (self.vol + fleet_vol).astype(np.float32)
        self.update_weights()


# ── Largest Connected Component (§2.4 step 2) ───────────────────────────────
def extract_lcc(g: Graph, strongly: bool = True) -> tuple[Graph, np.ndarray]:
    """Keep only the largest (strongly-)connected component.

    Returns (new_graph, kept_indices) where kept_indices are the ORIGINAL node
    ids retained. Depot is remapped: if the original depot survives, it keeps
    id 0 in the new graph; otherwise the highest-degree surviving node becomes
    the new depot. That last case should never happen on real inputs — we
    warn loudly if it does.
    """
    n = g.n
    if strongly:
        G = nx.from_numpy_array(g.adj_mask.astype(np.uint8), create_using=nx.DiGraph)
        comps = list(nx.strongly_connected_components(G))
    else:
        G = nx.from_numpy_array(g.adj_mask.astype(np.uint8), create_using=nx.Graph)
        comps = list(nx.connected_components(G))

    if not comps:
        raise ValueError("graph has no components (adj_mask all False)")

    largest = max(comps, key=len)
    kept = np.array(sorted(largest), dtype=np.int64)
    if len(kept) == n:
        _log.info("lcc: graph already single-component", extra={"n": n})
        return g, kept

    dropped = n - len(kept)
    _log.warning(
        "lcc: dropping isolated/unreachable nodes",
        extra={"n_original": n, "n_kept": len(kept), "n_dropped": dropped},
    )

    # Remap depot.
    if g.depot in largest:
        new_depot_pos = int(np.searchsorted(kept, g.depot))
    else:
        degrees = g.adj_mask[np.ix_(kept, kept)].sum(axis=1)
        new_depot_pos = int(np.argmax(degrees))
        _log.warning(
            "lcc: original depot was dropped; picking highest-degree replacement",
            extra={"orig_depot": g.depot, "new_depot_pos": new_depot_pos},
        )

    def _sub(mat: np.ndarray) -> np.ndarray:
        return mat[np.ix_(kept, kept)]

    def _sub1(vec: Optional[np.ndarray]) -> Optional[np.ndarray]:
        return None if vec is None else vec[kept]

    new = Graph(
        dist=_sub(g.dist),
        tau0=_sub(g.tau0),
        cap=_sub(g.cap),
        vol=_sub(g.vol),
        adj_mask=_sub(g.adj_mask),
        speed0=None if g.speed0 is None else _sub(g.speed0),
        depot=new_depot_pos,
        demand=_sub1(g.demand),
        tw_early=_sub1(g.tw_early),
        tw_late=_sub1(g.tw_late),
        service_time=_sub1(g.service_time),
        capacity=g.capacity,
    )
    return new, kept


def build_from_dist(
    dist: np.ndarray,
    *,
    depot: int = 0,
    default_speed: float = DEFAULT_FREE_SPEED,
    cap_default: float = 1.0e6,
    speed_dependent: bool = False,
    demand: Optional[np.ndarray] = None,
    tw_early: Optional[np.ndarray] = None,
    tw_late: Optional[np.ndarray] = None,
    service_time: Optional[np.ndarray] = None,
    capacity: Optional[float] = None,
) -> Graph:
    """Build a Graph from a (possibly incomplete) distance matrix.

    Non-edges: `dist=inf` OR `dist<=0` off-diagonal. Diagonal is forced to 0.
    tau0 derived from dist at `default_speed`. Capacity defaults to `cap_default`
    everywhere (effectively unlimited) — override per-edge for realistic BPR.
    """
    d = np.array(dist, dtype=np.float64, copy=True)
    n = d.shape[0]
    assert d.shape == (n, n), f"dist must be square, got {d.shape}"
    np.fill_diagonal(d, 0.0)

    off_diag = ~np.eye(n, dtype=bool)
    adj = off_diag & np.isfinite(d) & (d > 0)

    # tau0 in minutes: dist / speed * 60. Non-edges → inf.
    with np.errstate(divide="ignore", invalid="ignore"):
        tau0 = np.where(adj, d / default_speed * 60.0, np.inf)
    # Fill non-edges' dist with inf for consistency with adj_mask=False.
    dist_out = np.where(adj | np.eye(n, dtype=bool), d, np.inf)

    cap = np.where(adj, cap_default, 0.0)
    vol = np.zeros((n, n), dtype=np.float32)
    speed0 = np.where(adj, float(default_speed), 0.0) if speed_dependent else None

    return Graph(
        dist=dist_out.astype(np.float32),
        tau0=tau0.astype(np.float32),
        cap=cap.astype(np.float32),
        vol=vol,
        adj_mask=adj,
        speed0=speed0,
        depot=depot,
        demand=None if demand is None else np.asarray(demand, dtype=np.float32),
        tw_early=None if tw_early is None else np.asarray(tw_early, dtype=np.float32),
        tw_late=None if tw_late is None else np.asarray(tw_late, dtype=np.float32),
        service_time=None if service_time is None else np.asarray(service_time, dtype=np.float32),
        capacity=capacity,
    )
