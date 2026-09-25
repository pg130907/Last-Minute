"""Module 2 — Multi-objective fitness with penalties, finite guard, TDVRP walk.

Z_search = w1·Σdist·x + w2·Σtau(t)·x + w3·Σemis·x
         + R_cap·Σmax(0, load_k − Q_k) + R_tw·Σmax(0, t_i − l_i) + w4·Σusage²
Z_report = w1·Σdist·x + w2·Σtau(t)·x + w3·Σemis·x        (no penalties)

Rules (spec §2, Module 2):
  1. Early arrival = WAIT to e_i, never penalized.
  2. Infeasible = penalized, NOT discarded.
  3. tau read at predicted arrival time at edge tail (time-dependent).
  4. Fitness ALWAYS finite (§2.5).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np

from common.bootstrap import get_logger
from engine_numpy.graph import BIG_FINITE, Graph

_log = get_logger("fitness")

DEFAULT_WEIGHTS = {"w1": 0.3, "w2": 0.5, "w3": 0.2, "w4": 0.0}
DEFAULT_R_CAP = 1.0e5
DEFAULT_R_TW = 1.0e5

# tau_lookup contract — Module 5b installs a schedule-aware version.
# Default here is time-INDEPENDENT: read whatever is on g.tau[i,j] right now.
TauLookup = Callable[[Graph, int, int, float], float]


def default_tau_lookup(g: Graph, i: int, j: int, t_depart: float) -> float:
    """Slice-independent BPR lookup. Module 5b replaces with a FIFO-safe schedule."""
    tau = float(g.tau[i, j])
    return tau if np.isfinite(tau) else BIG_FINITE


# ── Result containers ───────────────────────────────────────────────────────
@dataclass
class RouteMetrics:
    vehicle_id: int
    sequence: List[int]
    arrival_times: List[float]
    depart_times: List[float]
    load: float
    distance: float
    time: float
    emissions: float
    over_capacity: float          # max(0, load − Q_k)
    minutes_late: float           # Σ_i max(0, arrival − l_i) on this route


@dataclass
class Violation:
    kind: str                     # "late" | "overload"
    detail: dict


@dataclass
class ViolationBreakdown:
    late: List[dict] = field(default_factory=list)      # {customer, minutes_late, penalty_cost}
    overload: List[dict] = field(default_factory=list)  # {vehicle, amount, penalty_cost}


@dataclass
class FitnessResult:
    Z_search: float
    Z_report: float
    per_route: List[RouteMetrics]
    violations: ViolationBreakdown
    feasible: bool
    total_distance: float
    total_time: float
    total_emissions: float
    fleet_utilization: float


# ── Core route walk (FIFO-safe TDVRP schedule) ──────────────────────────────
def walk_route(
    g: Graph,
    seq: Sequence[int],
    tau_lookup: TauLookup = default_tau_lookup,
    start_time: float = 0.0,
) -> tuple[List[float], List[float], float, float, float, float]:
    """Walk one route and return (arrival, depart, distance, travel_time, emis, load).

    Wait-if-early: t_depart[next] = max(t_arrival[next], e_next) + service_next.
    Load counted at customers only (depot demand assumed 0).
    """
    seq = list(seq)
    arrivals: List[float] = [start_time]
    departs: List[float] = [start_time]
    dist_total = 0.0
    time_total = 0.0
    emis_total = 0.0
    load = 0.0

    early = g.tw_early
    service = g.service_time

    for hop in range(1, len(seq)):
        i, j = seq[hop - 1], seq[hop]
        tau_ij = tau_lookup(g, i, j, departs[-1])
        arrive = departs[-1] + tau_ij
        d_ij = float(g.dist[i, j])
        e_ij = float(g.emis[i, j])
        if not np.isfinite(d_ij):
            d_ij = BIG_FINITE
        if not np.isfinite(e_ij):
            e_ij = BIG_FINITE
        dist_total += d_ij
        time_total += tau_ij
        emis_total += e_ij

        # Wait-if-early (never penalized).
        e_j = float(early[j]) if early is not None else 0.0
        s_j = float(service[j]) if service is not None else 0.0
        depart = max(arrive, e_j) + s_j

        arrivals.append(arrive)
        departs.append(depart)

        if j != g.depot and g.demand is not None:
            load += float(g.demand[j])

    return arrivals, departs, dist_total, time_total, emis_total, load


# ── Top-level evaluator ─────────────────────────────────────────────────────
def evaluate(
    g: Graph,
    routes: Sequence[Sequence[int]],
    weights: Optional[dict] = None,
    R_cap: float = DEFAULT_R_CAP,
    R_tw: float = DEFAULT_R_TW,
    tau_lookup: TauLookup = default_tau_lookup,
    fleet_capacity: Optional[float] = None,
    start_time: float = 0.0,
) -> FitnessResult:
    """Compute Z_search / Z_report / per-route metrics / violation breakdown.

    `routes` — list of routes, each a node sequence [depot, c1, …, ck, depot].
    Vehicle ids are the position in `routes` (0-indexed → +1 for reporting).
    """
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)
    ws = w["w1"] + w["w2"] + w["w3"]
    if not np.isclose(ws, 1.0):
        raise ValueError(f"w1+w2+w3 must be 1.0, got {ws} ({w})")

    Q = fleet_capacity if fleet_capacity is not None else g.capacity
    late = g.tw_late

    per_route: List[RouteMetrics] = []
    viol = ViolationBreakdown()
    dist_sum = time_sum = emis_sum = 0.0
    over_penalty_sum = 0.0
    late_penalty_sum = 0.0
    load_sum = 0.0
    fleet_cap_sum = 0.0

    # Edge usage counter for w4·Σusage² (default off).
    edge_use: dict[tuple[int, int], int] = {}

    for k, seq in enumerate(routes):
        arrivals, departs, d, tt, ee, load = walk_route(
            g, seq, tau_lookup=tau_lookup, start_time=start_time
        )

        # Capacity violation.
        over = 0.0
        if Q is not None:
            over = max(0.0, load - float(Q))
            if over > 0:
                pen = R_cap * over
                over_penalty_sum += pen
                viol.overload.append(
                    {"vehicle": k + 1, "amount": over, "penalty_cost": pen}
                )
        fleet_cap_sum += float(Q) if Q is not None else 0.0

        # Time-window violations (late only; early = wait, no penalty).
        route_late = 0.0
        if late is not None:
            for pos, node in enumerate(seq):
                if node == g.depot:
                    continue
                l_i = float(late[node])
                lateness = max(0.0, arrivals[pos] - l_i)
                if lateness > 0:
                    pen = R_tw * lateness
                    late_penalty_sum += pen
                    route_late += lateness
                    viol.late.append(
                        {"customer": int(node), "minutes_late": lateness, "penalty_cost": pen}
                    )

        # Edge usage (for w4).
        if w["w4"] > 0:
            for a, b in zip(seq[:-1], seq[1:]):
                key = (min(a, b), max(a, b))
                edge_use[key] = edge_use.get(key, 0) + 1

        per_route.append(RouteMetrics(
            vehicle_id=k + 1,
            sequence=list(seq),
            arrival_times=arrivals,
            depart_times=departs,
            load=load,
            distance=d,
            time=tt,
            emissions=ee,
            over_capacity=over,
            minutes_late=route_late,
        ))

        dist_sum += d
        time_sum += tt
        emis_sum += ee
        load_sum += load

    # Finite guard on the aggregate — belt-and-braces; walk already substitutes.
    dist_sum = float(dist_sum) if np.isfinite(dist_sum) else BIG_FINITE
    time_sum = float(time_sum) if np.isfinite(time_sum) else BIG_FINITE
    emis_sum = float(emis_sum) if np.isfinite(emis_sum) else BIG_FINITE

    Z_report = w["w1"] * dist_sum + w["w2"] * time_sum + w["w3"] * emis_sum
    diversity_pen = 0.0
    if w["w4"] > 0 and edge_use:
        diversity_pen = w["w4"] * sum(u * u for u in edge_use.values())
    Z_search = Z_report + over_penalty_sum + late_penalty_sum + diversity_pen

    if not np.isfinite(Z_search):
        Z_search = BIG_FINITE
    if not np.isfinite(Z_report):
        Z_report = BIG_FINITE

    feasible = (over_penalty_sum == 0.0) and (late_penalty_sum == 0.0)
    fleet_util = (load_sum / fleet_cap_sum) if fleet_cap_sum > 0 else 0.0

    return FitnessResult(
        Z_search=float(Z_search),
        Z_report=float(Z_report),
        per_route=per_route,
        violations=viol,
        feasible=feasible,
        total_distance=float(dist_sum),
        total_time=float(time_sum),
        total_emissions=float(emis_sum),
        fleet_utilization=float(fleet_util),
    )
