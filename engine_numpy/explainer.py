"""Module 4d — Solution explainer.

For each route, diff the chosen route against the shortest-DISTANCE route on
the same customers (a nearest-neighbor + 2-opt-ish baseline is sufficient —
we only need a REFERENCE, not a proven optimum). Identify which edges the
chosen route AVOIDED that the shortest-distance route uses, look them up in
the event schedule and current BPR tau, and emit one short human sentence
per route.

Example (spec §4d):
    "Route 2 avoids edges 14→17,17→22 due to predicted congestion from
     event 'School Dismissal' @15:00; +1.2 km but −8.3 min."

Trivially cheap; turns the optimizer from opaque to self-explaining.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from engine_numpy.fitness import RouteMetrics
from engine_numpy.graph import BIG_FINITE, Graph
from engine_numpy.predictor import EventShock


def _nn_tour(g: Graph, customers: Sequence[int], depot: int) -> List[int]:
    """Greedy nearest-neighbor tour on `customers`, starting/ending at depot."""
    remaining = set(int(c) for c in customers)
    seq = [depot]
    cur = depot
    while remaining:
        nxt = min(remaining, key=lambda c: float(g.dist[cur, c]))
        seq.append(nxt)
        remaining.discard(nxt)
        cur = nxt
    seq.append(depot)
    return seq


def _tour_distance(g: Graph, seq: Sequence[int]) -> float:
    total = 0.0
    for a, b in zip(seq[:-1], seq[1:]):
        d = float(g.dist[a, b])
        total += d if np.isfinite(d) else BIG_FINITE
    return total


def _tour_time(g: Graph, seq: Sequence[int]) -> float:
    total = 0.0
    for a, b in zip(seq[:-1], seq[1:]):
        t = float(g.tau[a, b])
        total += t if np.isfinite(t) else BIG_FINITE
    return total


def _edges_of(seq: Sequence[int]) -> set[tuple[int, int]]:
    return {(int(a), int(b)) for a, b in zip(seq[:-1], seq[1:])}


def _event_covering_edge(edge: tuple[int, int], events: Sequence[EventShock]) -> Optional[EventShock]:
    for ev in events:
        if edge in {(int(i), int(j)) for i, j in ev.edges}:
            return ev
    return None


def explain_route(
    g: Graph,
    route: RouteMetrics,
    events: Sequence[EventShock] = (),
    depot: int = 0,
) -> str:
    """Return one-sentence reasoning for this route."""
    customers = [n for n in route.sequence if n != depot]
    if not customers:
        return f"Route {route.vehicle_id}: empty (no customers assigned)."

    baseline = _nn_tour(g, customers, depot)
    base_dist = _tour_distance(g, baseline)
    base_time = _tour_time(g, baseline)

    chosen_edges = _edges_of(route.sequence)
    baseline_edges = _edges_of(baseline)
    avoided = list(baseline_edges - chosen_edges)

    d_dist = route.distance - base_dist
    d_time = route.time - base_time

    # Look for event-covered avoided edges.
    event_hits: dict[str, list[tuple[int, int]]] = {}
    for e in avoided:
        ev = _event_covering_edge(e, events)
        if ev is not None:
            event_hits.setdefault(ev.name, []).append(e)

    parts = [f"Route {route.vehicle_id}"]
    if event_hits:
        ev_name, hit_edges = next(iter(event_hits.items()))
        edge_str = ",".join(f"{a}→{b}" for a, b in hit_edges[:3])
        parts.append(f"avoids {edge_str} due to event '{ev_name}'")
    elif avoided:
        sample = avoided[0]
        parts.append(f"reroutes around {sample[0]}→{sample[1]} (higher predicted travel time)")
    else:
        parts.append("follows the shortest-distance path")

    if abs(d_dist) > 1e-3 or abs(d_time) > 1e-3:
        dist_str = f"{'+' if d_dist >= 0 else ''}{d_dist:.1f} km"
        time_str = f"{'+' if d_time >= 0 else ''}{d_time:.1f} min"
        parts.append(f"({dist_str}, {time_str} vs nearest-neighbor baseline)")

    parts.append(
        f"— load {route.load:.1f}/{route.load + max(0, route.over_capacity):.1f}, "
        f"{len(customers)} stops"
    )
    return " ".join(parts) + "."


def explain_solution(
    g: Graph,
    routes: Sequence[RouteMetrics],
    events: Sequence[EventShock] = (),
    depot: int = 0,
) -> List[str]:
    return [explain_route(g, r, events=events, depot=depot) for r in routes]
