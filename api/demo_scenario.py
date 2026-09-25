"""Bundled demo scenario (spec §5e stub — the real bundled scenario is a
Module-5 deliverable; this is the API-side canned instance the UI loads on
first paint).

A ~20-customer city grid around a plausible urban centre. Coordinates are
in a small area so Euclidean-on-lat/lng is a fine distance proxy for the
demo; the API adapter will materialize the dense-matrix graph from these.
"""
from __future__ import annotations

import math
from typing import Any

# Depot at a plausible urban centre; customers scattered in a ~5 km radius.
_CENTER_LAT = 12.9716
_CENTER_LNG = 77.5946
_RADIUS_KM = 4.0
_KM_PER_DEG_LAT = 111.0


def _lat_lng_offset(dx_km: float, dy_km: float) -> tuple[float, float]:
    dlat = dy_km / _KM_PER_DEG_LAT
    dlng = dx_km / (_KM_PER_DEG_LAT * math.cos(math.radians(_CENTER_LAT)))
    return _CENTER_LAT + dlat, _CENTER_LNG + dlng


def demo_solve_request(*, with_events: bool = False, T: int = 250) -> dict[str, Any]:
    """Return a SolveRequest-shaped dict the UI can POST to /solve verbatim."""
    import random
    rng = random.Random(42)

    nodes = [{"id": 0, "lat": _CENTER_LAT, "lng": _CENTER_LNG,
              "demand": 0.0, "service_time": 0.0}]
    N_CUSTOMERS = 20
    for i in range(1, N_CUSTOMERS + 1):
        theta = rng.uniform(0, 2 * math.pi)
        r = rng.uniform(0.6, _RADIUS_KM)
        lat, lng = _lat_lng_offset(r * math.cos(theta), r * math.sin(theta))
        nodes.append({
            "id": i,
            "lat": lat,
            "lng": lng,
            "demand": rng.choice([2.0, 3.0, 5.0, 7.0]),
            "tw_early": rng.choice([0.0, 30.0, 60.0, 120.0]),
            "tw_late": rng.choice([180.0, 240.0, 360.0, 480.0]),
            "service_time": 5.0,
        })

    fleet = [{"vehicle_id": k + 1, "capacity": 25.0} for k in range(4)]

    events = []
    if with_events:
        events.append({
            "name": "School Dismissal @ Central",
            "edges": [[0, 3], [3, 0], [0, 7], [7, 0]],
            "start": 900.0, "end": 960.0,
            "extra_vol": 3000.0,
        })
        events.append({
            "name": "Match-Day Traffic",
            "edges": [[0, 11], [11, 0], [5, 11], [11, 5]],
            "start": 1080.0, "end": 1200.0,
            "extra_vol": 2000.0,
        })

    return {
        "network": {"nodes": nodes, "edges": None},   # coords → Euclidean
        "fleet": fleet,
        "weights": {"w1": 0.3, "w2": 0.5, "w3": 0.2, "w4": 0.0},
        "solver": {"M": 32, "T": T, "seed": 42, "gls": True,
                   "diversity": True, "events": events},
    }
