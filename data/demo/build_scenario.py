"""Generate the bundled demo scenario JSON (spec §5e — removes #1 demo risk).

Writes `data/demo/bengaluru_district.json`:
  - 25 nodes (depot + 24 customers) with lat/lng
  - Pre-computed dense edge distances (haversine, km) so no network needed
  - Free-flow tau derived at 30 km/h (urban)
  - Realistic capacities per edge (residential/tertiary ranges)
  - 24 h synthetic BPR volume schedule (30-min buckets)
  - 3 named event shocks

The API's /demo/scenario currently synthesizes on the fly; this JSON is the
committed, run-anywhere version the demo team can load with ZERO external
deps — no OSMnx pull, no SUMO, no internet.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

_CENTER = (12.9716, 77.5946)   # Bengaluru
_R_KM = 4.0
_KM_PER_DEG_LAT = 111.0


def _offset(dx_km: float, dy_km: float) -> tuple[float, float]:
    dlat = dy_km / _KM_PER_DEG_LAT
    dlng = dx_km / (_KM_PER_DEG_LAT * math.cos(math.radians(_CENTER[0])))
    return _CENTER[0] + dlat, _CENTER[1] + dlng


def _haversine_km(lat1, lng1, lat2, lng2) -> float:
    R = 6371.0
    la1, la2 = math.radians(lat1), math.radians(lat2)
    dla = math.radians(lat2 - lat1)
    dlo = math.radians(lng2 - lng1)
    a = math.sin(dla / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin(dlo / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def build(seed: int = 42, n_customers: int = 24) -> dict[str, Any]:
    rng = random.Random(seed)

    nodes = [{
        "id": 0, "lat": _CENTER[0], "lng": _CENTER[1],
        "name": "Depot (Central Warehouse)",
        "demand": 0.0, "tw_early": 0.0, "tw_late": 1440.0, "service_time": 0.0,
    }]
    for i in range(1, n_customers + 1):
        theta = rng.uniform(0, 2 * math.pi)
        r = rng.uniform(0.8, _R_KM)
        lat, lng = _offset(r * math.cos(theta), r * math.sin(theta))
        nodes.append({
            "id": i, "lat": lat, "lng": lng,
            "name": f"Stop {i:02d}",
            "demand": float(rng.choice([2, 3, 4, 5, 6, 8])),
            "tw_early": float(rng.choice([0, 30, 60, 120, 180])),
            "tw_late":  float(rng.choice([240, 360, 480, 600, 720])),
            "service_time": 5.0,
        })

    N = len(nodes)
    edges: list[dict[str, Any]] = []
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            d = _haversine_km(nodes[i]["lat"], nodes[i]["lng"],
                              nodes[j]["lat"], nodes[j]["lng"])
            speed = rng.choice([25.0, 30.0, 35.0, 40.0])  # urban km/h
            cap = float(rng.choice([600, 900, 1200, 1500]))  # veh/hr
            edges.append({
                "i": i, "j": j,
                "dist": round(d, 4),
                "tau0": round(d / speed * 60.0, 3),
                "cap": cap,
                "speed0": speed,
            })

    fleet = [{"vehicle_id": k + 1, "capacity": 25.0} for k in range(4)]

    # 24h synthetic BPR volume series, 30-min buckets, morning + evening peaks.
    slot_min = 30
    n_slots = 24 * 60 // slot_min
    volume_series = []
    for slot in range(n_slots):
        t = slot * slot_min
        peak_am = math.exp(-((t - 8 * 60) ** 2) / (2 * 70 ** 2))
        peak_pm = math.exp(-((t - 18 * 60) ** 2) / (2 * 90 ** 2))
        base = 0.5 * (peak_am + peak_pm)
        # Per-edge scaling (deterministic in edge index for reproducibility).
        slot_vols = []
        for k, e in enumerate(edges):
            per_edge = 0.6 + 0.8 * ((k * 2654435761 % 1000) / 1000.0)
            slot_vols.append(round(150.0 + 1100.0 * base * per_edge, 1))
        volume_series.append({"slot_min": slot_min, "start_min": t, "volumes": slot_vols})

    events = [
        {
            "name": "School Dismissal — Central Corridor",
            "edges": [[0, 3], [3, 0], [0, 7], [7, 0]],
            "start": 15 * 60.0, "end": 16 * 60.0,
            "extra_vol": 2500.0,
        },
        {
            "name": "Match Day — Stadium Approach",
            "edges": [[0, 11], [11, 0], [5, 11], [11, 5]],
            "start": 18 * 60.0, "end": 20 * 60.0,
            "extra_vol": 2000.0,
        },
        {
            "name": "Rain Advisory — Peripheral",
            "edges": [[13, 20], [20, 13], [17, 22], [22, 17]],
            "start": 12 * 60.0, "end": 14 * 60.0,
            "extra_vol": 1200.0,
        },
    ]

    return {
        "meta": {
            "name": "bengaluru_district",
            "generator": "data/demo/build_scenario.py",
            "seed": seed,
            "center": {"lat": _CENTER[0], "lng": _CENTER[1]},
            "n_nodes": N, "n_edges": len(edges), "n_slots": n_slots,
        },
        "network": {"nodes": nodes, "edges": edges},
        "fleet": fleet,
        "weights": {"w1": 0.3, "w2": 0.5, "w3": 0.2, "w4": 0.0},
        "solver": {
            "M": 32, "T": 300, "seed": 42,
            "gls": True, "diversity": True, "time_dependent": True,
            "events": events,
        },
        "traffic_schedule": {"bucket_min": slot_min, "slots": volume_series},
    }


def main() -> None:
    out = Path(__file__).resolve().parent / "bengaluru_district.json"
    scenario = build()
    out.write_text(json.dumps(scenario))
    print(f"wrote {out}  ({out.stat().st_size / 1024:.1f} KiB)")
    print(f"nodes={scenario['meta']['n_nodes']}  edges={scenario['meta']['n_edges']}  "
          f"slots={scenario['meta']['n_slots']}  events={len(scenario['solver']['events'])}")


if __name__ == "__main__":
    main()
