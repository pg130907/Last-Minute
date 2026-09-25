"""Module 1 — Instance loaders.

Included: load_8node (hard-gate fixture from spec §3.3), make_synthetic (for
diversity/parametric tests), a basic Solomon parser, and load_osm behind a
lazy import (osmnx is heavyweight and Python-3.14-unfriendly today).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from common.bootstrap import get_logger
from engine_numpy.graph import (
    DEFAULT_FREE_SPEED,
    Graph,
    build_from_dist,
    extract_lcc,
)

_log = get_logger("loaders")


# ── 8-node hard-gate fixture (spec §3.3, verbatim) ──────────────────────────
_D8 = np.array([
    [0.0,  4.0,  6.0,  7.5,  9.0, 20.0, 10.0, 16.0,  8.0],
    [4.0,  0.0,  6.5,  4.0, 10.0,  5.0,  7.5, 11.0, 10.0],
    [6.0,  6.5,  0.0,  7.5, 10.0, 10.0,  7.5,  7.5,  7.5],
    [7.5,  4.0,  7.5,  0.0, 10.0,  5.0,  9.0,  9.0, 15.0],
    [9.0, 10.0, 10.0, 10.0,  0.0, 10.0,  7.5,  7.5, 10.0],
    [20.0, 5.0, 10.0,  5.0, 10.0,  0.0,  7.0,  9.0,  7.5],
    [10.0, 7.5,  7.5,  9.0,  7.5,  7.0,  0.0,  7.0, 10.0],
    [16.0, 11.0, 7.5,  9.0,  7.5,  9.0,  7.0,  0.0, 10.0],
    [8.0, 10.0,  7.5, 15.0, 10.0,  7.5, 10.0, 10.0,  0.0],
], dtype=np.float64)

_Q8 = np.array([0, 1, 2, 1, 2, 1, 4, 2, 2], dtype=np.float32)  # depot demand=0


def load_8node() -> Graph:
    """8 customers + depot(0), 2 vehicles of capacity 8t (spec §3.3).

    KNOWN OPTIMUM: 67.5 km on routes [0,4,7,6,0] and [0,1,3,5,8,2,0].
    """
    g = build_from_dist(
        _D8,
        depot=0,
        default_speed=DEFAULT_FREE_SPEED,
        cap_default=1.0e6,     # unlimited road capacity → BPR is a no-op
        speed_dependent=False, # flat emissions for a pure-CVRP benchmark
        demand=_Q8,
        capacity=8.0,
    )
    return g


# ── Synthetic instance (for diversity / parametric tests) ───────────────────
def make_synthetic(
    n: int,
    seed: int = 0,
    layout: str = "random",
    *,
    grid_side: Optional[float] = None,
    speed_dependent: bool = False,
    capacity: float = 100.0,
) -> Graph:
    """N customers + depot, Euclidean distances.

    layout='random' scatters uniformly on a `grid_side`×`grid_side` square
    (default 100). layout='grid' places on a √N grid.
    """
    rng = np.random.default_rng(seed)
    side = grid_side if grid_side is not None else 100.0
    N = n + 1  # + depot
    if layout == "grid":
        m = int(np.ceil(np.sqrt(N)))
        xs = np.tile(np.linspace(0, side, m), m)[:N]
        ys = np.repeat(np.linspace(0, side, m), m)[:N]
        coords = np.stack([xs, ys], axis=1)
    else:
        coords = rng.uniform(0, side, size=(N, 2))
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)

    demand = np.concatenate([[0.0], rng.uniform(1, 10, size=n)]).astype(np.float32)
    return build_from_dist(
        d,
        depot=0,
        speed_dependent=speed_dependent,
        demand=demand,
        capacity=capacity,
    )


def make_disconnected_synthetic(
    main_n: int = 6, isolated_n: int = 2, seed: int = 0
) -> Graph:
    """Fixture for test_lcc: a well-connected core plus `isolated_n` orphans.

    Orphans have adj_mask all False (both incoming and outgoing).
    """
    rng = np.random.default_rng(seed)
    N = main_n + isolated_n
    coords = rng.uniform(0, 100.0, size=(N, 2))
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    # Wipe every edge touching the last `isolated_n` nodes.
    for k in range(main_n, N):
        d[k, :] = np.inf
        d[:, k] = np.inf
    np.fill_diagonal(d, 0.0)
    return build_from_dist(d, depot=0)


# ── Solomon (VRPTW) — minimal parser sufficient for benchmark harness ───────
def load_solomon(path: str | Path) -> Graph:
    """Parse the classic Solomon 100-customer text format.

    File layout (excerpt):
        NAME
        VEHICLE
        NUMBER  CAPACITY
          25       200
        CUSTOMER
        CUST NO.  XCOORD.  YCOORD.  DEMAND  READY  DUE  SERVICE
          0        40       50        0        0    1236    0
          1        45       68       10      912     967    90
          ...
    """
    p = Path(path)
    lines = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
    # Find capacity: line after "NUMBER CAPACITY"
    for i, ln in enumerate(lines):
        if ln.upper().startswith("NUMBER"):
            _, cap = lines[i + 1].split()[:2]
            capacity = float(cap)
            break
    else:
        raise ValueError(f"{p}: no VEHICLE section")

    # Find CUSTOMER block
    for i, ln in enumerate(lines):
        if ln.upper().startswith("CUST"):
            data_lines = lines[i + 1:]
            break
    else:
        raise ValueError(f"{p}: no CUSTOMER section")

    rows = []
    for ln in data_lines:
        parts = ln.split()
        if len(parts) < 7:
            continue
        rows.append([float(x) for x in parts[:7]])
    arr = np.array(rows)
    xs, ys = arr[:, 1], arr[:, 2]
    demand = arr[:, 3].astype(np.float32)
    ready = arr[:, 4].astype(np.float32)
    due = arr[:, 5].astype(np.float32)
    service = arr[:, 6].astype(np.float32)
    coords = np.stack([xs, ys], axis=1)
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    g = build_from_dist(
        d, depot=0, speed_dependent=False,
        demand=demand, tw_early=ready, tw_late=due, service_time=service,
        capacity=capacity,
    )
    _log.info("loaded solomon", extra={"path": str(p), "n": g.n, "capacity": capacity})
    return g


def load_gh(path: str | Path) -> Graph:
    """Gehring–Homberger uses the same text schema as Solomon."""
    return load_solomon(path)


# ── Bundled zero-dependency demo scenario (spec §5e) ────────────────────────
def load_bundled_demo(
    path: str | Path | None = None,
) -> tuple[Graph, dict]:
    """Load the pre-baked Bengaluru district JSON. Returns (graph, meta) where
    `meta` carries fleet, weights, solver config, events, and a
    traffic_schedule dict for the dynamic-demo path.

    Runs with ZERO external deps — no OSMnx pull, no SUMO. Regenerate via
    `python data/demo/build_scenario.py` if you change the fixture.
    """
    import json
    if path is None:
        path = Path(__file__).resolve().parent.parent / "data" / "demo" / "bengaluru_district.json"
    p = Path(path)
    with p.open() as f:
        s = json.load(f)

    nodes = s["network"]["nodes"]
    N = len(nodes)
    d = np.full((N, N), np.inf, dtype=np.float64)
    tau0 = np.full((N, N), np.inf, dtype=np.float64)
    cap = np.zeros((N, N), dtype=np.float32)
    speed0 = np.zeros((N, N), dtype=np.float32)
    adj = np.zeros((N, N), dtype=bool)
    np.fill_diagonal(d, 0.0)

    for e in s["network"]["edges"]:
        i, j = int(e["i"]), int(e["j"])
        d[i, j] = float(e["dist"])
        tau0[i, j] = float(e["tau0"])
        cap[i, j] = float(e["cap"])
        speed0[i, j] = float(e.get("speed0", 30.0))
        adj[i, j] = True

    demand = np.array([n.get("demand", 0.0) for n in nodes], dtype=np.float32)
    tw_early = np.array([n.get("tw_early", 0.0) for n in nodes], dtype=np.float32)
    tw_late = np.array([n.get("tw_late", 1e9) for n in nodes], dtype=np.float32)
    service = np.array([n.get("service_time", 0.0) for n in nodes], dtype=np.float32)

    fleet_caps = [v["capacity"] for v in s["fleet"]]
    uniform_cap = float(min(fleet_caps))

    g = Graph(
        dist=d.astype(np.float32), tau0=tau0.astype(np.float32),
        cap=cap, vol=np.zeros((N, N), dtype=np.float32), adj_mask=adj,
        speed0=speed0, depot=0,
        demand=demand, tw_early=tw_early, tw_late=tw_late, service_time=service,
        capacity=uniform_cap,
    )
    return g, s


# ── OSM (deferred) ──────────────────────────────────────────────────────────
_OSM_CAP_PER_LANE = {
    "motorway": 2000, "trunk": 1800, "primary": 1500, "secondary": 1200,
    "tertiary": 900, "residential": 600, "service": 300, "unclassified": 300,
}


def load_osm(place_or_bbox, network_type: str = "drive", depot: Optional[int] = None) -> Graph:
    """Real network via OSMnx (spec §2.4).

    Deferred to when we actually have osmnx installable on the target Python.
    Raises ImportError if osmnx isn't importable — callers should fall back
    to a bundled demo scenario (§5e).
    """
    try:
        import osmnx as ox  # noqa: WPS433
    except ImportError as e:
        raise ImportError(
            "osmnx not installed; install `osmnx>=2.0` or use load_solomon / "
            "make_synthetic / the bundled demo scenario instead."
        ) from e

    if isinstance(place_or_bbox, str):
        G = ox.graph_from_place(place_or_bbox, network_type=network_type)
    else:
        G = ox.graph_from_bbox(*place_or_bbox, network_type=network_type)

    G = ox.truncate.largest_component(G, strongly=True)
    G = ox.simplify_graph(G)
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)

    nodes = list(G.nodes())
    idx = {osm_id: i for i, osm_id in enumerate(nodes)}
    N = len(nodes)

    dist = np.full((N, N), np.inf, dtype=np.float64)
    tau0 = np.full((N, N), np.inf, dtype=np.float64)
    cap = np.zeros((N, N), dtype=np.float32)
    speed0 = np.zeros((N, N), dtype=np.float32)
    adj = np.zeros((N, N), dtype=bool)
    np.fill_diagonal(dist, 0.0)

    for u, v, data in G.edges(data=True):
        i, j = idx[u], idx[v]
        length_km = float(data.get("length", 0.0)) / 1000.0
        speed_kmh = float(data.get("speed_kph", 30.0))
        travel_s = float(data.get("travel_time", length_km / max(speed_kmh, 1e-3) * 3600))
        hwy = data.get("highway", "unclassified")
        if isinstance(hwy, list):
            hwy = hwy[0]
        lanes = data.get("lanes", 1)
        if isinstance(lanes, list):
            lanes = lanes[0]
        try:
            lanes = int(lanes)
        except (TypeError, ValueError):
            lanes = 1
        cap_ij = _OSM_CAP_PER_LANE.get(str(hwy), 300) * max(lanes, 1)

        # Keep shortest parallel edge only (dense matrix can't hold multi-edges).
        if length_km < dist[i, j]:
            dist[i, j] = length_km
            tau0[i, j] = travel_s / 60.0
            speed0[i, j] = speed_kmh
            cap[i, j] = cap_ij
            adj[i, j] = True

    if depot is None:
        # Highest-degree node.
        depot = int(adj.sum(axis=1).argmax())
    else:
        depot = idx[depot]

    g = Graph(
        dist=dist.astype(np.float32),
        tau0=tau0.astype(np.float32),
        cap=cap,
        vol=np.zeros((N, N), dtype=np.float32),
        adj_mask=adj,
        speed0=speed0,
        depot=depot,
    )
    # LCC was applied via ox.truncate above; double-check on the densified form.
    g, _ = extract_lcc(g, strongly=True)
    return g
