"""Module 4c — FastAPI server.

Endpoints (spec §4c):
    POST /solve                 → SolveResponse (blocking, small instances)
    POST /solve/async           → {job_id}
    GET  /solve/status/{job_id} → JobStatus
    GET  /solve/result/{job_id} → SolveResponse
    GET  /solve/stream/{job_id} → SSE stream of iteration frames
    POST /traffic/update        → warm-start signal for a running job
    POST /compare               → ComparisonResponse (what-if)
    GET  /kpi/{job_id}          → KPIReport (incl. violation breakdown)

Infeasibility policy (spec §4c): if the instance is infeasible (e.g. total
demand > fleet capacity), return HTTP 200 with `feasible=false`, the
least-infeasible routes, and the violation breakdown. HTTP 400 only for
malformed input (bad schema, unknown node ids, w1+w2+w3 ≠ 1).

Storage: in-memory job store — a single process is fine for the demo. A
real deploy would swap to Redis; the store interface is a dict-like shim.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from api.demo_scenario import demo_solve_request

from api.schemas import (
    ComparisonDelta,
    ComparisonResponse,
    ErrorEnvelope,
    JobStatus,
    KPIReport,
    LateViolation,
    OverloadViolation,
    RouteOut,
    SolveRequest,
    SolveResponse,
    TotalsOut,
    ViolationsOut,
)
from common.bootstrap import get_logger
from engine_numpy.explainer import explain_solution
from engine_numpy.fitness import FitnessResult
from engine_numpy.graph import Graph, build_from_dist
from engine_numpy.predictor import EventShock
from engine_numpy.qpso import QPSOResult, run_qpso

_log = get_logger("api")
app = FastAPI(title="MC-GQPSO API", version="0.1.0")


# ── In-memory job store ─────────────────────────────────────────────────────
_JOBS: Dict[str, dict] = {}


# ── Request → Graph adapter ────────────────────────────────────────────────
def _build_graph_from_request(req: SolveRequest) -> tuple[Graph, int, float]:
    """Return (graph, K, uniform_capacity).

    Uses the dense-matrix path (build_from_dist) so we don't need to depend on
    OSMnx for API requests. If `network.edges` is missing, dist is computed
    from lat/lng haversine (Euclidean approximation is fine at demo scale).
    """
    if req.weights.sum123() != 1.0 and abs(req.weights.sum123() - 1.0) > 1e-6:
        raise HTTPException(status_code=400, detail={
            "error": "bad_weights", "detail": f"w1+w2+w3 must be 1, got {req.weights.sum123()}",
        })

    ids = [n.id for n in req.network.nodes]
    if len(set(ids)) != len(ids):
        raise HTTPException(status_code=400, detail={
            "error": "duplicate_node_ids", "detail": "network.nodes.id must be unique",
        })
    if 0 not in ids:
        raise HTTPException(status_code=400, detail={
            "error": "missing_depot", "detail": "network.nodes must contain node id 0 (depot)",
        })

    N = len(ids)
    id_to_pos = {nid: i for i, nid in enumerate(sorted(ids))}
    if id_to_pos[0] != 0:
        # Depot is expected at position 0 by build_from_dist; reindex.
        raise HTTPException(status_code=400, detail={
            "error": "depot_must_be_zero", "detail": "depot id 0 must sort first (use contiguous 0..N)",
        })

    if req.network.edges:
        d = np.full((N, N), np.inf, dtype=np.float64)
        np.fill_diagonal(d, 0.0)
        for e in req.network.edges:
            if e.i not in id_to_pos or e.j not in id_to_pos:
                raise HTTPException(status_code=400, detail={
                    "error": "unknown_node_id", "detail": f"edge references unknown node ({e.i}, {e.j})",
                })
            d[id_to_pos[e.i], id_to_pos[e.j]] = e.dist
    else:
        # Fallback: coordinates → haversine distance in km. A pure
        # Euclidean-in-degrees would leave `dist` in degree units — the
        # downstream tau derivation (dist / 50 km/h) then produces
        # nonsensical minute counts.
        coords_deg = np.zeros((N, 2), dtype=np.float64)
        for n in req.network.nodes:
            if n.lat is None or n.lng is None:
                raise HTTPException(status_code=400, detail={
                    "error": "missing_coords",
                    "detail": f"node {n.id} has neither edges nor lat/lng",
                })
            coords_deg[id_to_pos[n.id]] = (n.lat, n.lng)
        R = 6371.0  # earth radius km
        lat = np.radians(coords_deg[:, 0])
        lng = np.radians(coords_deg[:, 1])
        dlat = lat[:, None] - lat[None, :]
        dlng = lng[:, None] - lng[None, :]
        a = (np.sin(dlat / 2) ** 2
             + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin(dlng / 2) ** 2)
        d = 2 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))

    # Customer attrs.
    demand = np.zeros(N, dtype=np.float32)
    tw_early = np.zeros(N, dtype=np.float32)
    tw_late = np.full(N, 1e9, dtype=np.float32)
    service = np.zeros(N, dtype=np.float32)
    for n in req.network.nodes:
        p = id_to_pos[n.id]
        demand[p] = n.demand
        if n.tw_early is not None:
            tw_early[p] = n.tw_early
        if n.tw_late is not None:
            tw_late[p] = n.tw_late
        service[p] = n.service_time

    caps = [v.capacity for v in req.fleet]
    if len(set(caps)) > 1:
        _log.info("heterogeneous fleet — using min capacity as uniform Q_k",
                  extra={"caps": caps})
    uniform_cap = float(min(caps))

    g = build_from_dist(
        d, depot=0, demand=demand, tw_early=tw_early, tw_late=tw_late,
        service_time=service, capacity=uniform_cap,
    )
    return g, len(req.fleet), uniform_cap


def _apply_events_via_bpr(g: Graph, events: list[EventShock], at_time_min: float) -> None:
    """Simple event injection for the API: add extra_vol to the specified edges
    if the event is active at `at_time_min`, then rerun BPR. This is the
    single-slice equivalent of TrafficSchedule.apply_events — the numpy
    engine uses it directly during a warm-start callback.
    """
    for ev in events:
        if not ev.active_at(at_time_min):
            continue
        for i, j in ev.edges:
            g.vol[i, j] = float(g.vol[i, j]) + ev.extra_vol
    g.update_weights(at_time_min)


def _events_from_request(req: SolveRequest) -> list[EventShock]:
    return [
        EventShock(name=e.name, edges=[tuple(x) for x in e.edges],
                   start_min=e.start, end_min=e.end, extra_vol=e.extra_vol)
        for e in req.solver.events
    ]


# ── Core solve → response builder ──────────────────────────────────────────
def _build_response(
    result: QPSOResult, g: Graph, events: list[EventShock], job_id: Optional[str] = None,
) -> SolveResponse:
    fit: FitnessResult = result.best_fitness
    explanations = explain_solution(g, fit.per_route, events=events, depot=g.depot)
    routes_out = []
    for rm, why in zip(fit.per_route, explanations):
        routes_out.append(RouteOut(
            vehicle_id=rm.vehicle_id,
            sequence=[int(x) for x in rm.sequence],
            arrival_times=[float(x) for x in rm.arrival_times],
            load=float(rm.load),
            distance=float(rm.distance),
            time=float(rm.time),
            emissions=float(rm.emissions),
            reasoning=why,
        ))

    totals = TotalsOut(
        distance=float(fit.total_distance),
        time=float(fit.total_time),
        emissions=float(fit.total_emissions),
        cost_Z=float(fit.Z_search),
        cost_Z_report=float(fit.Z_report),
        fleet_utilization=float(fit.fleet_utilization),
    )
    viols = ViolationsOut(
        late=[LateViolation(customer=int(v["customer"]),
                            minutes_late=float(v["minutes_late"]),
                            penalty=float(v["penalty_cost"]))
              for v in fit.violations.late],
        overload=[OverloadViolation(vehicle=int(v["vehicle"]),
                                    amount=float(v["amount"]),
                                    penalty=float(v["penalty_cost"]))
                  for v in fit.violations.overload],
    )

    return SolveResponse(
        routes=routes_out,
        total=totals,
        violations=viols,
        convergence=[float(x) for x in result.convergence],
        runtime_s=float(result.wall_time_s),
        feasible=bool(result.feasible),
        job_id=job_id,
    )


def _make_traffic_warm_start_source(job_id: str):
    """Return a callable(t) → Optional[warm-start-callback] that drains
    _JOBS[job_id]['traffic_queue'] one entry per iter.

    Each queued TrafficUpdate payload {edges:[{i,j,new_vol}], at_time} becomes
    a callback that installs `new_vol` on the specified edges, refreshes BPR,
    and reports `at_time` to `apply_warm_start` (spec §5a). Popping is
    single-reader (worker thread) so no lock is needed — list.pop under the
    GIL is atomic in CPython, and HTTP handlers only append.
    """
    def _source(_t: int):
        q = _JOBS.get(job_id, {}).get("traffic_queue")
        if not q:
            return None
        upd = q.pop(0)
        at_time = float(upd.get("at_time", 0.0))
        edges = upd.get("edges", []) or []

        def _cb(g: Graph, _iter_t: int) -> float:
            for e in edges:
                try:
                    i, j = int(e["i"]), int(e["j"])
                except (KeyError, TypeError, ValueError):
                    continue
                if 0 <= i < g.n and 0 <= j < g.n and g.adj_mask[i, j]:
                    if "new_vol" in e:
                        g.vol[i, j] = float(e["new_vol"])
                    elif "delta_vol" in e:
                        g.vol[i, j] = float(g.vol[i, j]) + float(e["delta_vol"])
            g.update_weights(at_time)
            _log.info("traffic-update warm-start", extra={
                "job_id": job_id, "at_time": at_time, "n_edges": len(edges),
            })
            return at_time
        return _cb
    return _source


def _solve_blocking(req: SolveRequest, job_id: Optional[str] = None) -> SolveResponse:
    g, K, cap = _build_graph_from_request(req)
    events = _events_from_request(req)
    # Fire events at t=0 so their extra_vol influences the base BPR slice.
    if events:
        _apply_events_via_bpr(g, events, at_time_min=0.0)

    # Async jobs get the live warm-start channel; blocking /solve does not
    # (no way to inject mid-call, and small instances would race the client).
    warm_start_source = None
    if job_id is not None:
        _JOBS[job_id].setdefault("traffic_queue", [])
        warm_start_source = _make_traffic_warm_start_source(job_id)

    result = run_qpso(
        g, K=K, M=req.solver.M, T=req.solver.T, seed=req.solver.seed,
        weights=req.weights.model_dump(),
        fleet_capacity=cap,
        gls=req.solver.gls, diversity=req.solver.diversity,
        warm_start_source=warm_start_source,
    )
    return _build_response(result, g, events, job_id=job_id)


# ── Endpoints ──────────────────────────────────────────────────────────────
@app.post("/solve", response_model=SolveResponse)
def solve(req: SolveRequest) -> SolveResponse:
    """Blocking solve — intended for small instances (M·T·Nc small enough to
    finish inside the HTTP timeout). Larger runs → POST /solve/async."""
    return _solve_blocking(req)


@app.post("/solve/async")
def solve_async(req: SolveRequest, tasks: BackgroundTasks) -> dict:
    job_id = uuid.uuid4().hex
    _JOBS[job_id] = {
        "state": "pending", "req": req, "best_cost": None,
        "iteration": None, "started_at": None, "result": None, "error": None,
    }

    def _worker():
        _JOBS[job_id]["state"] = "running"
        _JOBS[job_id]["started_at"] = time.time()
        try:
            resp = _solve_blocking(req, job_id=job_id)
            _JOBS[job_id]["result"] = resp
            _JOBS[job_id]["best_cost"] = resp.total.cost_Z_report
            _JOBS[job_id]["iteration"] = req.solver.T
            _JOBS[job_id]["state"] = "done"
        except HTTPException as e:
            _JOBS[job_id]["state"] = "error"
            _JOBS[job_id]["error"] = str(e.detail)
        except Exception as e:
            _JOBS[job_id]["state"] = "error"
            _JOBS[job_id]["error"] = f"{type(e).__name__}: {e}"

    tasks.add_task(_worker)
    return {"job_id": job_id}


@app.get("/solve/status/{job_id}", response_model=JobStatus)
def solve_status(job_id: str) -> JobStatus:
    j = _JOBS.get(job_id)
    if j is None:
        raise HTTPException(status_code=404, detail={"error": "unknown_job", "detail": job_id})
    return JobStatus(
        job_id=job_id, state=j["state"], best_cost=j["best_cost"],
        iteration=j["iteration"], error=j["error"],
    )


@app.get("/solve/result/{job_id}", response_model=SolveResponse)
def solve_result(job_id: str) -> SolveResponse:
    j = _JOBS.get(job_id)
    if j is None:
        raise HTTPException(status_code=404, detail={"error": "unknown_job", "detail": job_id})
    if j["state"] != "done":
        raise HTTPException(status_code=409, detail={
            "error": "not_ready", "detail": f"job state = {j['state']}",
        })
    return j["result"]


@app.get("/solve/stream/{job_id}")
async def solve_stream(job_id: str) -> StreamingResponse:
    """SSE stream (spec §4c schema): each frame is
    {iteration, best_cost, wall_time_s, feasible}.

    Current MVP polls the job's convergence trace after completion — a
    later revision moves to a genuine iter-by-iter feed via the QPSO log
    callback.
    """
    if job_id not in _JOBS:
        raise HTTPException(status_code=404, detail={"error": "unknown_job", "detail": job_id})

    async def _iterate():
        # Wait until done (max 60s), then stream the convergence trace as frames.
        started = time.time()
        while _JOBS[job_id]["state"] in ("pending", "running"):
            if time.time() - started > 60:
                yield f"event: error\ndata: {json.dumps({'error': 'timeout'})}\n\n"
                return
            await asyncio.sleep(0.1)
        j = _JOBS[job_id]
        if j["state"] == "error":
            yield f"event: error\ndata: {json.dumps({'error': j['error']})}\n\n"
            return
        resp: SolveResponse = j["result"]
        for i, c in enumerate(resp.convergence):
            frame = {
                "iteration": i,
                "best_cost": float(c),
                "wall_time_s": (i / max(1, len(resp.convergence))) * resp.runtime_s,
                "feasible": resp.feasible if i == len(resp.convergence) - 1 else None,
            }
            yield f"data: {json.dumps(frame)}\n\n"
            await asyncio.sleep(0.005)
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(_iterate(), media_type="text/event-stream")


@app.post("/compare", response_model=ComparisonResponse)
def compare(a: SolveRequest, b: SolveRequest) -> ComparisonResponse:
    """What-if (spec §4e). Typically A = events-on, B = events-off."""
    resp_a = _solve_blocking(a)
    resp_b = _solve_blocking(b)
    delta = ComparisonDelta(
        distance=resp_a.total.distance - resp_b.total.distance,
        time=resp_a.total.time - resp_b.total.time,
        emissions=resp_a.total.emissions - resp_b.total.emissions,
        late_deliveries=len(resp_a.violations.late) - len(resp_b.violations.late),
    )
    return ComparisonResponse(a=resp_a, b=resp_b, delta=delta)


@app.get("/kpi/{job_id}", response_model=KPIReport)
def kpi(job_id: str) -> KPIReport:
    j = _JOBS.get(job_id)
    if j is None or j["state"] != "done":
        raise HTTPException(status_code=404, detail={
            "error": "no_result", "detail": f"job {job_id} not ready",
        })
    r: SolveResponse = j["result"]
    return KPIReport(
        job_id=job_id, total=r.total, violations=r.violations,
        routes=r.routes, convergence=r.convergence,
    )


@app.post("/traffic/update")
def traffic_update(update: dict) -> dict:
    """Warm-start signal for a running async job (spec §5a).

    Body: `{job_id, edges:[{i, j, new_vol}] | [{i, j, delta_vol}], at_time}`.
    Enqueues one update; the async worker's warm-start source pops it on the
    next QPSO iteration and applies `apply_warm_start`:
        1. install new_vol / delta_vol on the listed edges
        2. refresh BPR at `at_time`
        3. re-evaluate every particle in the new landscape
        4. RESET pbest values, KEEP pbest positions
        5. α *= 1.5

    Returns 202-shape body with the queue depth. If the job is already done
    the update is still queued but has no effect; state=done is discoverable
    via /solve/status.
    """
    job_id = update.get("job_id")
    if job_id not in _JOBS:
        raise HTTPException(status_code=404, detail={
            "error": "unknown_job", "detail": str(job_id),
        })
    if _JOBS[job_id]["state"] == "done":
        return {"accepted": False, "job_id": job_id, "reason": "job already done"}
    _JOBS[job_id].setdefault("traffic_queue", []).append(update)
    return {"accepted": True, "job_id": job_id, "queued": len(_JOBS[job_id]["traffic_queue"])}


@app.get("/health")
def health() -> dict:
    return {"ok": True, "jobs_in_memory": len(_JOBS)}


@app.get("/demo/scenario")
def demo_scenario(with_events: bool = False, T: int = 250) -> dict:
    """Return a canned SolveRequest the UI can POST straight back to /solve."""
    return demo_solve_request(with_events=with_events, T=T)


@app.get("/demo/bundled")
def demo_bundled() -> dict:
    """Return the committed bundled Bengaluru scenario (spec §5e).

    Zero external deps: this is the demo-day fallback that must run without
    OSMnx/SUMO/internet. Shape is a SolveRequest — POST straight to /solve.
    """
    from pathlib import Path
    import json
    p = Path(__file__).resolve().parent.parent / "data" / "demo" / "bengaluru_district.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail={
            "error": "no_bundled_demo",
            "detail": f"{p} missing — regenerate via `python data/demo/build_scenario.py`",
        })
    scenario = json.loads(p.read_text())
    # Strip the traffic_schedule (SolveRequest schema doesn't hold it — the
    # single-slice events channel carries the demo-critical shocks).
    return {
        "network": scenario["network"],
        "fleet": scenario["fleet"],
        "weights": scenario["weights"],
        "solver": scenario["solver"],
    }


# ── Static UI mount ────────────────────────────────────────────────────────
# Mount is best-effort: allows the API to be used standalone if `ui/` was
# stripped for a headless deploy.
_UI_DIR = Path(__file__).resolve().parent.parent / "ui"
if _UI_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_UI_DIR)), name="static")

    @app.get("/")
    def _index():
        return FileResponse(str(_UI_DIR / "index.html"))
