"""Module 3.2 — QPSO core.

Delta-potential-well update on the two angle chains IS the rotation gate
(one mechanism, not two per spec §3.2). Classical QPSO update:

    alpha_t = alpha0 - (alpha0 - alpha1) · t/T             # linear anneal
    mbest   = mean(pbest_theta, axis=0)                    # finite by §2.5 guard
    for each particle i, each chain c:
        phi_i, u_i, s_i ~ U(0,1)   sign = +1 if s_i ≥ 0.5 else −1
        P_i    = phi_i · pbest_i + (1 − phi_i) · gbest
        theta  = ( P_i + sign · alpha · |mbest − theta_old| · ln(1/u_i) ) mod 2π

Note on the spec's own pseudocode: the literal snippet writes
`theta = (theta + dtheta) mod 2π` and defines `P_i` on the line above but
never reads it. That drops the attractor, leaving the update as pure
scale-of-dispersion noise — nothing pulls theta toward the good region and
the swarm can't converge. Standard QPSO (`theta = P_i ± α|mbest−θ|·ln(1/u)`)
does use P_i as the pivot and is what the surrounding text describes
("delta-potential well" is exactly that pivot-plus-noise form). We
implement standard QPSO; the mod-2π wrap is preserved as the rotation gate.

Complexity: O(T · M · N logN) — argsort in decode dominates per iter.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from common.bootstrap import get_logger, set_all_seeds, wallclock
from engine_numpy.diversity import (
    DiversityState,
    apply_chaos_and_rescatter,
    chaos_escape,
    stalled,
)
from engine_numpy.encoding import decode, random_init, re_encode, wrap_angles
from engine_numpy.fitness import (
    DEFAULT_R_CAP,
    DEFAULT_R_TW,
    FitnessResult,
    default_tau_lookup,
    evaluate,
)
from engine_numpy.gls import compute_knn, guided_local_search
from engine_numpy.graph import Graph
from engine_numpy.warmstart import SwarmState, apply_warm_start

_log = get_logger("qpso")


@dataclass
class QPSOResult:
    best_cost: float                # gbest Z_search
    best_report_cost: float         # gbest Z_report
    best_routes: List[List[int]]
    best_fitness: FitnessResult
    convergence: List[float] = field(default_factory=list)
    iterations: int = 0
    wall_time_s: float = 0.0
    seed: int = 0
    feasible: bool = False


def run_qpso(
    g: Graph,
    K: int,
    M: int = 20,
    T: int = 500,
    seed: int = 0,
    weights: Optional[dict] = None,
    alpha0: float = 1.0,
    alpha1: float = 0.5,
    fleet_capacity: Optional[float] = None,
    R_cap: float = DEFAULT_R_CAP,
    R_tw: float = DEFAULT_R_TW,
    tau_lookup: Callable = default_tau_lookup,
    log_every: int = 0,
    # ── Module 4a: diversity control ──────────────────────────────────────
    diversity: bool = False,
    L_stall: int = 15,
    dist_eps: float = 0.05,
    fit_eps: float = 1e-4,
    T_chaos: int = 20,
    eps_saturation: float = 1e-3,
    L_sat: int = 10,
    # ── Module 4b: guided local search ────────────────────────────────────
    gls: bool = False,
    gls_every: int = 25,
    gls_elite_frac: float = 0.1,
    gls_knn: int = 10,
    gls_max_outer: int = 2,
    gls_max_inner: int = 20,
    # ── Module 5a: scheduled warm-starts (each = (iteration, callback(g, t_min)))
    # Callback should mutate `g` (install a new BPR/tau slice) before returning
    # the wall-clock t_min the swarm should be scored against.
    warm_starts: Optional[list] = None,
    # Dynamic warm-start source (polled each iter). Signature:
    #   warm_start_source(t: int) -> Optional[Callable[[Graph, int], float]]
    # If it returns a callback, the loop applies apply_warm_start with it.
    # This is the plumbing the API's /traffic/update uses to reroute an
    # in-flight async job — the HTTP handler enqueues into shared state, and
    # this source drains that queue one item per iteration.
    warm_start_source: Optional[Callable] = None,
    alpha_boost: float = 1.5,
) -> QPSOResult:
    """Run MC-GQPSO Module-3 core (no diversity, no GLS, no warm-start).

    Args:
        g: prepared Graph.
        K: number of vehicles.
        M, T: swarm size and iteration budget.
        seed: reproducibility seed (four-way per common.bootstrap).
        weights: {w1,w2,w3,w4}; sum(w1..w3)=1.
        alpha0, alpha1: linear-anneal endpoints for the delta-well scale.
        fleet_capacity: overrides g.capacity if given.
        log_every: >0 to emit iteration logs at that period; 0 silences.
    """
    set_all_seeds(seed)
    rng = np.random.default_rng(seed)
    Nc = g.n - 1
    if Nc < 1:
        raise ValueError(f"need at least 1 customer, got graph with n={g.n}")

    # Init swarm: shape (M, 2, Nc). Chain 0 = seq, chain 1 = veh.
    theta = np.stack(
        [np.stack(random_init(Nc, rng)) for _ in range(M)], axis=0
    ).astype(np.float32)

    pbest_theta = theta.copy()
    pbest_val = np.full(M, np.inf, dtype=np.float64)
    pbest_report = np.full(M, np.inf, dtype=np.float64)
    pbest_routes: List[List[List[int]]] = [None] * M  # type: ignore
    pbest_fitness: List[Optional[FitnessResult]] = [None] * M

    def _eval_particle(theta_i: np.ndarray) -> tuple[FitnessResult, list]:
        routes, _, _ = decode(theta_i[0], theta_i[1], K, depot=g.depot)
        res = evaluate(
            g, routes, weights=weights, R_cap=R_cap, R_tw=R_tw,
            tau_lookup=tau_lookup, fleet_capacity=fleet_capacity,
        )
        return res, routes

    # Initial evaluation.
    for i in range(M):
        res, routes = _eval_particle(theta[i])
        pbest_val[i] = res.Z_search
        pbest_report[i] = res.Z_report
        pbest_routes[i] = routes
        pbest_fitness[i] = res

    gbest_idx = int(np.argmin(pbest_val))
    gbest_theta = pbest_theta[gbest_idx].copy()
    gbest_val = float(pbest_val[gbest_idx])
    gbest_report = float(pbest_report[gbest_idx])
    gbest_routes = list(pbest_routes[gbest_idx])
    gbest_fitness = pbest_fitness[gbest_idx]

    convergence: List[float] = [gbest_val]
    t0 = wallclock()

    # Diversity + GLS pre-flight.
    div_state = DiversityState.new(theta.shape) if diversity else None
    knn = compute_knn(g, k=gls_knn) if gls else None
    n_elite = max(1, int(round(gls_elite_frac * M))) if gls else 0

    # Scheduled warm-starts (Module 5a). Sorted by trigger iteration.
    ws_queue: list = sorted(warm_starts or [], key=lambda x: x[0])
    alpha_boost_extra: float = 1.0   # pumped by warm_start; decays via anneal

    def _apply_ws(callback):
        """Shared machinery: build a SwarmState view, run apply_warm_start, and
        pull the mutated scalars back into the closure. Used by both the
        static schedule and the dynamic queue."""
        nonlocal gbest_theta, gbest_val, gbest_report, gbest_routes
        nonlocal gbest_fitness, alpha_boost_extra
        at_min = float(callback(g, t))    # callback mutates g; returns wall-time min
        state = SwarmState(
            theta=theta, pbest_theta=pbest_theta, pbest_val=pbest_val,
            pbest_report=pbest_report, pbest_routes=pbest_routes,
            pbest_fitness=pbest_fitness, gbest_theta=gbest_theta,
            gbest_val=gbest_val, gbest_report=gbest_report,
            gbest_routes=gbest_routes, gbest_fitness=gbest_fitness,
            alpha=alpha_boost_extra, K=K, depot=g.depot,
        )
        apply_warm_start(
            state, g, at_min, weights=weights, R_cap=R_cap, R_tw=R_tw,
            fleet_capacity=fleet_capacity, tau_lookup=tau_lookup,
            alpha_boost=alpha_boost,
        )
        gbest_theta = state.gbest_theta
        gbest_val = state.gbest_val
        gbest_report = state.gbest_report
        gbest_routes = state.gbest_routes
        gbest_fitness = state.gbest_fitness
        alpha_boost_extra = state.alpha

    for t in range(1, T + 1):
        # Module 5a — fire any statically-scheduled warm-starts due at this iter.
        while ws_queue and ws_queue[0][0] <= t:
            _, callback = ws_queue.pop(0)
            _apply_ws(callback)
        # Module 5a — drain dynamic queue (one item per iter, so slow-arriving
        # updates all get a shot at the swarm rather than blocking a single tick).
        if warm_start_source is not None:
            dyn_cb = warm_start_source(t)
            if dyn_cb is not None:
                _apply_ws(dyn_cb)

        base_alpha = alpha0 - (alpha0 - alpha1) * (t / T)
        # Spec §5a cap `alpha ← min(1.0, alpha * 1.5)` applies to the EFFECTIVE
        # α used this iter — not to the multiplier — so a warm-start that
        # arrives when base_alpha is 0.75 still pulls α to 1.0, and a shock
        # late in the run when base_alpha is 0.5 still pulls α to 0.75.
        alpha = min(alpha0, base_alpha * alpha_boost_extra)
        # Boost multiplier decays back to 1.0 (proportional kick fades over
        # ~35 iters; no discrete re-clamp).
        alpha_boost_extra = max(1.0, alpha_boost_extra * 0.98)

        # mbest: mean of pbest angles across the swarm (finite by §2.5 guard on fitness).
        mbest = pbest_theta.mean(axis=0)  # (2, Nc)

        # Vectorized update across particles and chains.
        phi = rng.uniform(0.0, 1.0, size=theta.shape).astype(np.float32)
        u = rng.uniform(1e-10, 1.0, size=theta.shape).astype(np.float32)
        sign = np.where(rng.uniform(0.0, 1.0, size=theta.shape) >= 0.5, 1.0, -1.0).astype(np.float32)

        P = phi * pbest_theta + (1.0 - phi) * gbest_theta[None, :, :]
        theta = P + sign * alpha * np.abs(mbest[None, :, :] - theta) * (-np.log(u))
        theta = wrap_angles(theta).astype(np.float32)

        # Module 4a — phase rescatter (every iter).
        if div_state is not None:
            theta = apply_chaos_and_rescatter(
                theta, div_state, rng, eps_sat=eps_saturation, L_sat=L_sat,
            )

        # Evaluate + update pbest/gbest.
        for i in range(M):
            res, routes = _eval_particle(theta[i])
            if res.Z_search < pbest_val[i]:
                pbest_val[i] = res.Z_search
                pbest_report[i] = res.Z_report
                pbest_theta[i] = theta[i]
                pbest_routes[i] = routes
                pbest_fitness[i] = res
                if res.Z_search < gbest_val:
                    gbest_val = float(res.Z_search)
                    gbest_report = float(res.Z_report)
                    gbest_theta = theta[i].copy()
                    gbest_routes = list(routes)
                    gbest_fitness = res

        # Module 4a — chaos escape on stall.
        if div_state is not None and stalled(
            div_state, theta, gbest_val,
            L_stall=L_stall, dist_eps=dist_eps, fit_eps=fit_eps,
        ):
            candidates = chaos_escape(gbest_theta, T_chaos=T_chaos)
            best_c = None
            best_c_res = None
            best_c_routes = None
            best_c_val = float("inf")
            for cand in candidates:
                # cand has shape (2, Nc) same as gbest_theta.
                c_res, c_routes = _eval_particle(cand)
                if c_res.Z_search < best_c_val:
                    best_c_val = c_res.Z_search
                    best_c = cand
                    best_c_res = c_res
                    best_c_routes = c_routes
            victim = int(rng.integers(0, M))
            theta[victim] = best_c
            if best_c_val < pbest_val[victim]:
                pbest_theta[victim] = best_c
                pbest_val[victim] = best_c_val
                pbest_report[victim] = best_c_res.Z_report
                pbest_routes[victim] = best_c_routes
                pbest_fitness[victim] = best_c_res
                if best_c_val < gbest_val:
                    gbest_val = float(best_c_val)
                    gbest_report = float(best_c_res.Z_report)
                    gbest_theta = best_c.copy()
                    gbest_routes = list(best_c_routes)
                    gbest_fitness = best_c_res
            div_state.n_chaos_fired += 1
            div_state.stall_count = 0

        # Module 4b — GLS on the top elite, periodically.
        if gls and (t % gls_every == 0):
            elite_idx = np.argsort(pbest_val)[:n_elite]
            for ei in elite_idx:
                base_routes = pbest_routes[ei]
                if base_routes is None:
                    continue
                improved_routes, improved_res = guided_local_search(
                    base_routes, g, weights or {"w1": 0.3, "w2": 0.5, "w3": 0.2, "w4": 0.0},
                    knn, R_cap=R_cap, R_tw=R_tw, fleet_capacity=fleet_capacity,
                    tau_lookup=tau_lookup, max_outer=gls_max_outer, max_inner=gls_max_inner,
                )
                if improved_res.Z_search < pbest_val[ei] - 1e-9:
                    # Re-encode is MANDATORY (spec §3.1) — otherwise the swarm
                    # never learns the local-search gain.
                    ts, tv = re_encode(improved_routes, Nc, K, depot=g.depot)
                    pbest_theta[ei] = np.stack([ts, tv]).astype(np.float32)
                    theta[ei] = pbest_theta[ei].copy()
                    pbest_val[ei] = improved_res.Z_search
                    pbest_report[ei] = improved_res.Z_report
                    pbest_routes[ei] = improved_routes
                    pbest_fitness[ei] = improved_res
                    if improved_res.Z_search < gbest_val:
                        gbest_val = float(improved_res.Z_search)
                        gbest_report = float(improved_res.Z_report)
                        gbest_theta = pbest_theta[ei].copy()
                        gbest_routes = list(improved_routes)
                        gbest_fitness = improved_res

        convergence.append(gbest_val)
        if log_every and (t % log_every == 0 or t == T):
            _log.info(
                "qpso iter",
                extra={
                    "iteration": t, "best_cost": gbest_val,
                    "wall_time_s": round(wallclock() - t0, 4),
                    "alpha": round(alpha, 4), "feasible": gbest_fitness.feasible if gbest_fitness else False,
                },
            )

    wall = wallclock() - t0
    return QPSOResult(
        best_cost=gbest_val,
        best_report_cost=gbest_report,
        best_routes=gbest_routes,
        best_fitness=gbest_fitness,  # type: ignore
        convergence=convergence,
        iterations=T,
        wall_time_s=wall,
        seed=seed,
        feasible=gbest_fitness.feasible if gbest_fitness else False,  # type: ignore
    )
