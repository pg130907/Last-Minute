"""Run each baseline × instance × seed. Aggregate mean/std/best/runtime.

Usage (script mode):
    python -m benchmark.run_baselines --instances 8node,synth_25_k3 \\
        --baselines ortools,ga,sa,pso,qpso_plain,mc_gqpso --seeds 5 --T 300

Programmatic use: `run_matrix(instance_names, baseline_names, n_seeds, T, M)`.
"""
from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from baselines.common import BaselineResult
from benchmark.instances import get as get_instance


# ── Baseline registry ──────────────────────────────────────────────────────
def _run_mc_gqpso(g, K, seed, weights, cap, M, T):
    """Our full engine, all mechanisms enabled — the headline algorithm."""
    from engine_numpy.qpso import run_qpso
    t0 = time.time()
    r = run_qpso(g, K=K, M=M, T=T, seed=seed, weights=weights,
                 fleet_capacity=cap, diversity=True, gls=True)
    return BaselineResult(
        name="mc_gqpso", best_cost=r.best_report_cost,
        best_cost_search=r.best_cost, best_routes=r.best_routes,
        wall_time_s=time.time() - t0, feasible=r.feasible, seed=seed,
        iterations=T, convergence=r.convergence,
    )


def _run_baseline(name: str, g, K, seed, weights, cap, M, T) -> BaselineResult:
    if name == "mc_gqpso":
        return _run_mc_gqpso(g, K, seed, weights, cap, M, T)
    if name == "qpso_plain":
        from baselines import qpso_plain
        return qpso_plain.run(g, K, M=M, T=T, seed=seed, weights=weights, fleet_capacity=cap)
    if name == "pso":
        from baselines import pso
        return pso.run(g, K, M=M, T=T, seed=seed, weights=weights, fleet_capacity=cap)
    if name == "ga":
        from baselines import ga
        return ga.run(g, K, M=M, T=T, seed=seed, weights=weights, fleet_capacity=cap)
    if name == "sa":
        from baselines import sa
        return sa.run(g, K, T_iters=M * T, seed=seed, weights=weights, fleet_capacity=cap)
    if name == "ortools":
        from baselines import ortools_vrp
        return ortools_vrp.run(g, K, seed=seed, weights=weights, fleet_capacity=cap,
                               time_limit_s=5)
    raise ValueError(f"unknown baseline: {name}")


# ── Aggregation ────────────────────────────────────────────────────────────
@dataclass
class Cell:
    instance: str
    baseline: str
    n_seeds: int
    mean: float
    std: float
    best: float
    runtime_mean_s: float
    feasible_frac: float
    raw: List[BaselineResult] = field(default_factory=list)


def run_matrix(
    instance_names: List[str],
    baseline_names: List[str],
    n_seeds: int = 5,
    T: int = 200,
    M: int = 20,
    weights: Optional[Dict[str, float]] = None,
    quiet: bool = False,
) -> List[Cell]:
    weights = weights or {"w1": 1.0, "w2": 0.0, "w3": 0.0, "w4": 0.0}
    cells: List[Cell] = []
    for inst in instance_names:
        g, K, cap = get_instance(inst)
        for bname in baseline_names:
            raws: List[BaselineResult] = []
            for s in range(n_seeds):
                if not quiet:
                    print(f"  ▸ {inst} · {bname} · seed={s} …", end=" ", flush=True)
                r = _run_baseline(bname, g, K, s, weights, cap, M, T)
                raws.append(r)
                if not quiet:
                    print(f"cost={r.best_cost:.3f} ({r.wall_time_s:.2f}s)"
                          + ("" if r.feasible else "  ⚠ infeasible"))
            costs = [r.best_cost for r in raws]
            cells.append(Cell(
                instance=inst, baseline=bname, n_seeds=n_seeds,
                mean=statistics.mean(costs),
                std=statistics.pstdev(costs) if len(costs) > 1 else 0.0,
                best=min(costs),
                runtime_mean_s=statistics.mean([r.wall_time_s for r in raws]),
                feasible_frac=sum(1 for r in raws if r.feasible) / len(raws),
                raw=raws,
            ))
    return cells


def _cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=str, default="8node")
    ap.add_argument("--baselines", type=str,
                    default="qpso_plain,mc_gqpso,ga,sa,pso,ortools")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--M", type=int, default=20)
    ap.add_argument("--out", type=str, default="benchmark_report.md")
    args = ap.parse_args()

    from benchmark.report import write_baselines_table
    cells = run_matrix(
        args.instances.split(","), args.baselines.split(","),
        n_seeds=args.seeds, T=args.T, M=args.M,
    )
    write_baselines_table(cells, args.out)
    print(f"\nwrote report → {args.out}")


if __name__ == "__main__":
    _cli()
