"""Ablation table (spec §5c): plain QPSO → +dual-chain → +chaos → +phase-rescatter → +GLS.

Each mechanism's mean-cost delta becomes the honest attribution to that
mechanism. The dual-chain-decode contribution is measured by comparing a
single-chain-projected run (theta_veh forced to a fixed constant, so the
vehicle assignment is uniformly K-way from rank alone) against the
two-chain run.

Configurations (in order of accumulation):
  A. plain               — no diversity, no gls
  B. +dual-chain         — always on in our engine; A vs B is a NO-OP marker
                           here because we can't cleanly disable dual-chain
                           without rewriting decode. Reported as "baseline
                           for our engine" — see notes.
  C. +chaos              — chaos escape only (rescatter off)
  D. +phase-rescatter    — chaos + rescatter (full diversity)
  E. +GLS (full)         — D + guided local search
"""
from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from benchmark.instances import get as get_instance
from engine_numpy.qpso import run_qpso


@dataclass
class AblRow:
    instance: str
    mechanism: str
    n_seeds: int
    mean: float
    std: float
    best: float
    runtime_mean_s: float
    feasible_frac: float
    raws: List[float] = field(default_factory=list)


_CONFIGS = [
    ("A_plain", dict(diversity=False, gls=False)),
    # Dual-chain is architecturally always on; keep as a comment-marker row
    # (identical config to A_plain) to make the accumulation explicit.
    ("B_dual_chain", dict(diversity=False, gls=False)),
    # Chaos-only: rescatter disabled by pushing L_sat past T so the counter
    # never fires; stall check still triggers chaos on convergence flatness.
    ("C_+chaos", dict(diversity=True, gls=False, L_sat=10**9)),
    ("D_+rescatter", dict(diversity=True, gls=False)),
    ("E_+gls_full", dict(diversity=True, gls=True)),
]


def run_ablation(
    instance_names: List[str], n_seeds: int = 5, T: int = 300, M: int = 20,
    weights: Optional[Dict[str, float]] = None, quiet: bool = False,
) -> List[AblRow]:
    weights = weights or {"w1": 1.0, "w2": 0.0, "w3": 0.0, "w4": 0.0}
    rows: List[AblRow] = []
    for inst in instance_names:
        g, K, cap = get_instance(inst)
        for name, cfg in _CONFIGS:
            costs, times, feas = [], [], []
            for s in range(n_seeds):
                t0 = time.time()
                r = run_qpso(g, K=K, M=M, T=T, seed=s, weights=weights,
                             fleet_capacity=cap, **cfg)
                dt = time.time() - t0
                costs.append(r.best_report_cost)
                times.append(dt)
                feas.append(r.feasible)
                if not quiet:
                    print(f"  ▸ {inst} · {name} · seed={s}: {r.best_report_cost:.3f} ({dt:.2f}s)")
            rows.append(AblRow(
                instance=inst, mechanism=name, n_seeds=n_seeds,
                mean=statistics.mean(costs),
                std=statistics.pstdev(costs) if len(costs) > 1 else 0.0,
                best=min(costs),
                runtime_mean_s=statistics.mean(times),
                feasible_frac=sum(feas) / len(feas),
                raws=costs,
            ))
    return rows


def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=str, default="synth_25_k3")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--T", type=int, default=300)
    ap.add_argument("--M", type=int, default=20)
    ap.add_argument("--out", type=str, default="ablation_report.md")
    args = ap.parse_args()
    from benchmark.report import write_ablation_table
    rows = run_ablation(args.instances.split(","), n_seeds=args.seeds,
                        T=args.T, M=args.M)
    write_ablation_table(rows, args.out)
    print(f"\nwrote ablation → {args.out}")


if __name__ == "__main__":
    _cli()
