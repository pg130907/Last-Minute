"""engine_torch parity vs engine_numpy (spec §8 step 6).

Two-tier check:
  1. Numerical equivalence of the fitness kernel: given a shared decode,
     torch batch_fitness matches numpy evaluate() per particle within float32
     tolerance.
  2. End-to-end hard-gate replication: torch QPSO on the 8-node instance
     hits 67.5 in ≥ 3 / 20 seeded runs, mean ≤ 71.0 — the same criteria as
     tests/test_8node.py for the numpy engine.
"""
from __future__ import annotations

import statistics
import sys
import time

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/tests/", 1)[0])

from engine_numpy.encoding import random_init, decode as np_decode  # noqa: E402
from engine_numpy.fitness import evaluate  # noqa: E402
from engine_numpy.loaders import load_8node  # noqa: E402
from engine_numpy.qpso import run_qpso  # noqa: E402
from engine_torch.batch_encoding import batch_decode  # noqa: E402
from engine_torch.batch_fitness import batch_fitness  # noqa: E402
from engine_torch.batch_qpso import run_qpso_torch  # noqa: E402
from engine_torch.device import pick_torch_device  # noqa: E402
from engine_torch.graph_torch import to_torch  # noqa: E402


def _build_swarm(Nc: int, M: int, seed: int):
    rng = np.random.default_rng(seed)
    theta = np.zeros((M, 2, Nc), dtype=np.float32)
    for i in range(M):
        s, v = random_init(Nc, rng)
        theta[i, 0] = s
        theta[i, 1] = v
    return theta


def test_fitness_kernel_parity_8node() -> None:
    """Given the same swarm, torch Z_search per particle must match numpy
    evaluate() to float32 tolerance on the 8-node CVRP.
    """
    g = load_8node()
    device = torch.device("cpu")   # kernel parity is device-agnostic; CPU-torch is deterministic
    tg = to_torch(g, device=device)
    tg.capacity = 8.0

    M, Nc = 32, g.n - 1
    theta = _build_swarm(Nc, M, seed=0)
    theta_t = torch.from_numpy(theta).to(device)

    weights = {"w1": 1.0, "w2": 0.0, "w3": 0.0}

    # Numpy: decode + evaluate each particle.
    np_costs = np.zeros(M, dtype=np.float64)
    for i in range(M):
        routes, _, _ = np_decode(theta[i, 0], theta[i, 1], K=2, depot=0)
        r = evaluate(g, routes, weights=weights, fleet_capacity=8.0)
        np_costs[i] = r.Z_search

    # Torch: batched.
    order, vehicle = batch_decode(theta_t[:, 0], theta_t[:, 1], K=2)
    Z_search, Z_report, feasible = batch_fitness(
        theta_t[:, 0], theta_t[:, 1], order, vehicle, tg, K=2, weights=weights,
    )
    torch_costs = Z_search.cpu().numpy().astype(np.float64)

    diff = np.abs(np_costs - torch_costs)
    max_diff = float(diff.max())
    print(f"  ├─ fitness parity: max |Δ| = {max_diff:.6f}")
    # Both engines apply the same finite guard; direct-arithmetic diff should
    # be well within float32 accumulation noise on 8-node.
    assert max_diff < 1e-3, f"kernel mismatch: max diff = {max_diff}\n np={np_costs[:5]}\n tt={torch_costs[:5]}"


def test_torch_qpso_hits_8node_gate() -> None:
    """Torch QPSO must clear the same 8-node gate: ≥ 3 / 20 hits @ 67.5,
    mean ≤ 71.0.
    """
    g = load_8node()
    device = pick_torch_device(use_gpu=True)
    print(f"  ├─ device: {device}")
    bests = []
    t0 = time.time()
    for s in range(20):
        r = run_qpso_torch(
            g, K=2, M=20, T=500, seed=s,
            weights={"w1": 1.0, "w2": 0.0, "w3": 0.0},
            fleet_capacity=8.0, device=device,
        )
        bests.append(r.best_report_cost)
    dt = time.time() - t0
    hits = sum(1 for b in bests if abs(b - 67.5) < 1e-2)
    mean_b = statistics.mean(bests)
    print(f"  ├─ per-run bests: {[round(b, 3) for b in bests]}")
    print(f"  ├─ hits @ 67.5   : {hits}/20")
    print(f"  ├─ mean          : {mean_b:.3f}  (cap 71.0)")
    print(f"  └─ wall          : {dt:.2f}s")
    assert hits >= 3, f"torch engine only hit gate {hits}/20 times"
    assert mean_b <= 71.0, f"torch engine mean {mean_b:.3f} > 71.0"


if __name__ == "__main__":
    test_fitness_kernel_parity_8node()
    test_torch_qpso_hits_8node_gate()
    print("test_torch_parity: OK")
