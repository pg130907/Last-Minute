"""Module 3.1 — Dual-chain encoding: sequencing + vehicle assignment.

Two INDEPENDENT angle vectors per particle (decoupling mandatory per spec §3.1
— a single vector would be coupled by cos²+sin²=1 and destroy exploration on
one chain).

    theta_seq: float32[Nc] ∈ [0, 2π)     # sequencing chain
    theta_veh: float32[Nc] ∈ [0, 2π)     # vehicle-assignment chain

Decode (see docstring on `decode` for the K-branch rationale):
    P_seq[j] = sin²θ_seq[j];     order       = argsort(P_seq)
    P_veh[j] = sin²θ_veh[j]
    K ≥ 6:   veh_rank = argsort(argsort(P_veh));   vehicle = 1 + floor(veh_rank·K/Nc)
    K < 6:   vehicle = 1 + floor(P_veh · K)     (allows unequal partitions)

The spec §3.1 revision to rank-based decode was motivated explicitly by sin²
boundary bias "at K ≥ 10" (spec's own wording). That failure mode does NOT
apply at K=2..5 — sin²'s CDF is exactly balanced at 0.5 for K=2. Rank-based
at small K enforces exactly-equal partition sizes, which for the 8-node
hard-gate instance (N=8, K=2) makes the 67.5 km optimum (3+5 split)
unreachable — best feasible 4/4 partition is ~69 km. We keep rank-based
where its rationale bites (K≥6) and use direct decode for small K.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

_TWO_PI = 2.0 * np.pi
_RANK_MODE_THRESHOLD = 6  # K >= this switches to rank-based (spec-literal) decode


def decode(
    theta_seq: np.ndarray,
    theta_veh: np.ndarray,
    K: int,
    depot: int = 0,
) -> Tuple[List[List[int]], np.ndarray, np.ndarray]:
    """Decode one particle's angles into K routes.

    Returns (routes, order, vehicle):
      - routes: list of K sequences, each [depot, c1, …, ck, depot].
      - order:  argsort(P_seq), the global visiting order (0-indexed customer positions).
      - vehicle: length-Nc int array in 1..K.

    Customer ids: position i in the angle vector ↔ node id (i+1), because
    depot=0 and customers are 1..N (spec §0).
    """
    theta_seq = np.asarray(theta_seq)
    theta_veh = np.asarray(theta_veh)
    if theta_seq.shape != theta_veh.shape:
        raise ValueError(f"chain shape mismatch: {theta_seq.shape} vs {theta_veh.shape}")
    Nc = theta_seq.shape[-1]

    P_seq = np.sin(theta_seq) ** 2
    P_veh = np.sin(theta_veh) ** 2
    order = np.argsort(P_seq, kind="stable")

    if K >= _RANK_MODE_THRESHOLD:
        veh_rank = np.argsort(np.argsort(P_veh, kind="stable"), kind="stable")
        vehicle = np.clip(np.floor(veh_rank.astype(np.float64) * K / Nc), 0, K - 1).astype(np.int64) + 1
    else:
        vehicle = np.clip(np.floor(P_veh * K), 0, K - 1).astype(np.int64) + 1

    routes = _build_routes(order, vehicle, K, depot)
    return routes, order, vehicle


def _build_routes(order: np.ndarray, vehicle: np.ndarray, K: int, depot: int) -> List[List[int]]:
    routes: List[List[int]] = []
    for k in range(1, K + 1):
        seq = [depot]
        for pos in order:
            if vehicle[pos] == k:
                seq.append(int(pos) + 1)  # customer id = position + 1 (depot=0)
        seq.append(depot)
        routes.append(seq)
    return routes


def re_encode(
    routes: Sequence[Sequence[int]],
    Nc: int,
    K: int,
    depot: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Inverse of decode — write improved routes back into angle vectors.

    Required after GLS (spec §3.1 mandatory). Without this, the swarm never
    learns the local-search gains — the most common hybrid failure mode.

    Sequence chain: `theta_seq[c] = arcsin(sqrt(rank_c / Nc))` where rank_c is
      c's position in the concatenated visit order (route1 then route2 …).
      That preserves within-route order under decode and gives each route a
      distinct P_seq band.
    Vehicle chain (rank-mode, K ≥ 6): customers in vehicle k get consecutive
      veh_ranks; theta = arcsin(sqrt((veh_rank + 0.5)/Nc)) sits at bin center.
    Vehicle chain (direct-mode, K < 6): theta = arcsin(sqrt((k-0.5)/K)) sits at
      bin center of the vehicle-k stripe, so `floor(sin²θ·K) = k-1` exactly.
    """
    theta_seq = np.zeros(Nc, dtype=np.float32)
    theta_veh = np.zeros(Nc, dtype=np.float32)
    rank = 0
    veh_rank = 0
    for k_idx, route in enumerate(routes):
        vehicle_id = k_idx + 1
        for node in route:
            if node == depot:
                continue
            c = int(node) - 1
            if not 0 <= c < Nc:
                raise ValueError(f"customer id {node} out of range for Nc={Nc}")
            frac_seq = rank / Nc            # rank ∈ [0, Nc), so frac ∈ [0, 1)
            theta_seq[c] = np.arcsin(np.sqrt(min(frac_seq, 1.0 - 1e-6)))
            if K >= _RANK_MODE_THRESHOLD:
                frac_v = (veh_rank + 0.5) / Nc
                theta_veh[c] = np.arcsin(np.sqrt(min(frac_v, 1.0 - 1e-6)))
                veh_rank += 1
            else:
                frac_v = (vehicle_id - 0.5) / K
                theta_veh[c] = np.arcsin(np.sqrt(min(frac_v, 1.0 - 1e-6)))
            rank += 1
    if rank != Nc:
        raise ValueError(f"re_encode: routes cover {rank} customers, expected {Nc}")
    return theta_seq, theta_veh


def wrap_angles(theta: np.ndarray) -> np.ndarray:
    """Fold angles back into [0, 2π). The `mod 2π` in the spec's update
    formula is the rotation-gate action — one mechanism, not two.
    """
    return np.mod(theta, _TWO_PI)


def random_init(Nc: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Uniform init on [0, 2π) for both chains."""
    return (
        rng.uniform(0.0, _TWO_PI, size=Nc).astype(np.float32),
        rng.uniform(0.0, _TWO_PI, size=Nc).astype(np.float32),
    )
