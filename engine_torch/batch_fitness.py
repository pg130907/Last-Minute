"""Batched fitness — Z_search / Z_report for the whole swarm in one pass.

For each particle m:
  - Flatten K routes into one length-(Nc+2) sequence: [depot, c1, …, c_Nc, depot]
    (batch_flatten_routes handles the interleave)
  - Vehicle labels padded with 0 at both ends so route boundaries show as
    label transitions between non-zero vehicles.
  - Direct hop cost between consecutive positions comes from `dist`/`tau`/`emis`
    via torch.gather. Where a hop crosses a route boundary (interior positions
    with different non-zero vehicle labels), replace the direct hop with a
    detour through the depot: `dist[a, depot] + dist[depot, b]`.

This computes Z_report directly. For Z_search we add:
  - capacity penalty via scatter_add on `demand[order] → loads[K]`
  - (VRPTW time-window penalty is a follow-up; the fitness assumes 0 for
    now on time windows — CVRP instances like 8-node / Solomon-CVRP are the
    parity target, matching the numpy engine's default when tw fields are None)

Every accumulator is finite by construction because TorchGraph pre-substitutes
BIG_FINITE on non-edges.
"""
from __future__ import annotations

import torch

from engine_torch.batch_encoding import batch_flatten_routes
from engine_torch.graph_torch import TorchGraph

_BIG = 1.0e9


def batch_fitness(
    theta_seq: torch.Tensor,     # (M, Nc)
    theta_veh: torch.Tensor,     # (M, Nc)
    order: torch.Tensor,         # (M, Nc)  from batch_decode
    vehicle: torch.Tensor,       # (M, Nc)  from batch_decode
    tg: TorchGraph,
    K: int,
    weights: dict,
    R_cap: float = 1.0e5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (Z_search, Z_report, feasible_mask) all shape (M,).

    Feasible: no capacity overload. (Time-window feasibility deferred until
    VRPTW support is added; caller can inspect the numpy engine for now.)
    """
    Nc = order.shape[1]
    depot = tg.depot
    dev = tg.dist.device

    padded_nodes, padded_vehs = batch_flatten_routes(order, vehicle, Nc, depot=depot)

    src = padded_nodes[:, :-1]     # (M, Nc+1) — hop source
    tgt = padded_nodes[:, 1:]      # (M, Nc+1) — hop dest
    src_v = padded_vehs[:, :-1]
    tgt_v = padded_vehs[:, 1:]

    # Direct hop costs.
    hop_dist = tg.dist[src, tgt]
    hop_tau = tg.tau[src, tgt]
    hop_emis = tg.emis[src, tgt]

    # Route boundary: both endpoints are non-depot AND vehicle labels differ.
    # At those positions, replace direct hop with a depot detour.
    boundary = (src_v != 0) & (tgt_v != 0) & (src_v != tgt_v)
    if boundary.any():
        # For an M×(Nc+1) index gather, build depot-column indices.
        depot_col = torch.full_like(src, depot)
        det_dist = tg.dist[src, depot_col] + tg.dist[depot_col, tgt]
        det_tau = tg.tau[src, depot_col] + tg.tau[depot_col, tgt]
        det_emis = tg.emis[src, depot_col] + tg.emis[depot_col, tgt]
        hop_dist = torch.where(boundary, det_dist, hop_dist)
        hop_tau = torch.where(boundary, det_tau, hop_tau)
        hop_emis = torch.where(boundary, det_emis, hop_emis)

    total_dist = hop_dist.sum(dim=1).clamp_max(_BIG)
    total_tau = hop_tau.sum(dim=1).clamp_max(_BIG)
    total_emis = hop_emis.sum(dim=1).clamp_max(_BIG)

    Z_report = (
        weights["w1"] * total_dist
        + weights["w2"] * total_tau
        + weights["w3"] * total_emis
    )

    # ── Capacity penalty ──────────────────────────────────────────────────
    # Loads per (particle, vehicle-1) via scatter_add on customer demand.
    # demand tensor is length N (depot+customers); customer c has demand[c].
    # `vehicle` is (M, Nc) with values ∈ [1, K]; customer index is (j+1)
    # for the j-th slot in the angle vector.
    M = theta_seq.shape[0]
    demand_customers = tg.demand[1:1 + Nc]     # (Nc,) — depot excluded
    demand_batch = demand_customers.unsqueeze(0).expand(M, -1).contiguous()
    loads = torch.zeros((M, K), device=dev, dtype=torch.float32)
    idx = (vehicle - 1).to(torch.int64)         # (M, Nc) in [0, K-1]
    loads.scatter_add_(1, idx, demand_batch)

    overload = torch.clamp(loads - tg.capacity, min=0.0)   # (M, K)
    cap_penalty = R_cap * overload.sum(dim=1)               # (M,)

    Z_search = Z_report + cap_penalty
    feasible = cap_penalty == 0.0
    return Z_search, Z_report, feasible
