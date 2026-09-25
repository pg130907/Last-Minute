"""Batched dual-chain decode — the whole swarm in one torch.argsort call.

Same K-branch rule as engine_numpy.encoding.decode (K<6 → direct, K≥6 →
rank-based). Returns index tensors only; the batch-fitness layer walks the
routes without ever materializing per-particle Python lists.
"""
from __future__ import annotations

import torch

_RANK_MODE_THRESHOLD = 6


def batch_decode(
    theta_seq: torch.Tensor,       # (M, Nc)
    theta_veh: torch.Tensor,       # (M, Nc)
    K: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (order, vehicle) both shape (M, Nc), int64.

    order[m, i] = customer position (0-indexed) at visit rank i for particle m.
    vehicle[m, j] ∈ [1, K] = vehicle assigned to customer position j.
    """
    if theta_seq.shape != theta_veh.shape:
        raise ValueError(f"chain shape mismatch: {theta_seq.shape} vs {theta_veh.shape}")
    M, Nc = theta_seq.shape

    P_seq = torch.sin(theta_seq) ** 2
    P_veh = torch.sin(theta_veh) ** 2

    order = torch.argsort(P_seq, dim=-1, stable=True)

    if K >= _RANK_MODE_THRESHOLD:
        veh_rank = torch.argsort(torch.argsort(P_veh, dim=-1, stable=True), dim=-1, stable=True)
        vehicle = torch.clamp((veh_rank.to(torch.float32) * K / Nc).floor().to(torch.int64), 0, K - 1) + 1
    else:
        vehicle = torch.clamp((P_veh * K).floor().to(torch.int64), 0, K - 1) + 1

    return order, vehicle


def batch_flatten_routes(
    order: torch.Tensor,          # (M, Nc) int64
    vehicle: torch.Tensor,        # (M, Nc) int64 ∈ [1, K]
    Nc: int,
    depot: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sort customers per particle by (vehicle, visit_order); return
    (padded_nodes, padded_vehicles) both shape (M, Nc+2).

    padded_nodes[m, 0] == padded_nodes[m, -1] == depot; interior columns hold
    the flattened concat of route_1 ++ route_2 ++ … ++ route_K (each in
    within-route visit order). padded_vehicles uses `0` as the depot marker
    so the fitness layer can detect route boundaries with a simple mask.
    """
    M = order.shape[0]
    dev = order.device
    # For each visit rank i, what vehicle it goes to.
    visit_vehicle = torch.gather(vehicle, 1, order)           # (M, Nc)
    visit_position = torch.arange(Nc, device=dev).unsqueeze(0).expand(M, -1)  # (M, Nc)
    # Composite sort key: primary = vehicle, secondary = visit_position.
    # (Nc + 1) as multiplier is safe since visit_position < Nc.
    sort_key = visit_vehicle * (Nc + 1) + visit_position       # (M, Nc)
    reorder_idx = torch.argsort(sort_key, dim=-1, stable=True) # (M, Nc)

    sorted_positions = torch.gather(order, 1, reorder_idx)     # (M, Nc)
    sorted_vehicles = torch.gather(visit_vehicle, 1, reorder_idx)  # (M, Nc)

    # Customer node id = position + 1 (depot=0, customers=1..Nc). depot=0.
    if depot != 0:
        # Safe general remapping — but our contract puts depot=0 and customers
        # at 1..Nc. Reject for now rather than silently miscompute.
        raise NotImplementedError("engine_torch currently assumes depot=0")

    node_ids = sorted_positions + 1                            # (M, Nc)

    depot_col = torch.full((M, 1), depot, dtype=node_ids.dtype, device=dev)
    zero_col = torch.zeros((M, 1), dtype=sorted_vehicles.dtype, device=dev)

    padded_nodes = torch.cat([depot_col, node_ids, depot_col], dim=1)         # (M, Nc+2)
    padded_vehs = torch.cat([zero_col, sorted_vehicles, zero_col], dim=1)      # (M, Nc+2)
    return padded_nodes, padded_vehs
