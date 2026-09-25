"""Genetic algorithm on permutation + partition (K-way split point) chromosome.

Chromosome: (permutation of 1..Nc, K-1 monotone split cutoffs). Crossover:
order-1 (OX) on the permutation, uniform on splits. Mutation: swap 2 genes
+ jitter a split.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from baselines.common import BaselineResult
from common.bootstrap import set_all_seeds
from engine_numpy.fitness import evaluate
from engine_numpy.graph import Graph


def _chromo_to_routes(perm: np.ndarray, cuts: np.ndarray, K: int, depot: int) -> list[list[int]]:
    cuts_sorted = np.sort(np.clip(cuts, 0, len(perm)).astype(int))
    routes = []
    prev = 0
    for k in range(K):
        end = int(cuts_sorted[k]) if k < K - 1 else len(perm)
        end = max(end, prev)
        seg = [int(x) for x in perm[prev:end]]
        routes.append([depot] + seg + [depot])
        prev = end
    return routes


def _ox_crossover(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(a)
    i, j = sorted(rng.integers(0, n, size=2).tolist())
    if i == j:
        return a.copy()
    child = np.full(n, -1, dtype=a.dtype)
    child[i:j] = a[i:j]
    used = set(child[i:j].tolist())
    idx = j % n
    for k in range(n):
        gene = b[(j + k) % n]
        if gene not in used:
            child[idx] = gene
            used.add(int(gene))
            idx = (idx + 1) % n
    return child


def run(
    g: Graph, K: int, M: int = 20, T: int = 300, seed: int = 0,
    weights: Optional[dict] = None, fleet_capacity: Optional[float] = None,
    p_mut: float = 0.2,
) -> BaselineResult:
    set_all_seeds(seed)
    rng = np.random.default_rng(seed)
    Nc = g.n - 1
    t0 = time.time()

    def _rand_chromo():
        perm = rng.permutation(np.arange(1, Nc + 1))
        cuts = np.sort(rng.integers(0, Nc + 1, size=K - 1)) if K > 1 else np.array([Nc])
        return perm, cuts.astype(int)

    pop_perm = [None] * M
    pop_cuts = [None] * M
    pop_val = np.full(M, np.inf)
    pop_report = np.full(M, np.inf)
    pop_routes = [None] * M

    def _eval(perm, cuts):
        routes = _chromo_to_routes(perm, cuts, K, g.depot)
        res = evaluate(g, routes, weights=weights, fleet_capacity=fleet_capacity)
        return res, routes

    for i in range(M):
        pop_perm[i], pop_cuts[i] = _rand_chromo()
        res, routes = _eval(pop_perm[i], pop_cuts[i])
        pop_val[i] = res.Z_search; pop_report[i] = res.Z_report; pop_routes[i] = routes

    def _best_idx():
        return int(np.argmin(pop_val))

    b = _best_idx()
    best_val = float(pop_val[b]); best_report = float(pop_report[b])
    best_routes = pop_routes[b]
    conv = [best_val]

    for _ in range(T):
        # Tournament(3) selection + OX + mutation.
        for i in range(M):
            cand = rng.integers(0, M, size=3)
            a = cand[int(np.argmin(pop_val[cand]))]
            cand2 = rng.integers(0, M, size=3)
            c = cand2[int(np.argmin(pop_val[cand2]))]
            child_perm = _ox_crossover(pop_perm[a], pop_perm[c], rng)
            # Uniform cut crossover.
            mask = rng.uniform(0, 1, size=len(pop_cuts[a])) < 0.5
            child_cuts = np.where(mask, pop_cuts[a], pop_cuts[c])
            # Mutation.
            if rng.uniform() < p_mut:
                x, y = rng.integers(0, Nc, size=2)
                child_perm[x], child_perm[y] = child_perm[y], child_perm[x]
            if rng.uniform() < p_mut and K > 1:
                which = int(rng.integers(0, K - 1))
                child_cuts[which] = int(np.clip(child_cuts[which] + rng.integers(-2, 3), 0, Nc))
            res, routes = _eval(child_perm, child_cuts)
            # Replace worst if better (steady-state GA).
            w_i = int(np.argmax(pop_val))
            if res.Z_search < pop_val[w_i]:
                pop_perm[w_i] = child_perm; pop_cuts[w_i] = child_cuts
                pop_val[w_i] = res.Z_search; pop_report[w_i] = res.Z_report
                pop_routes[w_i] = routes
                if res.Z_search < best_val:
                    best_val = float(res.Z_search); best_report = float(res.Z_report)
                    best_routes = routes
        conv.append(best_val)

    return BaselineResult(
        name="ga", best_cost=best_report, best_cost_search=best_val,
        best_routes=best_routes or [], wall_time_s=time.time() - t0,
        feasible=(best_val == best_report), seed=seed, iterations=T, convergence=conv,
    )
