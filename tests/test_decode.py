"""Module 3.1 gate — decode∘re_encode idempotent; vehicle dist near-uniform."""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/tests/", 1)[0])

from engine_numpy.encoding import decode, random_init, re_encode  # noqa: E402


def _flatten_visit_order(routes, depot=0):
    order = []
    for r in routes:
        for n in r:
            if n != depot:
                order.append(n)
    return order


def test_decode_re_encode_idempotent_direct_mode_K2() -> None:
    """K=2 → direct decode; re_encode must produce identical routes on re-decode."""
    rng = np.random.default_rng(123)
    Nc = 8
    for _ in range(50):
        seq, veh = random_init(Nc, rng)
        routes, _, _ = decode(seq, veh, K=2, depot=0)
        seq2, veh2 = re_encode(routes, Nc, K=2, depot=0)
        routes2, _, _ = decode(seq2, veh2, K=2, depot=0)
        assert routes == routes2, f"round-trip changed routes:\n {routes}\n {routes2}"


def test_decode_re_encode_idempotent_rank_mode_K10() -> None:
    rng = np.random.default_rng(7)
    Nc = 50
    for _ in range(20):
        seq, veh = random_init(Nc, rng)
        routes, _, _ = decode(seq, veh, K=10, depot=0)
        seq2, veh2 = re_encode(routes, Nc, K=10, depot=0)
        routes2, _, _ = decode(seq2, veh2, K=10, depot=0)
        assert routes == routes2


def test_vehicle_assignment_uniform_rank_mode() -> None:
    """K=10, rank-based: each vehicle should get ~Nc/K customers ± 1."""
    rng = np.random.default_rng(0)
    Nc = 100
    K = 10
    sizes = np.zeros(K, dtype=int)
    for _ in range(30):
        seq, veh = random_init(Nc, rng)
        routes, _, vehicle = decode(seq, veh, K=K, depot=0)
        for k in range(K):
            sizes[k] += int((vehicle == k + 1).sum())
    avg_per_vehicle = sizes / 30
    # Rank-based should give exactly Nc/K per vehicle every time.
    assert (avg_per_vehicle == Nc // K).all(), (
        f"rank-based decode should give uniform sizes; got means {avg_per_vehicle}"
    )


def test_vehicle_assignment_allows_unequal_direct_mode() -> None:
    """K=2, direct: over many random inits we MUST see splits other than 4/4
    on Nc=8 — that's exactly the property that lets the 8-node 3/5 optimum
    be reachable.
    """
    rng = np.random.default_rng(31)
    Nc = 8
    seen = set()
    for _ in range(200):
        seq, veh = random_init(Nc, rng)
        _, _, vehicle = decode(seq, veh, K=2, depot=0)
        seen.add(int((vehicle == 1).sum()))
    # Expect a range around 4 including 3 and 5 (the split sizes needed for
    # the 8-node optimum). If ONLY 4 appears, direct decode is broken.
    assert 3 in seen or 5 in seen, f"direct decode never produced 3/5 split; sizes seen: {seen}"
    assert 4 in seen, f"direct decode never produced balanced 4/4: {seen}"


def test_decode_covers_all_customers() -> None:
    """Every customer 1..Nc must appear in exactly one route."""
    rng = np.random.default_rng(2)
    Nc = 12
    for K in (2, 3, 5, 6, 10):
        seq, veh = random_init(Nc, rng)
        routes, _, _ = decode(seq, veh, K=K, depot=0)
        visits = _flatten_visit_order(routes, depot=0)
        assert sorted(visits) == list(range(1, Nc + 1)), (
            f"K={K}: coverage broken. visits={visits}"
        )
        # Every route starts and ends at depot.
        for r in routes:
            assert r[0] == 0 and r[-1] == 0


if __name__ == "__main__":
    test_decode_re_encode_idempotent_direct_mode_K2()
    test_decode_re_encode_idempotent_rank_mode_K10()
    test_vehicle_assignment_uniform_rank_mode()
    test_vehicle_assignment_allows_unequal_direct_mode()
    test_decode_covers_all_customers()
    print("test_decode: OK")
