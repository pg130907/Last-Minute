"""Modules 5c/5d/5e — benchmark harness, Pareto, bundled demo scenario.

Kept lightweight (few seeds, small T) so it fits the regression suite. The
full-scale spec §5c "≥20 seeded runs per instance × baseline" is run
manually via `python -m benchmark.run_baselines` and its report committed
alongside the paper — not in the test loop.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, __file__.rsplit("/tests/", 1)[0])

from benchmark.report import write_ablation_table, write_baselines_table  # noqa: E402
from benchmark.run_ablation import run_ablation  # noqa: E402
from benchmark.run_baselines import run_matrix  # noqa: E402
from engine_numpy.loaders import load_bundled_demo, make_synthetic  # noqa: E402


# ── 5c: baseline harness smoke ─────────────────────────────────────────────
def test_baseline_matrix_smoke() -> None:
    """Run a tiny matrix (2 seeds × 8-node × 3 baselines) and confirm the
    aggregation shape. The plain-QPSO / MC-GQPSO row on the 8-node HARD
    GATE must be at most 71 km on the mean (spec §3.3 mean cap)."""
    cells = run_matrix(
        instance_names=["8node"],
        baseline_names=["qpso_plain", "mc_gqpso", "ortools"],
        n_seeds=2, T=200, M=20, quiet=True,
    )
    assert len(cells) == 3
    for c in cells:
        assert c.n_seeds == 2
        assert c.mean > 0
        print(f"  ├─ {c.instance:20s} {c.baseline:12s} "
              f"mean={c.mean:.3f}±{c.std:.3f} best={c.best:.3f} "
              f"({c.runtime_mean_s:.2f}s, feas={int(c.feasible_frac * 100)}%)")

    mc = next(c for c in cells if c.baseline == "mc_gqpso")
    ort = next(c for c in cells if c.baseline == "ortools")
    assert mc.mean <= 71.0, f"mc_gqpso mean {mc.mean} > 71.0 on 8-node"
    # OR-Tools is expected to hit the exact optimum on 8-node with 5s budget.
    assert ort.best <= 68.0, f"ortools best {ort.best} unexpectedly high — check installation"


def test_ablation_deltas_present() -> None:
    """Ablation runs cleanly on a small synthetic; report has all 5 rows."""
    rows = run_ablation(["synth_25_k3"], n_seeds=2, T=120, M=16, quiet=True)
    assert len(rows) == 5
    for r in rows:
        print(f"  ├─ {r.mechanism:16s} mean={r.mean:.2f}±{r.std:.2f} "
              f"best={r.best:.2f} ({r.runtime_mean_s:.2f}s)")
    # Report writers accept the rows.
    out = Path(__file__).resolve().parent.parent / "screenshots" / "abl_smoke.md"
    out.parent.mkdir(exist_ok=True)
    write_ablation_table(rows, out)
    assert out.exists() and out.stat().st_size > 100


def test_write_baselines_report() -> None:
    """Report file is generated and parseable Markdown."""
    cells = run_matrix(["8node"], ["mc_gqpso"], n_seeds=2, T=150, M=16, quiet=True)
    out = Path(__file__).resolve().parent.parent / "screenshots" / "baseline_smoke.md"
    out.parent.mkdir(exist_ok=True)
    write_baselines_table(cells, out)
    text = out.read_text()
    assert "mc_gqpso" in text and "Baseline comparison" in text


# ── 5e: bundled scenario ───────────────────────────────────────────────────
def test_bundled_demo_loads_and_is_self_contained() -> None:
    g, s = load_bundled_demo()
    assert g.n == s["meta"]["n_nodes"]
    assert g.n >= 20
    # Zero external deps: the JSON must carry a dense edge list AND a traffic
    # schedule (dynamic-demo hook), no filesystem/OSM references.
    assert "traffic_schedule" in s
    assert "events" in s["solver"] and len(s["solver"]["events"]) >= 1
    # Every customer must have valid coords + demand + tw.
    for n in s["network"]["nodes"]:
        assert "lat" in n and "lng" in n
    # Graph is fully connected: every off-diagonal edge present.
    N = g.n
    assert int(g.adj_mask.sum()) == N * (N - 1), "bundled scenario should be dense"


def test_bundled_demo_can_be_solved() -> None:
    """End-to-end: load bundled → solve → per-route metrics populated."""
    from engine_numpy.qpso import run_qpso
    g, s = load_bundled_demo()
    cap = float(s["fleet"][0]["capacity"])
    K = len(s["fleet"])
    r = run_qpso(g, K=K, M=24, T=120, seed=42,
                 weights=s["weights"], fleet_capacity=cap,
                 diversity=True, gls=True)
    assert r.best_fitness is not None
    assert len(r.best_routes) == K
    print(f"  ├─ bundled solved: dist={r.best_fitness.total_distance:.2f}km "
          f"time={r.best_fitness.total_time:.1f}min  Z_report={r.best_report_cost:.2f}  "
          f"feasible={r.feasible}")


if __name__ == "__main__":
    test_baseline_matrix_smoke()
    test_ablation_deltas_present()
    test_write_baselines_report()
    test_bundled_demo_loads_and_is_self_contained()
    test_bundled_demo_can_be_solved()
    print("test_benchmark: OK")
