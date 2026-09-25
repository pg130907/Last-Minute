"""Report writers — Markdown tables for the comparison and ablation studies.

Numbers stay in the shape the spec §5c honesty rules require: mean±std,
best, mean runtime, feasibility fraction. Never best-only.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from benchmark.run_ablation import AblRow
from benchmark.run_baselines import Cell


def _fmt(x: float, digits: int = 3) -> str:
    if x != x:
        return "NaN"
    if abs(x) >= 1e6:
        return f"{x:.2e}"
    return f"{x:.{digits}f}"


def write_baselines_table(cells: Iterable[Cell], path: str | Path) -> None:
    cells = list(cells)
    instances = sorted({c.instance for c in cells})
    lines: list[str] = []
    lines.append("# MC-GQPSO — Baseline comparison\n")
    lines.append(
        "Report of mean ± std (population) across seeded runs per instance × baseline. "
        "Spec §5c honesty: never best-only.\n"
    )
    for inst in instances:
        rows = [c for c in cells if c.instance == inst]
        lines.append(f"\n## Instance: `{inst}`\n")
        lines.append("| Baseline | Mean ± Std | Best | Runtime (s, mean) | Feasible | N seeds |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        # Sort with our engine last for visual comparison.
        rows.sort(key=lambda c: (c.baseline == "mc_gqpso", c.mean))
        for c in rows:
            lines.append(
                f"| `{c.baseline}` | {_fmt(c.mean)} ± {_fmt(c.std)} | {_fmt(c.best)} | "
                f"{_fmt(c.runtime_mean_s, 2)} | {int(c.feasible_frac * 100)}% | {c.n_seeds} |"
            )
    Path(path).write_text("\n".join(lines) + "\n")


def write_ablation_table(rows: Iterable[AblRow], path: str | Path) -> None:
    rows = list(rows)
    instances = sorted({r.instance for r in rows})
    lines: list[str] = []
    lines.append("# MC-GQPSO — Ablation\n")
    lines.append(
        "Plain → +dual-chain → +chaos → +phase-rescatter → +GLS. Each Δ column reports "
        "the mean-cost improvement over the previous mechanism row (spec §5c: each "
        "mechanism yields a measurable reported delta).\n"
    )
    for inst in instances:
        inst_rows = [r for r in rows if r.instance == inst]
        # Preserve declared order.
        order = ["A_plain", "B_dual_chain", "C_+chaos", "D_+rescatter", "E_+gls_full"]
        inst_rows.sort(key=lambda r: order.index(r.mechanism) if r.mechanism in order else 99)
        lines.append(f"\n## Instance: `{inst}`\n")
        lines.append("| Mechanism | Mean ± Std | Best | Runtime (s) | Δ vs prev | Feasible | N |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        prev_mean = None
        for r in inst_rows:
            delta = "—" if prev_mean is None else f"{r.mean - prev_mean:+.3f}"
            lines.append(
                f"| `{r.mechanism}` | {_fmt(r.mean)} ± {_fmt(r.std)} | {_fmt(r.best)} | "
                f"{_fmt(r.runtime_mean_s, 2)} | {delta} | {int(r.feasible_frac * 100)}% | {r.n_seeds} |"
            )
            prev_mean = r.mean
    Path(path).write_text("\n".join(lines) + "\n")


