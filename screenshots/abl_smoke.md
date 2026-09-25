# MC-GQPSO — Ablation

Plain → +dual-chain → +chaos → +phase-rescatter → +GLS. Each Δ column reports the mean-cost improvement over the previous mechanism row (spec §5c: each mechanism yields a measurable reported delta).


## Instance: `synth_25_k3`

| Mechanism | Mean ± Std | Best | Runtime (s) | Δ vs prev | Feasible | N |
|---|---:|---:|---:|---:|---:|---:|
| `A_plain` | 1273.814 ± 22.740 | 1251.074 | 0.09 | — | 0% | 2 |
| `B_dual_chain` | 1273.814 ± 22.740 | 1251.074 | 0.09 | +0.000 | 0% | 2 |
| `C_+chaos` | 1273.814 ± 22.740 | 1251.074 | 0.09 | +0.000 | 0% | 2 |
| `D_+rescatter` | 1273.814 ± 22.740 | 1251.074 | 0.09 | +0.000 | 0% | 2 |
| `E_+gls_full` | 670.608 ± 10.318 | 660.291 | 1.26 | -603.206 | 0% | 2 |
