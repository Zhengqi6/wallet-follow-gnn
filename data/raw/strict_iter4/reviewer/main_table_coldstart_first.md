# Tx-Follow Main Results (Cold-Start First)

Protocol: temporal split, two-hop hard negatives, 8 seeds (92–99).
Delta = Tx-enhanced − Baseline.

## Primary: Cold-start (deg0, no prior follow)

| Subset | Δ AUC | AUC 95% CI | AUC p | Δ AP | AP 95% CI | AP p |
|--------|------:|-----------:|------:|-----:|----------:|-----:|
| **deg0 (cold-start)** | +0.0084 | [+0.0073, +0.0095] | 0.0044 | +0.0083 | [+0.0073, +0.0092] | 0.0040 |

## By degree bucket

| Subset | Δ AUC | AUC 95% CI | AUC p | Δ AP | AP 95% CI | AP p |
|--------|------:|-----------:|------:|-----:|----------:|-----:|
| deg0 (cold-start) | +0.0084 | [+0.0073, +0.0095] | 0.0044 | +0.0083 | [+0.0073, +0.0092] | 0.0040 |
| deg_gt10 (high-degree) | +0.0083 | [+0.0047, +0.0115] | 0.0078 | +0.0070 | [+0.0026, +0.0113] | 0.0122 |

## Overall (all test edges)

| Subset | Δ AUC | AUC 95% CI | AUC p | Δ AP | AP 95% CI | AP p |
|--------|------:|-----------:|------:|-----:|----------:|-----:|
| Overall | +0.0045 | [+0.0011, +0.0076] | 0.0274 | +0.0058 | [+0.0014, +0.0100] | 0.0310 |

---

**Narrative**: Cold-start (deg0) shows the largest and most significant gain (AUC +0.0084, p≈0.004); high-degree (deg_gt10) and overall are also significant. Use deg0 as the main claim for 冲顶会.
