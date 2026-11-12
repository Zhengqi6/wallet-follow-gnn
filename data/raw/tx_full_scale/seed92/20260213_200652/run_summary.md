# Tx-Enhanced Follow Recommendation Results

- Device: `cuda`
- Nodes: `28517`
- Follow edges: `250000`
- Tx-sim edges: `110232`

## Test Metrics

| Model | AUC | AP | Hits@K |
|---|---:|---:|---:|
| baseline | 0.4264 | 0.4601 | 0.0905 |
| tx_enhanced | 0.5584 | 0.5099 | 0.0940 |
| delta | +0.1319 | +0.0498 | +0.0035 |

## Cold-Start / Degree Buckets (Test)

| Bucket | Baseline AUC | Tx AUC | Delta AUC | Baseline Hits@K | Tx Hits@K | Delta Hits@K | N |
|---|---:|---:|---:|---:|---:|---:|---:|
| deg0 | 0.7125 | 0.5547 | -0.1578 | 0.5687 | 0.3565 | -0.2121 | 1565 |
| deg_gt10 | 0.4118 | 0.5642 | +0.1524 | 0.0605 | 0.0780 | +0.0175 | 23433 |
