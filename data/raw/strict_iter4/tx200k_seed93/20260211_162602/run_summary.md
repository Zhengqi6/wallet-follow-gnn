# Tx-Enhanced Follow Recommendation Results

- Device: `cuda`
- Nodes: `14801`
- Follow edges: `250000`
- Tx-sim edges: `65923`

## Test Metrics

| Model | AUC | AP | Hits@K |
|---|---:|---:|---:|
| baseline | 0.6887 | 0.6446 | 0.2190 |
| tx_enhanced | 0.6900 | 0.6433 | 0.2170 |
| delta | +0.0012 | -0.0012 | -0.0020 |

## Cold-Start / Degree Buckets (Test)

| Bucket | Baseline AUC | Tx AUC | Delta AUC | Baseline Hits@K | Tx Hits@K | Delta Hits@K | N |
|---|---:|---:|---:|---:|---:|---:|---:|
| deg0 | 0.8379 | 0.8438 | +0.0060 | 0.6142 | 0.6408 | +0.0266 | 1690 |
| deg_gt10 | 0.6798 | 0.6855 | +0.0057 | 0.1915 | 0.1860 | -0.0055 | 23248 |
