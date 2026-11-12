# Tx-Enhanced Follow Recommendation Results

- Device: `cuda`
- Nodes: `14801`
- Follow edges: `250000`
- Tx-sim edges: `65923`

## Test Metrics

| Model | AUC | AP | Hits@K |
|---|---:|---:|---:|
| baseline | 0.6807 | 0.6304 | 0.1980 |
| tx_enhanced | 0.6884 | 0.6442 | 0.2235 |
| delta | +0.0078 | +0.0137 | +0.0255 |

## Cold-Start / Degree Buckets (Test)

| Bucket | Baseline AUC | Tx AUC | Delta AUC | Baseline Hits@K | Tx Hits@K | Delta Hits@K | N |
|---|---:|---:|---:|---:|---:|---:|---:|
| deg0 | 0.8326 | 0.8398 | +0.0072 | 0.6071 | 0.6385 | +0.0314 | 1690 |
| deg_gt10 | 0.6735 | 0.6861 | +0.0126 | 0.1620 | 0.1960 | +0.0340 | 23248 |
