# Tx-Enhanced Follow Recommendation Results

- Device: `cuda`
- Nodes: `14801`
- Follow edges: `250000`
- Tx-sim edges: `65923`

## Test Metrics

| Model | AUC | AP | Hits@K |
|---|---:|---:|---:|
| baseline | 0.6734 | 0.6212 | 0.1900 |
| tx_enhanced | 0.6834 | 0.6354 | 0.2080 |
| delta | +0.0100 | +0.0142 | +0.0180 |

## Cold-Start / Degree Buckets (Test)

| Bucket | Baseline AUC | Tx AUC | Delta AUC | Baseline Hits@K | Tx Hits@K | Delta Hits@K | N |
|---|---:|---:|---:|---:|---:|---:|---:|
| deg0 | 0.8430 | 0.8525 | +0.0095 | 0.6154 | 0.6396 | +0.0243 | 1690 |
| deg_gt10 | 0.6643 | 0.6784 | +0.0141 | 0.1750 | 0.1880 | +0.0130 | 23248 |
