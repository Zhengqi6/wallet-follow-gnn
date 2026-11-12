# Tx-Enhanced Follow Recommendation Results

- Device: `cuda`
- Nodes: `14801`
- Follow edges: `250000`
- Tx-sim edges: `65923`

## Test Metrics

| Model | AUC | AP | Hits@K |
|---|---:|---:|---:|
| baseline | 0.6812 | 0.6327 | 0.2105 |
| tx_enhanced | 0.6866 | 0.6363 | 0.2175 |
| delta | +0.0053 | +0.0036 | +0.0070 |

## Cold-Start / Degree Buckets (Test)

| Bucket | Baseline AUC | Tx AUC | Delta AUC | Baseline Hits@K | Tx Hits@K | Delta Hits@K | N |
|---|---:|---:|---:|---:|---:|---:|---:|
| deg0 | 0.8478 | 0.8568 | +0.0090 | 0.6036 | 0.6361 | +0.0325 | 1690 |
| deg_gt10 | 0.6704 | 0.6800 | +0.0097 | 0.1800 | 0.1875 | +0.0075 | 23248 |
