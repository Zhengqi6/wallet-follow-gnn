# Tx-Enhanced Follow Recommendation Results

- Device: `cuda`
- Nodes: `14801`
- Follow edges: `250000`
- Tx-sim edges: `65923`

## Test Metrics

| Model | AUC | AP | Hits@K |
|---|---:|---:|---:|
| baseline | 0.6827 | 0.6354 | 0.2170 |
| tx_enhanced | 0.6904 | 0.6437 | 0.2315 |
| delta | +0.0077 | +0.0082 | +0.0145 |

## Cold-Start / Degree Buckets (Test)

| Bucket | Baseline AUC | Tx AUC | Delta AUC | Baseline Hits@K | Tx Hits@K | Delta Hits@K | N |
|---|---:|---:|---:|---:|---:|---:|---:|
| deg0 | 0.8355 | 0.8437 | +0.0083 | 0.6083 | 0.6320 | +0.0237 | 1690 |
| deg_gt10 | 0.6721 | 0.6830 | +0.0109 | 0.1920 | 0.2010 | +0.0090 | 23248 |
