# Tx-Enhanced Follow Recommendation Results

- Device: `cuda`
- Nodes: `14801`
- Follow edges: `250000`
- Tx-sim edges: `65923`

## Test Metrics

| Model | AUC | AP | Hits@K |
|---|---:|---:|---:|
| baseline | 0.6920 | 0.6419 | 0.2255 |
| tx_enhanced | 0.6930 | 0.6439 | 0.2270 |
| delta | +0.0010 | +0.0020 | +0.0015 |

## Cold-Start / Degree Buckets (Test)

| Bucket | Baseline AUC | Tx AUC | Delta AUC | Baseline Hits@K | Tx Hits@K | Delta Hits@K | N |
|---|---:|---:|---:|---:|---:|---:|---:|
| deg0 | 0.8427 | 0.8540 | +0.0113 | 0.6201 | 0.6361 | +0.0160 | 1690 |
| deg_gt10 | 0.6777 | 0.6815 | +0.0038 | 0.1920 | 0.1935 | +0.0015 | 23248 |
