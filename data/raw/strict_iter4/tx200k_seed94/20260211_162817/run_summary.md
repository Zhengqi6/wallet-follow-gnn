# Tx-Enhanced Follow Recommendation Results

- Device: `cuda`
- Nodes: `14801`
- Follow edges: `250000`
- Tx-sim edges: `65923`

## Test Metrics

| Model | AUC | AP | Hits@K |
|---|---:|---:|---:|
| baseline | 0.6859 | 0.6406 | 0.2055 |
| tx_enhanced | 0.6947 | 0.6502 | 0.2160 |
| delta | +0.0088 | +0.0096 | +0.0105 |

## Cold-Start / Degree Buckets (Test)

| Bucket | Baseline AUC | Tx AUC | Delta AUC | Baseline Hits@K | Tx Hits@K | Delta Hits@K | N |
|---|---:|---:|---:|---:|---:|---:|---:|
| deg0 | 0.8341 | 0.8432 | +0.0091 | 0.6201 | 0.6391 | +0.0189 | 1690 |
| deg_gt10 | 0.6704 | 0.6820 | +0.0116 | 0.1750 | 0.1915 | +0.0165 | 23248 |
