# Tx-Enhanced Follow Recommendation Results

- Device: `cuda`
- Nodes: `14801`
- Follow edges: `250000`
- Tx-sim edges: `65923`

## Test Metrics

| Model | AUC | AP | Hits@K |
|---|---:|---:|---:|
| baseline | 0.6961 | 0.6480 | 0.2400 |
| tx_enhanced | 0.6905 | 0.6443 | 0.2310 |
| delta | -0.0056 | -0.0037 | -0.0090 |

## Cold-Start / Degree Buckets (Test)

| Bucket | Baseline AUC | Tx AUC | Delta AUC | Baseline Hits@K | Tx Hits@K | Delta Hits@K | N |
|---|---:|---:|---:|---:|---:|---:|---:|
| deg0 | 0.8479 | 0.8546 | +0.0067 | 0.6178 | 0.6373 | +0.0195 | 1690 |
| deg_gt10 | 0.6844 | 0.6822 | -0.0022 | 0.2025 | 0.1965 | -0.0060 | 23248 |
