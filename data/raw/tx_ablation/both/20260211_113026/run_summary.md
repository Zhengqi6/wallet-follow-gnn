# Tx-Enhanced Follow Recommendation Results

- Device: `cuda`
- Nodes: `13746`
- Follow edges: `250000`
- Tx-sim edges: `62122`

## Test Metrics

| Model | AUC | AP | Hits@K |
|---|---:|---:|---:|
| baseline | 0.8298 | 0.8322 | 0.7105 |
| tx_enhanced | 0.8307 | 0.8328 | 0.6490 |
| delta | +0.0009 | +0.0005 | -0.0615 |
