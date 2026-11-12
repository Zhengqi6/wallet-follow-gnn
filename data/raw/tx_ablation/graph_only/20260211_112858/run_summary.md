# Tx-Enhanced Follow Recommendation Results

- Device: `cuda`
- Nodes: `13746`
- Follow edges: `250000`
- Tx-sim edges: `62122`

## Test Metrics

| Model | AUC | AP | Hits@K |
|---|---:|---:|---:|
| baseline | 0.8298 | 0.8322 | 0.7105 |
| tx_enhanced | 0.8138 | 0.8164 | 0.6825 |
| delta | -0.0160 | -0.0158 | -0.0280 |
