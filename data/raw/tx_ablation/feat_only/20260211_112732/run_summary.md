# Tx-Enhanced Follow Recommendation Results

- Device: `cuda`
- Nodes: `13746`
- Follow edges: `250000`
- Tx-sim edges: `62122`

## Test Metrics

| Model | AUC | AP | Hits@K |
|---|---:|---:|---:|
| baseline | 0.8298 | 0.8322 | 0.7105 |
| tx_enhanced | 0.8822 | 0.8802 | 0.7160 |
| delta | +0.0525 | +0.0479 | +0.0055 |
