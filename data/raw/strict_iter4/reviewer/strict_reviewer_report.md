# Strict Reviewer Report (Tx-Follow)

- Total runs scanned: `8`
- Protocol groups: `1`

## Decisions by Protocol

### split=temporal|eval_neg=two_hop|rank_negs=99|tx_feat=1|tx_graph=0|active_only=0|tx_gate=1|deg_gate=1|hard_ratio=0.3|deg_alpha=1.1|deg_beta=1.2|deg_cut=0.25|deg_sharp=10.0|deg_min=0.02|residual_decode=1|residual_scale=0.15
- Decision: **PASS**
- n_runs: `8` | seeds: `[92, 93, 94, 95, 96, 97, 98, 99]`
- delta AUC/AP/MRR/NDCG@10 = `+0.0045` / `+0.0058` / `+0.0031` / `+0.0032`
- AUC CI95 / p = `[+0.0011, +0.0076]` / `0.0274`
- AP CI95 / p = `[+0.0014, +0.0100]` / `0.0310`

## Top-Tier Gate

- PASS requires: multi-seed stability + significant AUC/AP + ranking lift + no calibration collapse.
- If WEAK_REJECT/REJECT: treat as not ready for top-tier submission.
