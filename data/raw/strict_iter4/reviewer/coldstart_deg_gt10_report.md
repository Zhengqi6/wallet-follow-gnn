# Cold-start / by-degree report (bucket=deg_gt10)

- Runs with this bucket: `8`
- Protocol groups: `1`

## By protocol

### split=temporal|eval_neg=two_hop|rank_negs=99|tx_feat=1|tx_graph=0|active_only=0|tx_gate=1|deg_gate=1|hard_ratio=0.3|deg_alpha=1.1|deg_beta=1.2|deg_cut=0.25|deg_sharp=10.0|deg_min=0.02|residual_decode=1|residual_scale=0.15
- n_runs: `8` | seeds: `[92, 93, 94, 95, 96, 97, 98, 99]`
- delta AUC/AP/MRR/NDCG@10 = `+0.0083` / `+0.0070` / `+0.0020` / `+0.0024`
- AUC CI95 / p = `[+0.0047, +0.0115]` / `0.0078`
- AP  CI95 / p = `[+0.0026, +0.0113]` / `0.0122`
- **Cold-start significance: AUC significant=True, AP significant=True**
