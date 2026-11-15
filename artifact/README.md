# txfollow artifact: `train_tx_enhanced_gnn.py`

This folder holds the training script behind every number in the report and poster.
The script trains two models on the same follow graph and writes their test metrics:

- a follow-only GNN baseline
- the gated, degree-aware wallet-transaction ("tx-enhanced") variant

## Where the source comes from

The original `.py` file was lost. Only its compiled form survived:
`__pycache__/train_tx_enhanced_gnn.cpython-39.pyc` (CPython 3.9).
`train_tx_enhanced_gnn.py` was rebuilt from that bytecode with `dis` and `marshal`.

**Bytecode check.** We compiled the rebuilt file with CPython 3.9 and compared it with the
original `.pyc`. All 48 code objects match instruction for instruction, including every
argument, jump target, name, cell variable and constant. That covers the module body, all
functions, the lambdas and the comprehensions. Default values and annotations match as well.
The only differences are comments and a few added docstrings, and neither changes what runs.

**Run check.** We also ran both files on the same machine with the same command (see below);
their metrics are compared under "Equivalence run".

## Environment

Use Python 3.9 (we used 3.9.13). The runs used one NVIDIA A100. If no GPU is found, the script
runs on CPU.

```bash
python3.9 -m venv .venv && source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

The CUDA 12.8 wheel of torch is pinned to match the original runs. On a machine without that
CUDA version, install `torch==2.8.0` from PyPI instead.

## Data

The script reads two JSONL files. Only the first rows of each are used, so neither file has to
be complete.

- `--links-path`: Farcaster link messages, one hub message per line.
  The script reads `message.data.linkBody` and keeps `type == "follow"` together with
  `MESSAGE_TYPE_LINK_ADD`.
- `--tx-path`: one row per (fid, address) pair, holding
  `fid`, `address` and `rpc_result.result.addressTransactions[].{hash,status,ethereum.{from,to,value}}`.

## Command: main experiment, seed 92

```bash
python train_tx_enhanced_gnn.py \
  --links-path data/links.jsonl --tx-path fid_main_addr_all_transactions.jsonl \
  --max-tx-rows 200000 --max-link-rows 1000000 --max-follow-edges 250000 \
  --split-mode temporal --negative-mode two_hop --eval-negative-mode two_hop \
  --hard-negative-ratio 0.3 --rank-negatives 99 --epochs 8 --hidden-dim 128 \
  --enhanced-use-tx-features 1 --enhanced-use-tx-graph 0 --tx-graph-weight 0 \
  --tx-feature-gate 1 --degree-aware-gate 1 --residual-tx-decode 1 \
  --residual-tx-scale 0.15 \
  --train-edge-batch-size 60000 --eval-edge-batch-size 60000 --eval-every 1 \
  --degree-gate-alpha 1.1 --degree-gate-beta 1.2 --degree-gate-cutoff 0.25 \
  --degree-gate-min 0.02 --two-hop-mid-cap 32 --two-hop-pool-cap 128 \
  --seed 92 --out-dir runs/tx200k_seed92
```

**Important:** you need the last group of flags, from `--train-edge-batch-size` to
`--two-hop-pool-cap`. They come from the `config` block of the archived run in
`data/raw/strict_iter4/tx200k_seed92/.../metrics_report.json`. Without them the script uses
its defaults (batch 200000, `--eval-every 2`, different gate and two-hop settings), and seed 92
then gives a test AUC of about 0.525 for the baseline and 0.566 for tx-enhanced, not 0.683 and
0.690. The same flags are kept in `configs/main_experiment.args`, checked against the archived
run configuration.

For all seeds of the thesis (92–99), use `scripts/run_seeds.sh` or the Slurm array job
`scripts/slurm_run_seeds.sbatch` from the repository root (see the top-level README).
Each run takes about 2–3 minutes on an A100.

Every run writes into `<out-dir>/<YYYYmmdd_HHMMSS>/`:

- `metrics_report.json`: the config, data summaries, graph statistics, and the per-epoch
  validation history of both models. It also holds test metrics (AUC, AP, logloss, Brier,
  Hits@10, MRR, NDCG@10, P@10, R@10), the same metrics by source-degree bucket, and the
  enhanced-minus-baseline deltas.
- `run_summary.md`: the main test table and the degree-bucket table.
- `analysis_plots.png`: histograms of the tx feature, the follow out-degree and the tx-graph degree.

## Notes on behaviour (kept as in the original)

- **Normalised adjacency.** `make_sparse_norm_adj` uses edge weights only for the node degree.
  Every stored entry is `1/sqrt(d_i d_j)`.
- **Temporal split.** Edges with an unknown timestamp sort after all timestamped edges, so they
  land in the test split.
- **Degree feature.** The feature is `log1p(train out-degree) / max(log1p(train out-degree))`,
  which lies in [0, 1].
- **Unused kNN graph.** A tx kNN graph is always built (`--tx-knn-k 24`). In the main
  experiment it is never used, because `--enhanced-use-tx-graph 0` is set.
- **Small run-to-run noise.** GPU sparse kernels are not bit-deterministic, so repeated runs can
  differ in the last decimal places.

## Equivalence run (cluster `i1`, A100, Python 3.9.13, torch 2.8.0+cu128, 2026-11-12)

Three runs were compared, all with the full command above at seed 92:

- **orig .pyc**: the original `train_tx_enhanced_gnn.cpython-39.pyc`, rerun today.
- **this .py**: the rebuilt source in this folder.
- **archived**: the February run in `data/raw/strict_iter4/tx200k_seed92/20260211_162344`.

All three produce identical data summaries and graph statistics: 14,801 nodes and 250,000
follow edges, split into 200k train, 25k validation and 25k test edges.

| Test metric | archived | orig .pyc | this .py |
|---|---:|---:|---:|
| baseline AUC | 0.682718 | 0.682718 | 0.682718 |
| baseline AP | 0.635447 | 0.635447 | 0.635447 |
| baseline Hits@10 | 0.2170 | 0.2170 | 0.2170 |
| baseline MRR | 0.118572 | 0.118572 | 0.118572 |
| tx_enhanced AUC | 0.690402 | 0.690401 | 0.690401 |
| tx_enhanced AP | 0.643693 | 0.643693 | 0.643693 |
| tx_enhanced Hits@10 | 0.2315 | 0.2315 | 0.2315 |
| tx_enhanced MRR | 0.122218 | 0.122218 | 0.122218 |

Largest absolute difference across all test metrics, degree buckets and per-epoch validation history:

- **this .py vs archived**: 2.0e-7.
- **this .py vs today's .pyc run**: 3.0e-4, but only in four `tx_enhanced` degree-bucket values
  (deg0 MRR/NDCG/AUC, deg_gt10 AUC) and one epoch-2 validation MRR. The main test metrics agree
  to 1e-6. Today's `.pyc` run differs from its own archived run by the same amount in the same
  places. This is GPU non-determinism (score ties in the sampled ranking), not a code difference.

We also ran both files with the appendix command alone (default batch/gate/two-hop settings).
They agreed to 2.6e-7: baseline AUC 0.525412, tx_enhanced AUC 0.566043.
