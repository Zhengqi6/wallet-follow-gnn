# Does Your Wallet Know Whom You'll Follow?

Code and results for an AIS5281 project (NUS, AI for Science): who-to-follow
recommendation on Farcaster, a blockchain-based social network, using each
user's public crypto-wallet activity as a cold-start signal.

A two-layer graph neural network learns social profiles from the follow graph
(the baseline). We add a wallet branch: 20 behavioural wallet features, a learned
gate that mutes empty wallets, and a degree-aware trust weight, so the wallet
signal counts most for users with no follows. The two branches meet only in the
link score:

```
score(u, v) = z_u · z_v + 0.15 · p(d_u) · t_u · t_v
```

## Main results

Time-based split (train on the oldest 80% of follows, test on the newest 10%),
each test follow ranked against 99 decoys (30% friends of friends), 8 paired seeds.

| Metric | Baseline | + Wallet (ours) | Δ [95% CI] | p |
|---|---:|---:|---:|---:|
| AUC | 0.6851 | **0.6896** | +0.0045 [0.0011, 0.0076] | 0.027 |
| AP | 0.6369 | **0.6427** | +0.0058 [0.0014, 0.0100] | 0.031 |
| Hits@10 | 0.2132 | **0.2214** | +0.0083 [0.0012, 0.0152] | 0.045 |
| MRR | 0.1126 | **0.1157** | +0.0031 [0.0011, 0.0051] | 0.020 |

For users with no follows, Hits@10 rises from 61.3% to 63.7% (p = 0.003).
Under a harder test (70% friend-of-friend decoys, 5 seeds) LightGCN is the
stronger encoder (AUC 0.767 vs 0.617 for ours), and adding wallet features to
it gives the best Hits@10 (0.259 → 0.359).

## Layout

| Path | Content |
|---|---|
| `artifact/train_tx_enhanced_gnn.py` | Training and evaluation script (baseline and wallet model in one run) |
| `artifact/README.md` | Environment, data format, the exact command, and the equivalence check |
| `artifact/requirements.txt` | Pinned versions (Python 3.9, PyTorch 2.8.0+cu128, NumPy 1.26.4, scikit-learn 1.0.2) |
| `scripts/collect_results.py` | Reads every run in `data/raw/` and writes `data/results.json` |
| `scripts/make_figures.py` | Result figures from `data/results.json` |
| `scripts/extract_graph_sample.py` | Rebuilds the sample graph from the raw Farcaster and transaction files |
| `scripts/make_real_figures.py` | Figures of the real sample graph (needs the output of `extract_graph_sample.py`) |
| `data/raw/` | Per-run `metrics_report.json` and reviewer reports (bootstrap CIs, permutation tests) |
| `data/results.json` | Every number reported in the thesis and poster |

## Reproduce

Training (one seed; the thesis uses seeds 92–99), on a machine with the raw
data. All flags in `artifact/README.md` are required to match the reported runs:

```bash
pip install -r artifact/requirements.txt
python artifact/train_tx_enhanced_gnn.py --seed 92 --out-dir runs/tx200k_seed92 ...
```

Numbers and figures from the archived runs (no GPU or raw data needed):

```bash
python scripts/collect_results.py
python scripts/make_figures.py
```

## Data

The raw data (about 1.1 TB of public Farcaster protocol messages and wallet
transaction records) is not included. The sample graph with user IDs is also
left out; `scripts/extract_graph_sample.py` rebuilds it from the raw files.

## Notes

- The original training source was lost; `artifact/train_tx_enhanced_gnn.py`
  was rebuilt from its compiled bytecode. All 48 code objects match the original
  instruction for instruction, and seed 92 reproduces the archived metrics.
- The LightGCN comparison runs used a later script version that was not
  preserved; they are documented by their results in `data/raw/strict_lightgcn/`.
