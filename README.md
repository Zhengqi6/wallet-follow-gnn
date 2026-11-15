# Decentralized Social Network Recommendation

*Does your wallet know whom you'll follow?*

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
| `artifact/train_tx_enhanced_gnn.py` | Training and evaluation: baseline and wallet model, same split and seed, in one run |
| `artifact/README.md` | Environment, the exact command, and the equivalence check of the rebuilt source |
| `artifact/requirements.txt` | Pinned versions (Python 3.9, PyTorch 2.8.0+cu128, NumPy 1.26.4, scikit-learn 1.0.2, SciPy 1.13.1, Matplotlib 3.5.2) |
| `configs/main_experiment.args` | All flags of the main experiment (checked against the archived run configs) |
| `configs/toy.args` | The same model and protocol, scaled down for the toy data |
| `scripts/run_seeds.sh` | Runs the main experiment for a range of seeds |
| `scripts/slurm_run_seeds.sbatch` | Slurm array job: seeds 92–99, one A100 each |
| `scripts/paired_stats.py` | Paired bootstrap CIs and exact sign-flip tests, overall and per source-degree bucket |
| `scripts/collect_results.py` | Reads every run in `data/raw/` and writes `data/results.json` |
| `scripts/make_figures.py` | Result figures from `data/results.json` |
| `scripts/make_toy_data.py` | Small synthetic dataset in the exact input format |
| `scripts/extract_graph_sample.py` | Rebuilds the sample graph from the raw Farcaster and transaction files |
| `scripts/make_real_figures.py` | Figures of the real sample graph (needs the output of `extract_graph_sample.py`) |
| `tests/smoke_test.sh` | End-to-end check on toy data: train 2 seeds, run the paired statistics |
| `docs/DATA.md` | Input file format, sizes, and how to get the data |
| `data/raw/` | Per-run `metrics_report.json` and the archived reviewer reports |
| `data/results.json` | Every number reported in the thesis and poster |

## Reproduce

Hardware used for the thesis: one NVIDIA A100 80GB (Slurm cluster). One seed
trains both models in about 2–3 minutes.

**1. Environment**

```bash
python3.9 -m venv .venv && source .venv/bin/activate
pip install -r artifact/requirements.txt
```

**2. Check the pipeline on toy data** (about a minute):

```bash
PYTHON=python tests/smoke_test.sh
```

**3. Main experiment** with the raw data (see `docs/DATA.md`), seeds 92–99:

```bash
LINKS=/data/links.jsonl TX=/data/fid_main_addr_all_transactions.jsonl \
  scripts/run_seeds.sh 92 99 runs/main
```

or as a Slurm array job, one A100 per seed:

```bash
sbatch --export=ALL,LINKS=/data/links.jsonl,TX=/data/fid_main_addr_all_transactions.jsonl \
  scripts/slurm_run_seeds.sbatch
```

**4. Statistics** for new runs, or for the archived ones:

```bash
python scripts/paired_stats.py runs/main --out runs/main/stats
python scripts/paired_stats.py data/raw/strict_iter4
```

On the archived runs this reproduces the mean and standard deviation of every
paired difference in the reviewer reports exactly (27 metric-bucket pairs); the
p-values come from the exact sign-flip test (2^8 sign patterns) instead of a
random sample of flips.

**5. Numbers and figures** of the thesis, from the archived runs (no GPU or raw
data needed):

```bash
python scripts/collect_results.py
python scripts/make_figures.py
```

## Data

The raw data (about 1.1 TB of public Farcaster messages and wallet
transactions) is not included; it is available from the author on request.
`docs/DATA.md` gives the exact input format. The sample graph with account IDs
is also left out; `scripts/extract_graph_sample.py` rebuilds it.

## Notes

- The original training source was lost; `artifact/train_tx_enhanced_gnn.py`
  was rebuilt from its compiled bytecode. All 48 code objects match the original
  instruction for instruction, and seed 92 reproduces the archived metrics.
- The LightGCN comparison runs used a later script version that was not
  preserved; they are documented by their results in `data/raw/strict_lightgcn/`.
