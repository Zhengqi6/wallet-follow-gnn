"""Paired significance tests: wallet model vs. baseline over seeds.

Every training run evaluates the baseline and the wallet model on the same
split, negatives and seed, so the natural unit is the per-seed difference
d_s = metric(wallet model) - metric(baseline). For each metric this script
reports, over seeds:

  * mean and standard deviation of d_s,
  * a 95% percentile bootstrap confidence interval for the mean of d_s
    (resampling seeds with replacement),
  * a one-sided sign-flip permutation p-value for H0: mean(d_s) <= 0.
    With n seeds there are 2**n sign patterns; for n <= 16 the test is exact,
    otherwise it is estimated from random sign flips.

It does this for all test edges and for each source-degree bucket
(e.g. deg0 = users who follow no one in the training graph).

Usage:
  python scripts/paired_stats.py data/raw/strict_iter4            # archived runs
  python scripts/paired_stats.py runs/main --out runs/main/stats   # new runs
"""
import argparse
import glob
import itertools
import json
import os

import numpy as np

METRICS = ["auc", "ap", "hits@10", "mrr", "ndcg@10", "precision@10", "recall@10", "logloss", "brier"]
LOWER_IS_BETTER = {"logloss", "brier"}


def load_runs(root):
    """Latest metrics_report.json in each seed directory under root."""
    runs = []
    for seed_dir in sorted(glob.glob(os.path.join(root, "*seed*"))):
        reports = sorted(glob.glob(os.path.join(seed_dir, "*", "metrics_report.json")))
        if reports:
            with open(reports[-1]) as f:
                runs.append(json.load(f))
    if not runs:
        raise SystemExit(f"no */<timestamp>/metrics_report.json found under {root}")
    return sorted(runs, key=lambda r: r["config"]["seed"])


def protocol_key(cfg):
    """Runs are only paired across seeds if everything except the seed matches."""
    return json.dumps({k: v for k, v in sorted(cfg.items()) if k not in ("seed", "out_dir")}, sort_keys=True)


def sign_flip_p(deltas, n_random=5000, rng=None):
    """One-sided p-value for mean(deltas) > 0 under random sign flips."""
    d = np.asarray(deltas, dtype=float)
    observed = d.mean()
    if len(d) <= 16:
        signs = np.array(list(itertools.product([1.0, -1.0], repeat=len(d))))
        null = (signs * d).mean(axis=1)
        return float((null >= observed - 1e-15).mean()), "exact"
    rng = rng or np.random.default_rng(0)
    null = (rng.choice([1.0, -1.0], size=(n_random, len(d))) * d).mean(axis=1)
    return float(((null >= observed - 1e-15).sum() + 1) / (n_random + 1)), f"{n_random} random flips"


def bootstrap_ci(deltas, n_boot=3000, rng=None):
    d = np.asarray(deltas, dtype=float)
    rng = rng or np.random.default_rng(0)
    means = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_table(base_vals, wallet_vals, rng):
    rows = {}
    for m in METRICS:
        if m not in base_vals[0] or m not in wallet_vals[0]:
            continue
        b = np.array([v[m] for v in base_vals], dtype=float)
        w = np.array([v[m] for v in wallet_vals], dtype=float)
        # Orient so that a positive delta always means "wallet model is better".
        d = (b - w) if m in LOWER_IS_BETTER else (w - b)
        lo, hi = bootstrap_ci(d, rng=rng)
        p, how = sign_flip_p(d, rng=rng)
        rows[m] = {
            "n": int(len(d)),
            "baseline_mean": float(b.mean()),
            "wallet_mean": float(w.mean()),
            "mean_delta": float(d.mean()),
            "std_delta": float(d.std()),  # population SD, as in the archived reports
            "seeds_improved": int((d > 0).sum()),
            "ci95_low": lo,
            "ci95_high": hi,
            "p_one_sided": p,
            "p_method": how,
        }
    return rows


def analyse(runs, seed=0):
    rng = np.random.default_rng(seed)
    groups = {}
    for r in runs:
        groups.setdefault(protocol_key(r["config"]), []).append(r)
    out = []
    for key, rs in groups.items():
        entry = {
            "seeds": [r["config"]["seed"] for r in rs],
            "overall": paired_table([r["baseline"]["test"] for r in rs],
                                    [r["tx_enhanced"]["test"] for r in rs], rng),
            "by_source_degree": {},
        }
        buckets = rs[0]["baseline"].get("test_by_source_degree", {})
        for b in buckets:
            entry["by_source_degree"][b] = paired_table(
                [r["baseline"]["test_by_source_degree"][b] for r in rs],
                [r["tx_enhanced"]["test_by_source_degree"][b] for r in rs], rng)
        entry["config"] = json.loads(key)
        out.append(entry)
    return out


def to_markdown(results):
    lines = ["# Paired comparison: wallet model vs. baseline", ""]
    for g in results:
        lines.append(f"Seeds: {g['seeds']}  (positive delta = wallet model better; "
                     "logloss and Brier are sign-flipped)")
        for name, table in [("all test edges", g["overall"])] + [
                (f"source degree {b}", t) for b, t in g["by_source_degree"].items()]:
            lines += ["", f"## {name}", "",
                      "| metric | baseline | wallet | delta | 95% CI | improved | p (one-sided) |",
                      "|---|---:|---:|---:|---|---:|---:|"]
            for m, s in table.items():
                lines.append(f"| {m} | {s['baseline_mean']:.4f} | {s['wallet_mean']:.4f} | "
                             f"{s['mean_delta']:+.4f} | [{s['ci95_low']:+.4f}, {s['ci95_high']:+.4f}] | "
                             f"{s['seeds_improved']}/{s['n']} | {s['p_one_sided']:.4f} |")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs_root", help="directory holding <name>_seed<S>/<timestamp>/metrics_report.json")
    ap.add_argument("--out", help="write paired_stats.json and paired_stats.md here")
    ap.add_argument("--rng-seed", type=int, default=0)
    args = ap.parse_args()

    results = analyse(load_runs(args.runs_root), seed=args.rng_seed)
    md = to_markdown(results)
    print(md)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "paired_stats.json"), "w") as f:
            json.dump(results, f, indent=1)
        with open(os.path.join(args.out, "paired_stats.md"), "w") as f:
            f.write(md)


if __name__ == "__main__":
    main()
