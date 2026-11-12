"""Collect every number used in the report and poster into data/results.json.

All values are read from the raw run artifacts (each run's metrics_report.json
and the reviewer reports), copied from the training cluster into data/raw/.
"""
import glob
import json
import math
import os
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
METRICS = ["auc", "ap", "hits@10", "mrr", "ndcg@10", "logloss", "brier"]


def latest_report(seed_dir):
    runs = sorted(glob.glob(os.path.join(seed_dir, "*", "metrics_report.json")))
    with open(runs[-1]) as f:
        return json.load(f)


def mean_std(values):
    return {"mean": st.mean(values), "std": st.stdev(values) if len(values) > 1 else 0.0}


def gate_prior(d, alpha=1.1, beta=1.2, cutoff=0.25, sharp=10.0, gmin=0.02):
    """Degree prior p(d) exactly as in HybridGNN._degree_prior."""
    s = 1.0 / (1.0 + math.exp(-sharp * (cutoff - d)))
    s = min(max(s, 0.0), 1.0) ** beta
    s = gmin + (1.0 - gmin) * s
    return math.exp(-alpha * (1.0 - s))


def main_experiment():
    seeds = []
    for seed_dir in sorted(glob.glob(os.path.join(RAW, "strict_iter4", "tx200k_seed*"))):
        r = latest_report(seed_dir)
        row = {"seed": r["config"]["seed"]}
        for model in ("baseline", "tx_enhanced"):
            row[model] = {m: r[model]["test"][m] for m in METRICS}
            row[model]["buckets"] = {
                b: {m: v[m] for m in METRICS + ["num_edges"]}
                for b, v in r[model]["test_by_source_degree"].items()
            }
        seeds.append(row)
    cfg = r["config"]

    absolute = {
        model: {m: mean_std([s[model][m] for s in seeds]) for m in METRICS}
        for model in ("baseline", "tx_enhanced")
    }
    buckets_abs = {}
    for b in seeds[0]["baseline"]["buckets"]:
        buckets_abs[b] = {
            model: {m: st.mean(s[model]["buckets"][b][m] for s in seeds) for m in METRICS}
            for model in ("baseline", "tx_enhanced")
        }
        buckets_abs[b]["num_edges"] = st.mean(s["baseline"]["buckets"][b]["num_edges"] for s in seeds)

    rev_dir = os.path.join(RAW, "strict_iter4", "reviewer")
    with open(os.path.join(rev_dir, "strict_reviewer_report.json")) as f:
        rev = json.load(f)
    proto = next(iter(rev["protocol_summaries"].values()))
    paired = {"overall": proto["stats"], "decision": rev["decisions"]}
    for b in ("deg0", "deg_gt10"):
        with open(os.path.join(rev_dir, f"coldstart_{b}_report.json")) as f:
            paired[b] = next(iter(json.load(f)["coldstart_summaries"].values()))["stats"]

    return {
        "config": cfg,
        "graph_stats": r["graph_stats"],
        "tx_summary": r["tx_summary"],
        "elapsed_seconds_last_seed": r["elapsed_seconds"],
        "per_seed": seeds,
        "absolute": absolute,
        "buckets_absolute": buckets_abs,
        "paired": paired,
    }


def strong_baseline():
    out = {"per_seed": []}
    models = ("baseline", "tx_enhanced", "lightgcn", "lightgcn_tx")
    for seed_dir in sorted(glob.glob(os.path.join(RAW, "strict_lightgcn", "tx200k_seed*"))):
        runs = glob.glob(os.path.join(seed_dir, "*", "metrics_report.json"))
        if not runs:
            continue
        r = latest_report(seed_dir)
        if not all(m in r for m in models):
            continue
        out["per_seed"].append({"seed": r["config"]["seed"], **{m: {k: r[m]["test"][k] for k in METRICS} for m in models}})
        out["config"] = {k: r["config"].get(k) for k in ("hard_negative_ratio", "two_hop_mid_cap", "two_hop_pool_cap", "epochs", "hidden_dim")}
    out["absolute"] = {m: {k: mean_std([s[m][k] for s in out["per_seed"]]) for k in METRICS} for m in models}
    out["n_seeds"] = len(out["per_seed"])
    return out


def single_runs(name):
    res = {}
    for path in sorted(glob.glob(os.path.join(RAW, name, "*", "*", "metrics_report.json"))):
        variant = path.split(os.sep)[-3]
        with open(path) as f:
            r = json.load(f)
        res[variant] = {
            "config": {k: r["config"].get(k) for k in ("enhanced_use_tx_features", "enhanced_use_tx_graph", "tx_graph_weight", "negative_mode", "eval_negative_mode", "split_mode", "max_tx_rows", "seed", "epochs", "residual_tx_decode", "degree_aware_gate", "tx_feature_gate")},
            "graph_stats": r["graph_stats"],
            # Early runs (tx_ablation) logged fewer metrics than the strict protocol.
            "baseline": {m: v for m, v in r["baseline"]["test"].items() if m in METRICS or m == "hits_at_k"},
            "tx_enhanced": {m: v for m, v in r["tx_enhanced"]["test"].items() if m in METRICS or m == "hits_at_k"},
            "buckets": {
                model: {b: {m: v[m] for m in ("auc", "hits@10", "num_edges") if m in v} for b, v in r[model].get("test_by_source_degree", {}).items()}
                for model in ("baseline", "tx_enhanced")
            },
        }
    return res


if __name__ == "__main__":
    results = {
        "main": main_experiment(),
        "strong_baseline": strong_baseline(),
        "ablation": single_runs("tx_ablation"),
        "full_scale": single_runs("tx_full_scale"),
        "gate_prior": {f"{d:.2f}": gate_prior(d) for d in [i / 20 for i in range(21)]},
    }
    out = os.path.join(ROOT, "data", "results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    m = results["main"]
    print("main abs AUC", {k: round(v["auc"]["mean"], 4) for k, v in m["absolute"].items()})
    print("paired overall AUC", {k: round(v, 4) for k, v in m["paired"]["overall"]["auc"].items()})
    print("strong baseline", results["strong_baseline"]["n_seeds"], {k: round(v["auc"]["mean"], 4) for k, v in results["strong_baseline"]["absolute"].items()})
    print("ablation", {k: round(v["tx_enhanced"]["auc"] - v["baseline"]["auc"], 4) for k, v in results["ablation"].items()}, {k: v["config"]["eval_negative_mode"] for k, v in results["ablation"].items()})
    print("full scale", {k: v["config"] for k, v in results["full_scale"].items()})
    print("gate", {k: round(v, 3) for k, v in results["gate_prior"].items() if k in ("0.00", "0.25", "0.50", "1.00")})
