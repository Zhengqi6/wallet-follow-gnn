"""Render all report/poster figures (vector PDF + PNG preview) from data/results.json."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
with open(os.path.join(ROOT, "data", "results.json")) as f:
    R = json.load(f)

# Reference categorical palette (light mode), fixed order.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2,
    "axes.titlecolor": INK,
    "axes.titlesize": 9.5,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "pdf.fonttype": 42,
})


def save(fig, name):
    fig.savefig(os.path.join(FIG, f"{name}.pdf"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(os.path.join(FIG, f"{name}.png"), dpi=220, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def fig_seed_deltas():
    main = R["main"]
    seeds = [s["seed"] for s in main["per_seed"]]
    d = [s["tx_enhanced"]["auc"] - s["baseline"]["auc"] for s in main["per_seed"]]
    st = main["paired"]["overall"]["auc"]
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    x = np.arange(len(seeds))
    ax.axhspan(st["ci95_low"], st["ci95_high"], color=BLUE, alpha=0.12, lw=0)
    ax.axhline(st["mean_delta"], color=BLUE, lw=1.2)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.vlines(x, 0, d, color=BLUE, lw=2, capstyle="round")
    ax.scatter(x, d, s=26, color=BLUE, edgecolor="white", linewidth=1.2, zorder=3)
    ax.set_xticks(x, [str(s) for s in seeds])
    ax.set_xlabel("Seed")
    ax.set_ylabel("ΔAUC (Tx − baseline)")
    ax.set_ylim(-0.0068, 0.0122)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.text(-0.45, 0.0112, f"mean +{st['mean_delta']:.4f}, 95% CI [{st['ci95_low']:.4f}, {st['ci95_high']:.4f}], p = {st['perm_p_greater_0']:.3f}",
            va="top", ha="left", fontsize=7, color=INK2)
    ax.set_xlim(-0.6, len(seeds) - 0.4)
    ax.set_title("Paired per-seed AUC gain (8 seeds)")
    save(fig, "fig_seed_deltas")


def fig_buckets():
    p = R["main"]["paired"]
    metrics = [("auc", "AUC"), ("ap", "AP"), ("hits@10", "Hits@10"), ("mrr", "MRR"), ("ndcg@10", "NDCG@10")]
    groups = [("overall", "All sources", BLUE), ("deg0", "Cold-start (0 follows)", ORANGE), ("deg_gt10", "High-degree (>10)", AQUA)]
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    y = np.arange(len(metrics))[::-1]
    off = [-0.24, 0.0, 0.24]
    for (key, label, color), o in zip(groups, off):
        mean = [p[key][m]["mean_delta"] for m, _ in metrics]
        lo = [p[key][m]["ci95_low"] for m, _ in metrics]
        hi = [p[key][m]["ci95_high"] for m, _ in metrics]
        ax.hlines(y + o, lo, hi, color=color, lw=2, capstyle="round")
        ax.scatter(mean, y + o, s=22, color=color, edgecolor="white", linewidth=1.0, zorder=3, label=label)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(y, [lbl for _, lbl in metrics])
    ax.set_xlabel("Gain over baseline (mean, 95% CI)")
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=1, fontsize=7, handletextpad=0.2, borderaxespad=0.1)
    ax.set_title("Gains by source-user degree", pad=44)
    save(fig, "fig_buckets")


def fig_gate():
    d = np.linspace(0, 1, 201)

    def prior(x, alpha=1.1, beta=1.2, cutoff=0.25, sharp=10.0, gmin=0.02):
        s = np.clip(1 / (1 + np.exp(-sharp * (cutoff - x))), 0, 1) ** beta
        return np.exp(-alpha * (1 - (gmin + (1 - gmin) * s)))

    fig, ax = plt.subplots(figsize=(3.0, 1.9))
    ax.plot(d, prior(d), color=BLUE, lw=2)
    for x0 in (0.0, 1.0):
        ax.scatter([x0], [prior(x0)], s=28, color=BLUE, edgecolor="white", linewidth=1.2, zorder=3, clip_on=False)
    ax.annotate(f"cold-start: {prior(0.0):.2f}", (0.0, prior(0.0)), xytext=(10, -2), textcoords="offset points", fontsize=7.5, color=INK2, va="center")
    ax.annotate(f"max degree: {prior(1.0):.2f}", (1.0, prior(1.0)), xytext=(-4, 12), textcoords="offset points", fontsize=7.5, color=INK2, ha="right")
    ax.axvline(0.25, color=MUTED, lw=0.8, ls=(0, (3, 2)))
    ax.text(0.27, 0.78, "cutoff c = 0.25", fontsize=7, color=MUTED)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Normalized source degree d")
    ax.set_ylabel("Degree prior p(d)")
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_title("Degree-aware gate")
    save(fig, "fig_gate")


def fig_strong_baseline():
    sb = R["strong_baseline"]["absolute"]
    models = [("baseline", "HybridGNN"), ("tx_enhanced", "HybridGNN + Tx"), ("lightgcn", "LightGCN"), ("lightgcn_tx", "LightGCN + Tx")]
    colors = [ORANGE, BLUE, ORANGE, BLUE]
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 1.9), sharey=True)
    y = np.arange(len(models))[::-1]
    for ax, (metric, title) in zip(axes, [("auc", "Test AUC"), ("hits@10", "Test Hits@10")]):
        vals = [sb[m][metric]["mean"] for m, _ in models]
        errs = [sb[m][metric]["std"] for m, _ in models]
        ax.barh(y, vals, height=0.55, color=colors, edgecolor="white", linewidth=1.5)
        ax.errorbar(vals, y, xerr=errs, fmt="none", ecolor=INK2, elinewidth=0.8, capsize=2)
        for yy, v, e in zip(y, vals, errs):
            ax.text(v + e + max(vals) * 0.04, yy, f"{v:.3f}", va="center", fontsize=7.5, color=INK)
        ax.set_xlim(0, max(vals) * 1.25)
        ax.xaxis.grid(True)
        ax.set_axisbelow(True)
        ax.set_title(title)
    axes[0].set_yticks(y, [lbl for _, lbl in models])
    if "auc" in sb["baseline"]:
        axes[0].axvline(0.5, color=MUTED, lw=0.8, ls=(0, (3, 2)))
    fig.suptitle(f"Harder protocol, {R['strong_baseline']['n_seeds']} seeds (70% two-hop negatives)", x=0.01, ha="left", fontsize=9.5, fontweight="bold", color=INK, y=1.04)
    for ax, t in zip(axes, ("Test AUC", "Test Hits@10")):
        ax.set_title(t, fontweight="normal", color=INK2, fontsize=8.5)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=ORANGE, label="without tx"), Patch(color=BLUE, label="with tx")],
               loc="upper right", bbox_to_anchor=(1.0, 1.09), ncol=2, fontsize=7.5, handlelength=1.0)
    save(fig, "fig_strong_baseline")


def fig_ablation():
    ab = R["ablation"]
    variants = [("feat_only", "Tx features only"), ("graph_only", "Tx-similarity graph only"), ("both", "Features + graph")]
    vals = [ab[k]["tx_enhanced"]["auc"] - ab[k]["baseline"]["auc"] for k, _ in variants]
    fig, ax = plt.subplots(figsize=(3.2, 1.5))
    y = np.arange(len(variants))[::-1]
    ax.barh(y, vals, height=0.5, color=BLUE, edgecolor="white", linewidth=1.5)
    for yy, v in zip(y, vals):
        ax.text(v + (0.002 if v >= 0 else -0.002), yy, f"{v:+.4f}", va="center", ha="left" if v >= 0 else "right", fontsize=7.5, color=INK)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(y, [lbl for _, lbl in variants])
    ax.set_xlim(-0.04, 0.075)
    ax.set_xlabel("ΔAUC vs. same model without tx (1 seed)")
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_title("Where to inject transactions (early pipeline)")
    save(fig, "fig_ablation")


def fig_convergence():
    """Training loss and validation AUC per epoch, 8 seeds (strict_iter4 histories)."""
    import glob
    runs = sorted(glob.glob(os.path.join(ROOT, "data", "raw", "strict_iter4", "tx200k_seed*", "*", "metrics_report.json")))
    hist = {"baseline": [], "tx_enhanced": []}
    for path in runs:
        with open(path) as f:
            r = json.load(f)
        for m in hist:
            hist[m].append(r[m]["history"])
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.3))
    for ax, key, label in ((axes[0], "loss", "Training loss (BCE)"), (axes[1], "auc", "Validation AUC")):
        for m, color, name in (("baseline", ORANGE, "Baseline"), ("tx_enhanced", BLUE, "+ Tx (ours)")):
            arr = np.array([[h[key] for h in run] for run in hist[m]])
            ep = np.arange(1, arr.shape[1] + 1)
            ax.fill_between(ep, arr.min(0), arr.max(0), color=color, alpha=0.15, lw=0)
            ax.plot(ep, arr.mean(0), color=color, lw=2, marker="o", markersize=4, label=name)
        ax.set_xlabel("Epoch")
        ax.set_title(label)
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
        ax.set_xticks(ep)
    axes[1].legend(loc="lower right", fontsize=7.5)
    fig.tight_layout(w_pad=2)
    save(fig, "fig_convergence")
    best = {m: [int(np.argmax([h["auc"] for h in run])) + 1 for run in hist[m]] for m in hist}
    return {m: {"best_epochs": best[m],
                "final_val_auc_mean": float(np.mean([run[-1]["auc"] for run in hist[m]])),
                "final_loss_mean": float(np.mean([run[-1]["loss"] for run in hist[m]]))} for m in hist}


def fig_hits_poster():
    """Poster-size result chart with plain-language axes: share of real follows
    in the top 10, by user group, without vs with wallets (8 seeds, mean +- std)."""
    seeds = R["main"]["per_seed"]
    groups = [("deg0", "New users\n(0 follows)"), ("deg_gt10", "Users following\nmore than 10")]
    stats = {}
    for b, _ in groups:
        for m in ("baseline", "tx_enhanced"):
            v = [100 * s[m]["buckets"][b]["hits@10"] for s in seeds]
            stats[b, m] = (float(np.mean(v)), float(np.std(v, ddof=1)))
    with plt.rc_context({"font.family": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 22,
                         "axes.titlesize": 24, "axes.linewidth": 1.4,
                         "xtick.major.width": 1.4, "ytick.major.width": 1.4}):
        fig, ax = plt.subplots(figsize=(8.2, 3.9))
        x = np.arange(len(groups))
        w = 0.36
        for k, (m, color, name) in enumerate((("baseline", ORANGE, "without wallets"), ("tx_enhanced", BLUE, "with wallets"))):
            means = [stats[b, m][0] for b, _ in groups]
            sds = [stats[b, m][1] for b, _ in groups]
            xs = x + (k - 0.5) * w
            ax.bar(xs, means, width=w * 0.94, color=color, edgecolor="white", linewidth=3, label=name)
            ax.errorbar(xs, means, yerr=sds, fmt="none", ecolor=INK2, elinewidth=1.6, capsize=5)
            for xi, mval in zip(xs, means):
                # round half up so 19.25 prints as 19.3, matching the report tables
                ax.text(xi, mval / 2, f"{np.floor(mval * 10 + 0.5) / 10:.1f}%", ha="center", va="center", color="white", fontsize=21, fontweight="bold")
        for xi, (b, _) in zip(x, groups):
            gain = stats[b, "tx_enhanced"][0] - stats[b, "baseline"][0]
            top = max(stats[b, m][0] + stats[b, m][1] for m in ("baseline", "tx_enhanced"))
            ax.text(xi, top + 3, f"+{np.floor(gain * 10 + 0.5) / 10:.1f} pts", ha="center", va="bottom", fontsize=23, fontweight="bold", color=INK)
        ax.set_xticks(x, [g for _, g in groups])
        ax.set_ylabel("Hit rate: real follow\nin our top 10 (%)", fontsize=20)
        ax.set_ylim(0, 80)
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
        ax.legend(loc="upper right", fontsize=20, handlelength=1.2)
        save(fig, "fig_hits_poster")
    return stats


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    print(fig_hits_poster())
    fig_seed_deltas()
    fig_buckets()
    fig_gate()
    fig_strong_baseline()
    fig_ablation()
    print(fig_convergence())
    print(sorted(os.listdir(FIG)))
