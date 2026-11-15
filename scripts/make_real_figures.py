"""Figures drawn from the real Farcaster sample (data/sample/graph_sample.json,
produced on i1 by extract_graph_sample.py). No account IDs are shown."""
import collections
import json
import os
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
with open(os.path.join(ROOT, "data", "sample", "graph_sample.json")) as f:
    D = json.load(f)
TX = {int(k): v for k, v in D["tx_count"].items()}
EDGES = [(u, v) for u, v, _ in D["edges"]]

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2, "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False, "pdf.fonttype": 42,
})


def save(fig, name):
    for ext, kw in (("pdf", {}), ("png", {"dpi": 220})):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), bbox_inches="tight", pad_inches=0.03, **kw)
    plt.close(fig)


def fig_real_network(n_sources=12, seed=7, poster=False):
    """A slice of the real follow graph: a few users and everyone they follow.

    poster=True draws it at its printed poster size (about 160 mm wide) with
    poster-sized type, instead of the report size."""
    out = collections.defaultdict(list)
    for u, v in EDGES:
        out[u].append(v)
    rng = random.Random(seed)
    # Mid-sized followers keep the picture readable; prefer a mix of wallet-active ones.
    pool = [u for u, vs in out.items() if 15 <= len(vs) <= 45]
    pool.sort()
    sources = rng.sample(pool, n_sources)
    g = nx.Graph()
    for u in sources:
        for v in out[u]:
            g.add_edge(u, v)
    # Accounts followed by several sampled users join the clusters together.
    pos = nx.spring_layout(g, seed=seed, k=0.12, iterations=300)
    active = [n for n in g.nodes if TX.get(n, 0) > 0]
    idle = [n for n in g.nodes if TX.get(n, 0) == 0]
    deg = dict(g.degree)

    k = 1.9 if poster else 1.0          # area scale for markers at poster size
    fig, ax = plt.subplots(figsize=(6.3, 5.9) if poster else (4.6, 4.0))
    size = {n: k * (70 if n in sources else 10 + 6 * min(deg[n], 6)) for n in g.nodes}
    nx.draw_networkx_edges(g, pos, ax=ax, edge_color=ORANGE, alpha=0.35, width=1.0 if poster else 0.6)
    nx.draw_networkx_nodes(g, pos, nodelist=idle, ax=ax, node_color="#c9c8c2",
                           node_size=[size[n] for n in idle], edgecolors="white", linewidths=0.4)
    nx.draw_networkx_nodes(g, pos, nodelist=active, ax=ax, node_color=BLUE,
                           node_size=[size[n] for n in active], edgecolors="white", linewidths=0.6)
    ax.set_axis_off()
    share = len(active) / g.number_of_nodes()
    ax.set_title(f"{g.number_of_nodes()} real users, {g.number_of_edges()} follows:\nmost wallets are silent" if poster
                 else f"{g.number_of_nodes()} real users, {g.number_of_edges()} follows", loc="left",
                 fontsize=24 if poster else None)
    from matplotlib.lines import Line2D
    ms = 13 if poster else 6
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", color=BLUE, markersize=ms, label=f"has wallet payments ({share:.0%})"),
        Line2D([], [], marker="o", ls="", color="#c9c8c2", markersize=ms, label="no wallet payments"),
        Line2D([], [], color=ORANGE, alpha=0.6, lw=2.4 if poster else 1.2, label="follow"),
    ], loc="lower left", bbox_to_anchor=(0, -0.2 if poster else -0.08), ncol=2 if poster else 3,
       fontsize=21 if poster else 7, handletextpad=0.3, columnspacing=1.0)
    save(fig, "fig_real_network_poster" if poster else "fig_real_network")
    return g.number_of_nodes(), g.number_of_edges(), share


def fig_tx_hist():
    """How many wallet payments each user in the sample has."""
    bins = [("0", 0, 0), ("1", 1, 1), ("2–3", 2, 3), ("4–10", 4, 10), ("11–100", 11, 100), (">100", 101, 10**9)]
    counts = [sum(lo <= c <= hi for c in TX.values()) for _, lo, hi in bins]
    fig, ax = plt.subplots(figsize=(3.6, 2.0))
    x = range(len(bins))
    colors = ["#c9c8c2"] + [BLUE] * (len(bins) - 1)
    ax.bar(x, counts, width=0.62, color=colors, edgecolor="white", linewidth=1.5)
    for xi, c in zip(x, counts):
        ax.text(xi, c + max(counts) * 0.02, f"{c:,}", ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.set_xticks(list(x), [b[0] for b in bins])
    ax.set_xlabel("wallet payments per user")
    ax.set_ylabel("users")
    ax.set_ylim(0, max(counts) * 1.15)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_title(f"{len(TX):,} real users: most wallets are quiet")
    save(fig, "fig_tx_hist")
    return dict(zip([b[0] for b in bins], counts))


if __name__ == "__main__":
    print("network", fig_real_network())
    with plt.rc_context({"font.family": ["Helvetica", "Arial", "DejaVu Sans"], "axes.titleweight": "bold"}):
        print("network (poster)", fig_real_network(poster=True))
    print("hist", fig_tx_hist())
