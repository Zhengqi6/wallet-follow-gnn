#!/usr/bin/env python3
"""
Train follow recommendation GNN with and without transaction signals.

Pipeline:
1) Stream transaction JSONL and build per-fid transaction features.
2) Stream follow JSONL and build a follow subgraph over fids that have tx data.
3) Build transaction-similarity graph via shared counterparties.
4) Train two models:
   - baseline: follow-only message passing
   - tx-enhanced: follow + tx-sim message passing + tx node features
5) Compare metrics (AUC/AP/Hits@10) and export artifacts.
"""

import argparse
import json
import math
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class TxUserStats:
    tx_events: int = 0
    in_txs: int = 0
    out_txs: int = 0
    self_txs: int = 0
    unknown_dir_txs: int = 0
    total_value_wei: int = 0
    confirmed_txs: int = 0
    unique_counterparties: Set[str] = None
    unique_hashes: Set[str] = None
    address_count: int = 0
    declared_total_count: int = 0
    recent3_txs: int = 0
    recent10_txs: int = 0
    recent3_value_wei: int = 0
    recent10_value_wei: int = 0
    recent10_counterparties: Set[str] = None

    def __post_init__(self) -> None:
        if self.unique_counterparties is None:
            self.unique_counterparties = set()
        if self.unique_hashes is None:
            self.unique_hashes = set()
        if self.recent10_counterparties is None:
            self.recent10_counterparties = set()


def safe_int(value, default=0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def stream_tx_features(
    tx_path: str,
    max_tx_rows: int,
    counterparty_fid_cap: int,
    min_shared_counterparty: int,
    max_fids_per_counterparty_for_edges: int,
) -> Tuple[Dict[int, TxUserStats], Dict[Tuple[int, int], int], Dict[str, int]]:
    """
    Parse tx records and build:
    - per-fid transaction stats
    - tx similarity edges (fid-fid weighted by shared counterparties)
    - analysis summary
    """
    fid_stats = {}
    counterparty_to_fids = {}
    overflow_counterparties = set()

    summary = {
        "tx_rows_seen": 0,
        "tx_rows_valid": 0,
        "tx_events_seen": 0,
        "tx_events_with_eth": 0,
        "tx_rows_with_rpc_error": 0,
    }

    with open(tx_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            if max_tx_rows > 0 and line_idx > max_tx_rows:
                break
            summary["tx_rows_seen"] += 1
            if line_idx % 2000 == 0:
                print(
                    f"[tx-parse] rows={line_idx} valid={summary['tx_rows_valid']} events="
                    f"{summary['tx_events_seen']}",
                    flush=True,
                )

            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            fid = obj.get("fid")
            addr = str(obj.get("address", "")).lower()
            if fid is None or not addr:
                continue
            fid = safe_int(fid, -1)
            if fid < 0:
                continue

            summary["tx_rows_valid"] += 1
            if fid not in fid_stats:
                fid_stats[fid] = TxUserStats()
            user = fid_stats[fid]
            user.address_count += 1

            rpc = obj.get("rpc_result", {}) if isinstance(obj.get("rpc_result", {}), dict) else {}
            if "error" in rpc or rpc.get("rpc_error"):
                summary["tx_rows_with_rpc_error"] += 1

            result = rpc.get("result", {}) if isinstance(rpc.get("result", {}), dict) else {}
            user.declared_total_count += safe_int(result.get("total_count", 0), 0)
            txs = result.get("addressTransactions", [])
            if not isinstance(txs, list):
                continue

            for tx_idx, tx in enumerate(txs):
                summary["tx_events_seen"] += 1
                user.tx_events += 1

                if not isinstance(tx, dict):
                    continue
                tx_hash = tx.get("hash")
                if isinstance(tx_hash, str) and tx_hash:
                    user.unique_hashes.add(tx_hash)

                status = str(tx.get("status", ""))
                if status.upper() == "CONFIRMED":
                    user.confirmed_txs += 1

                eth = tx.get("ethereum", {})
                if not isinstance(eth, dict):
                    continue
                summary["tx_events_with_eth"] += 1

                frm = str(eth.get("from", "")).lower()
                to = str(eth.get("to", "")).lower()
                value = safe_int(eth.get("value", 0), 0)
                user.total_value_wei += max(value, 0)
                if tx_idx < 3:
                    user.recent3_txs += 1
                    user.recent3_value_wei += max(value, 0)
                if tx_idx < 10:
                    user.recent10_txs += 1
                    user.recent10_value_wei += max(value, 0)

                if frm and to and frm == to:
                    user.self_txs += 1

                if frm == addr and to:
                    user.out_txs += 1
                    cp = to
                elif to == addr and frm:
                    user.in_txs += 1
                    cp = frm
                else:
                    user.unknown_dir_txs += 1
                    cp = ""
                    if frm and frm != addr:
                        cp = frm
                    elif to and to != addr:
                        cp = to

                if cp:
                    user.unique_counterparties.add(cp)
                    if tx_idx < 10:
                        user.recent10_counterparties.add(cp)
                    if cp not in overflow_counterparties:
                        bucket = counterparty_to_fids.get(cp)
                        if bucket is None:
                            counterparty_to_fids[cp] = {fid}
                        else:
                            bucket.add(fid)
                            if len(bucket) > counterparty_fid_cap:
                                overflow_counterparties.add(cp)
                                counterparty_to_fids.pop(cp, None)

    tx_pair_weights = defaultdict(int)
    valid_counterparties = 0
    for cp, fids in counterparty_to_fids.items():
        if len(fids) < min_shared_counterparty:
            continue
        if len(fids) > max_fids_per_counterparty_for_edges:
            continue
        valid_counterparties += 1
        sorted_fids = sorted(fids)
        for a, b in combinations(sorted_fids, 2):
            tx_pair_weights[(a, b)] += 1

    summary["counterparty_nodes_kept"] = len(counterparty_to_fids)
    summary["counterparty_nodes_overflow"] = len(overflow_counterparties)
    summary["counterparty_nodes_used_for_edges"] = valid_counterparties
    summary["tx_similarity_pairs"] = len(tx_pair_weights)
    summary["tx_fid_count"] = len(fid_stats)

    return fid_stats, tx_pair_weights, summary


def stream_follow_edges(
    links_path: str,
    allowed_fids: Set[int],
    max_link_rows: int,
    max_follow_edges: int,
) -> Tuple[Dict[Tuple[int, int], int], Dict[str, int]]:
    follow_edges = {}
    summary = {
        "link_rows_seen": 0,
        "follow_add_rows": 0,
        "edges_kept": 0,
    }

    with open(links_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            if max_link_rows > 0 and line_idx > max_link_rows:
                break
            summary["link_rows_seen"] += 1
            if line_idx % 200000 == 0:
                print(
                    f"[follow-parse] rows={line_idx} follow_rows={summary['follow_add_rows']} kept="
                    f"{summary['edges_kept']}",
                    flush=True,
                )

            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            msg = obj.get("message", {})
            data = msg.get("data", {}) if isinstance(msg, dict) else {}
            body = data.get("linkBody", {}) if isinstance(data, dict) else {}
            if not isinstance(body, dict):
                continue

            if body.get("type") != "follow":
                continue
            if data.get("type") != "MESSAGE_TYPE_LINK_ADD":
                continue

            src = safe_int(obj.get("fid", data.get("fid")), -1)
            dst = safe_int(body.get("targetFid"), -1)
            if src < 0 or dst < 0 or src == dst:
                continue
            summary["follow_add_rows"] += 1

            if src in allowed_fids and dst in allowed_fids:
                ts = safe_int(data.get("timestamp"), -1)
                if (src, dst) not in follow_edges:
                    follow_edges[(src, dst)] = ts
                    summary["edges_kept"] += 1
                    if max_follow_edges > 0 and len(follow_edges) >= max_follow_edges:
                        break
                else:
                    prev = follow_edges[(src, dst)]
                    if prev < 0 or (ts >= 0 and ts < prev):
                        follow_edges[(src, dst)] = ts

    return follow_edges, summary


def build_tx_feature_matrix(
    fid_stats: Dict[int, TxUserStats],
    fid_to_idx: Dict[int, int],
) -> np.ndarray:
    x = np.zeros((len(fid_to_idx), 20), dtype=np.float32)
    for fid, idx in fid_to_idx.items():
        s = fid_stats.get(fid, TxUserStats())
        unique_cp = len(s.unique_counterparties)
        recent10_unique_cp = len(s.recent10_counterparties)
        unique_hashes = len(s.unique_hashes)
        log_tx_events = math.log1p(s.tx_events)
        log_value = math.log1p(max(s.total_value_wei, 0))
        ratio_in = s.in_txs / max(1, s.in_txs + s.out_txs)
        ratio_out = s.out_txs / max(1, s.in_txs + s.out_txs)
        confirm_ratio = s.confirmed_txs / max(1, s.tx_events)
        mean_declared = s.declared_total_count / max(1, s.address_count)
        recent10_ratio = s.recent10_txs / max(1, s.tx_events)
        recent3_ratio = s.recent3_txs / max(1, s.tx_events)

        x[idx] = np.array(
            [
                log_tx_events,
                math.log1p(s.in_txs),
                math.log1p(s.out_txs),
                math.log1p(s.self_txs),
                math.log1p(s.unknown_dir_txs),
                math.log1p(unique_cp),
                math.log1p(unique_hashes),
                log_value,
                ratio_in,
                ratio_out,
                confirm_ratio,
                math.log1p(mean_declared),
                math.log1p(s.recent3_txs),
                math.log1p(s.recent10_txs),
                math.log1p(max(s.recent3_value_wei, 0)),
                math.log1p(max(s.recent10_value_wei, 0)),
                math.log1p(recent10_unique_cp),
                recent3_ratio,
                recent10_ratio,
                math.log1p(s.recent10_txs) - log_tx_events,
            ],
            dtype=np.float32,
        )

    mu = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    x = (x - mu) / std
    return x


def split_edges(
    edges: List[Tuple[int, int]],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]]]:
    rng = random.Random(seed)
    rng.shuffle(edges)

    n = len(edges)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    test_edges = edges[:n_test]
    val_edges = edges[n_test : n_test + n_val]
    train_edges = edges[n_test + n_val :]
    return train_edges, val_edges, test_edges


def split_edges_temporal(
    edge_time_items: List[Tuple[Tuple[int, int], int]],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]]]:
    rng = random.Random(seed)
    # Edges with unknown timestamps (< 0) sort after all timestamped edges.
    edge_time_items = sorted(edge_time_items, key=lambda x: (x[1] < 0, x[1]))
    n = len(edge_time_items)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)

    train_items = edge_time_items[: n - n_val - n_test]
    val_items = edge_time_items[n - n_val - n_test : n - n_test]
    test_items = edge_time_items[n - n_test :]

    train_edges = [e for e, _ in train_items]
    val_edges = [e for e, _ in val_items]
    test_edges = [e for e, _ in test_items]

    # Shuffle within each split to avoid batch-order artifacts.
    rng.shuffle(train_edges)
    rng.shuffle(val_edges)
    rng.shuffle(test_edges)
    return train_edges, val_edges, test_edges


def make_sparse_norm_adj(
    edge_list: List[Tuple],
    num_nodes: int,
    undirected: bool = True,
    add_self_loops: bool = True,
    device: str = "cpu",
) -> torch.Tensor:
    """Symmetric-normalised sparse adjacency.

    `edge_list` items are (u, v) (weight 1.0) or (u, v, w). Edge weights only
    enter the (weighted) degree; the stored value of every entry is
    1 / sqrt(deg[row] * deg[col]), exactly as in the original compiled script.
    """
    rows = []
    cols = []
    vals = []
    for item in edge_list:
        if len(item) == 2:
            u, v = item
            w = 1.0
        else:
            u, v, w = item
            w = float(w)
        rows.append(u)
        cols.append(v)
        vals.append(w)
        if undirected:
            rows.append(v)
            cols.append(u)
            vals.append(w)

    if add_self_loops:
        for i in range(num_nodes):
            rows.append(i)
            cols.append(i)
            vals.append(1.0)

    idx = torch.tensor([rows, cols], dtype=torch.long)
    val = torch.tensor(vals, dtype=torch.float32)
    deg = torch.zeros(num_nodes, dtype=torch.float32)
    deg.index_add_(0, idx[0], val)
    deg = torch.clamp(deg, min=1.0)
    norm = 1.0 / torch.sqrt(deg[idx[0]] * deg[idx[1]])

    adj = torch.sparse_coo_tensor(idx, norm, (num_nodes, num_nodes))
    return adj.coalesce().to(device)


def build_tx_knn_edges(
    tx_feats: np.ndarray,
    active_nodes: np.ndarray,
    knn_k: int,
    weight_scale: float,
    max_edges: int,
) -> List[Tuple[int, int, float]]:
    """Undirected kNN graph in tx-feature space, weight = exp(-dist) * scale."""
    if knn_k <= 0:
        return []
    if len(active_nodes) <= 1:
        return []

    n_neighbors = min(knn_k + 1, len(active_nodes))
    nn_model = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean", algorithm="auto", n_jobs=-1)
    nn_model.fit(tx_feats[active_nodes])
    dists, nbrs = nn_model.kneighbors(tx_feats[active_nodes], return_distance=True)

    edge_w = {}
    for i in range(len(active_nodes)):
        u = int(active_nodes[i])
        for j in range(1, n_neighbors):
            v = int(active_nodes[int(nbrs[i, j])])
            if u == v:
                continue
            a, b = (u, v) if u < v else (v, u)
            sim = math.exp(-float(dists[i, j])) * weight_scale
            if sim <= 0:
                continue
            prev = edge_w.get((a, b), 0.0)
            if sim > prev:
                edge_w[(a, b)] = sim

    items = sorted(edge_w.items(), key=lambda x: x[1], reverse=True)
    if max_edges > 0:
        items = items[:max_edges]
    return [(a, b, w) for (a, b), w in items]


class HybridGNN(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        tx_feat_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        use_tx_features: bool,
        use_tx_feature_gate: bool,
        degree_aware_gate: bool,
        degree_gate_alpha: float,
        degree_gate_beta: float,
        degree_gate_cutoff: float,
        degree_gate_sharpness: float,
        degree_gate_min: float,
        residual_tx_decode: bool,
        residual_tx_scale: float,
        source_degree_tx_max: float,
        tx_graph_weight: float,
        residual: float,
    ) -> None:
        super().__init__()
        self.use_tx_features = use_tx_features
        self.use_tx_feature_gate = use_tx_feature_gate
        self.degree_aware_gate = degree_aware_gate
        self.degree_gate_alpha = degree_gate_alpha
        self.degree_gate_beta = degree_gate_beta
        self.degree_gate_cutoff = degree_gate_cutoff
        self.degree_gate_sharpness = degree_gate_sharpness
        self.degree_gate_min = degree_gate_min
        self.residual_tx_decode = residual_tx_decode
        self.residual_tx_scale = residual_tx_scale
        self.source_degree_tx_max = source_degree_tx_max
        self.tx_graph_weight = tx_graph_weight
        self.residual = residual
        self.dropout = dropout

        self.id_embed = nn.Embedding(num_nodes, hidden_dim)
        nn.init.xavier_uniform_(self.id_embed.weight)

        self.tx_proj = nn.Linear(tx_feat_dim, hidden_dim)
        self.tx_gate = nn.Sequential(
            nn.Linear(tx_feat_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])

    def _degree_prior(self, degree_feature: torch.Tensor) -> torch.Tensor:
        # degree_feature = log1p(train out-degree) / max, in [0, 1]; low degree -> prior near 1.
        degree_prior = torch.sigmoid(
            self.degree_gate_sharpness * (self.degree_gate_cutoff - degree_feature)
        )
        degree_prior = torch.pow(torch.clamp(degree_prior, min=0.0, max=1.0), self.degree_gate_beta)
        degree_prior = self.degree_gate_min + (1.0 - self.degree_gate_min) * degree_prior
        degree_prior = torch.exp(-self.degree_gate_alpha * (1.0 - degree_prior))
        return degree_prior

    def build_tx_embedding(
        self,
        tx_features: torch.Tensor,
        degree_feature: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        tx_emb = self.tx_proj(tx_features)
        if self.use_tx_feature_gate:
            gate = self.tx_gate(tx_features)
            if self.degree_aware_gate and degree_feature is not None:
                gate = gate * self._degree_prior(degree_feature)
            tx_emb = tx_emb * gate
        return tx_emb

    def forward(
        self,
        adj_follow: torch.Tensor,
        adj_tx: Optional[torch.Tensor],
        tx_features: torch.Tensor,
        degree_feature: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.id_embed.weight
        if self.use_tx_features and not self.residual_tx_decode:
            h = h + self.build_tx_embedding(tx_features, degree_feature=degree_feature)

        for layer in self.layers:
            agg = torch.sparse.mm(adj_follow, h)
            if adj_tx is not None and self.tx_graph_weight > 0:
                agg_tx = torch.sparse.mm(adj_tx, h)
                agg = (agg + self.tx_graph_weight * agg_tx) / (1.0 + self.tx_graph_weight)
            h = (1 - self.residual) * agg + self.residual * h
            h = layer(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def decode(
        self,
        z: torch.Tensor,
        edges: torch.Tensor,
        tx_emb: Optional[torch.Tensor] = None,
        degree_feature: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        src = edges[:, 0]
        dst = edges[:, 1]
        score = (z[src] * z[dst]).sum(dim=1)
        if self.residual_tx_decode and tx_emb is not None:
            tx_score = (tx_emb[src] * tx_emb[dst]).sum(dim=1)
            if self.degree_aware_gate and degree_feature is not None:
                src_prior = self._degree_prior(degree_feature[src]).reshape(-1)
                tx_score = tx_score * src_prior
            if self.source_degree_tx_max < 1.0 and degree_feature is not None:
                src_mask = (degree_feature[src].reshape(-1) <= self.source_degree_tx_max).float()
                tx_score = tx_score * src_mask
            score = score + self.residual_tx_scale * tx_score
        return score


def score_edges_in_batches(
    model: HybridGNN,
    z: torch.Tensor,
    edges: np.ndarray,
    device: str,
    batch_size: int,
    tx_emb: Optional[torch.Tensor] = None,
    degree_feature: Optional[torch.Tensor] = None,
) -> np.ndarray:
    out = []
    n = len(edges)
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        edge_t = torch.tensor(edges[s:e], dtype=torch.long, device=device)
        scores = torch.sigmoid(
            model.decode(z, edge_t, tx_emb=tx_emb, degree_feature=degree_feature)
        ).detach().cpu().numpy()
        out.append(scores)
    if not out:
        return np.array([], dtype=np.float32)
    return np.concatenate(out, axis=0)


def build_neighbor_map(edges: List[Tuple[int, int]], num_nodes: int) -> List[Set[int]]:
    nbrs = [set() for _ in range(num_nodes)]
    for u, v in edges:
        nbrs[u].add(v)
    return nbrs


def sample_negative_edges(
    num_samples: int,
    num_nodes: int,
    src_candidates: np.ndarray,
    existing_nbrs: List[Set[int]],
    rng: np.random.Generator,
    hard_pools: Optional[List[Optional[np.ndarray]]] = None,
    hard_ratio: float = 0.0,
) -> np.ndarray:
    """Sample (src, dst) non-edges; with prob. hard_ratio dst comes from src's two-hop pool."""
    neg = np.zeros((num_samples, 2), dtype=np.int64)
    i = 0
    while i < num_samples:
        src = int(src_candidates[rng.integers(0, len(src_candidates))])
        use_hard = (
            hard_pools is not None
            and 0 <= src < len(hard_pools)
            and hard_pools[src] is not None
            and len(hard_pools[src]) > 0
            and rng.random() < hard_ratio
        )
        if use_hard:
            pool = hard_pools[src]
            dst = int(pool[rng.integers(0, len(pool))])
        else:
            dst = int(rng.integers(0, num_nodes))
        if dst == src:
            continue
        if dst in existing_nbrs[src]:
            continue
        neg[i, 0] = src
        neg[i, 1] = dst
        i += 1
    return neg


def build_two_hop_hard_pools(
    nbrs: List[Set[int]],
    mid_cap: int,
    pool_cap: int,
) -> List[Optional[np.ndarray]]:
    """For each node: up to pool_cap two-hop out-neighbours that are not direct out-neighbours."""
    pools = [None] * len(nbrs)
    for src in range(len(nbrs)):
        mids = list(nbrs[src])
        if not mids:
            continue
        if mid_cap > 0 and len(mids) > mid_cap:
            mids = mids[:mid_cap]
        cands = set()
        for m in mids:
            for v in nbrs[m]:
                if v == src:
                    continue
                if v in nbrs[src]:
                    continue
                cands.add(v)
                if pool_cap > 0 and len(cands) >= pool_cap:
                    break
            if pool_cap > 0 and len(cands) >= pool_cap:
                break
        if cands:
            pools[src] = np.array(list(cands), dtype=np.int64)
    return pools


def evaluate_link_prediction(
    model: HybridGNN,
    adj_follow: torch.Tensor,
    adj_tx: Optional[torch.Tensor],
    tx_features: torch.Tensor,
    pos_edges: np.ndarray,
    all_nbrs: List[Set[int]],
    device: str,
    seed: int,
    hits_k: int = 10,
    eval_edge_batch_size: int = 200000,
    negative_mode: str = "random",
    hard_pools: Optional[List[Optional[np.ndarray]]] = None,
    hard_ratio: float = 0.0,
    rank_negatives: int = 99,
    degree_feature: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Classification metrics (AUC/AP/logloss/Brier) on pos edges vs. an equal number of
    sampled negatives, plus sampled ranking metrics: each of the first <=2000 positives
    is ranked against `rank_negatives` negatives with the same source.
    """
    model.eval()
    rng = np.random.default_rng(seed)
    src_candidates = pos_edges[:, 0]
    neg_edges = sample_negative_edges(
        len(pos_edges),
        len(all_nbrs),
        src_candidates,
        all_nbrs,
        rng,
        hard_pools=hard_pools if negative_mode == "two_hop" else None,
        hard_ratio=hard_ratio if negative_mode == "two_hop" else 0.0,
    )

    with torch.no_grad():
        z = model(adj_follow, adj_tx, tx_features, degree_feature=degree_feature)
        tx_emb = model.build_tx_embedding(tx_features, degree_feature=degree_feature) if model.use_tx_features else None
        pos_scores = score_edges_in_batches(
            model=model,
            z=z,
            edges=pos_edges,
            device=device,
            batch_size=eval_edge_batch_size,
            tx_emb=tx_emb,
            degree_feature=degree_feature,
        )
        neg_scores = score_edges_in_batches(
            model=model,
            z=z,
            edges=neg_edges,
            device=device,
            batch_size=eval_edge_batch_size,
            tx_emb=tx_emb,
            degree_feature=degree_feature,
        )

    y_true = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
    y_score = np.concatenate([pos_scores, neg_scores])
    y_score = np.nan_to_num(y_score, nan=0.5, posinf=1.0, neginf=0.0)
    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    eps = 1e-12
    y_score_clip = np.clip(y_score, eps, 1 - eps)
    logloss = float(-np.mean(y_true * np.log(y_score_clip) + (1 - y_true) * np.log(1 - y_score_clip)))
    brier = float(np.mean((y_score - y_true) ** 2))

    # Sampled ranking metrics (pessimistic ties: rank = #negatives scoring >= positive, + 1).
    eval_n = min(2000, len(pos_edges))
    hit = 0
    mrr_sum = 0.0
    ndcg_sum = 0.0
    prec_sum = 0.0
    rec_sum = 0.0
    with torch.no_grad():
        for i in range(eval_n):
            src, dst_pos = pos_edges[i]
            negs = sample_negative_edges(
                rank_negatives,
                len(all_nbrs),
                np.array([src], dtype=np.int64),
                all_nbrs,
                rng,
                hard_pools=hard_pools if negative_mode == "two_hop" else None,
                hard_ratio=hard_ratio if negative_mode == "two_hop" else 0.0,
            )
            candidates = np.vstack([np.array([[src, dst_pos]], dtype=np.int64), negs])
            cand_t = torch.tensor(candidates, dtype=torch.long, device=device)
            scores = torch.sigmoid(
                model.decode(z, cand_t, tx_emb=tx_emb, degree_feature=degree_feature)
            ).cpu().numpy()
            rank = int(np.sum(scores[1:] >= scores[0])) + 1
            if rank <= hits_k:
                hit += 1
                prec_sum += 1.0 / max(1, hits_k)
                rec_sum += 1.0
            mrr_sum += 1.0 / rank
            ndcg_sum += 1.0 / math.log2(rank + 1)
    hits = hit / max(1, eval_n)
    mrr = mrr_sum / max(1, eval_n)
    ndcg = ndcg_sum / max(1, eval_n)
    precision_k = prec_sum / max(1, eval_n)
    recall_k = rec_sum / max(1, eval_n)

    return {
        "auc": float(auc),
        "ap": float(ap),
        "logloss": float(logloss),
        "brier": float(brier),
        f"hits@{hits_k}": float(hits),
        "mrr": float(mrr),
        f"ndcg@{hits_k}": float(ndcg),
        f"precision@{hits_k}": float(precision_k),
        f"recall@{hits_k}": float(recall_k),
    }


def build_out_degree(edges: List[Tuple[int, int]], num_nodes: int) -> np.ndarray:
    deg = np.zeros((num_nodes,), dtype=np.int64)
    for u, _ in edges:
        deg[u] += 1
    return deg


def evaluate_by_source_degree_buckets(
    model: HybridGNN,
    adj_follow: torch.Tensor,
    adj_tx: Optional[torch.Tensor],
    tx_features: torch.Tensor,
    pos_edges: np.ndarray,
    all_nbrs: List[Set[int]],
    device: str,
    seed: int,
    train_out_degree: np.ndarray,
    args,
    hard_pools: Optional[List[Optional[np.ndarray]]] = None,
    degree_feature: Optional[torch.Tensor] = None,
) -> Dict[str, Dict[str, float]]:
    """Evaluate test edges grouped by the source's out-degree in the training graph."""
    src_deg = train_out_degree[pos_edges[:, 0]]
    bucket_masks = {
        "deg0": src_deg == 0,
        "deg1_3": (src_deg >= 1) & (src_deg <= 3),
        "deg4_10": (src_deg >= 4) & (src_deg <= 10),
        "deg_gt10": src_deg > 10,
    }
    out = {}
    for name, mask in bucket_masks.items():
        idx = np.where(mask)[0]
        if len(idx) < 200:
            continue
        sub = pos_edges[idx]
        out[name] = evaluate_link_prediction(
            model=model,
            adj_follow=adj_follow,
            adj_tx=adj_tx,
            tx_features=tx_features,
            pos_edges=sub,
            all_nbrs=all_nbrs,
            device=device,
            seed=seed + len(name),
            hits_k=args.hits_k,
            eval_edge_batch_size=args.eval_edge_batch_size,
            negative_mode=args.eval_negative_mode,
            hard_pools=hard_pools,
            hard_ratio=args.hard_negative_ratio,
            rank_negatives=args.rank_negatives,
            degree_feature=degree_feature,
        )
        out[name]["num_edges"] = int(len(idx))
    return out


def run_training(
    name: str,
    use_tx_features: bool,
    use_tx_graph: bool,
    train_edges: np.ndarray,
    val_edges: np.ndarray,
    test_edges: np.ndarray,
    train_adj_follow: torch.Tensor,
    train_adj_tx: Optional[torch.Tensor],
    tx_features: torch.Tensor,
    all_nbrs: List[Set[int]],
    num_nodes: int,
    args,
    device: str,
    train_nbrs: Optional[List[Set[int]]] = None,
    train_out_degree: Optional[np.ndarray] = None,
    degree_feature: Optional[torch.Tensor] = None,
) -> Dict[str, Dict[str, float]]:
    model = HybridGNN(
        num_nodes=num_nodes,
        tx_feat_dim=tx_features.shape[1],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_tx_features=use_tx_features,
        use_tx_feature_gate=bool(args.tx_feature_gate),
        degree_aware_gate=bool(args.degree_aware_gate),
        degree_gate_alpha=float(args.degree_gate_alpha),
        degree_gate_beta=float(args.degree_gate_beta),
        degree_gate_cutoff=float(args.degree_gate_cutoff),
        degree_gate_sharpness=float(args.degree_gate_sharpness),
        degree_gate_min=float(args.degree_gate_min),
        residual_tx_decode=bool(args.residual_tx_decode),
        residual_tx_scale=float(args.residual_tx_scale),
        source_degree_tx_max=float(args.source_degree_tx_max),
        tx_graph_weight=args.tx_graph_weight if use_tx_graph else 0.0,
        residual=args.residual,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = -1.0
    best_state = None
    history = []

    train_edge_batch_size = max(1, int(args.train_edge_batch_size))
    num_train = len(train_edges)
    train_indices = np.arange(num_train)
    hard_pools = None
    if args.negative_mode == "two_hop":
        base_nbrs = train_nbrs if train_nbrs is not None else all_nbrs
        hard_pools = build_two_hop_hard_pools(
            base_nbrs,
            mid_cap=args.two_hop_mid_cap,
            pool_cap=args.two_hop_pool_cap,
        )
        print("[neg-sampling] using two_hop hard negatives")
    for epoch in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(train_indices)
        epoch_loss_sum = 0.0
        epoch_samples = 0

        for s in range(0, num_train, train_edge_batch_size):
            e = min(s + train_edge_batch_size, num_train)
            batch_idx = train_indices[s:e]
            pos_batch = train_edges[batch_idx]

            neg_batch = sample_negative_edges(
                len(pos_batch),
                num_nodes,
                pos_batch[:, 0],
                all_nbrs,
                rng,
                hard_pools=hard_pools if args.negative_mode == "two_hop" else None,
                hard_ratio=args.hard_negative_ratio if args.negative_mode == "two_hop" else 0.0,
            )
            pos_t = torch.tensor(pos_batch, dtype=torch.long, device=device)
            neg_t = torch.tensor(neg_batch, dtype=torch.long, device=device)

            z = model(
                train_adj_follow,
                train_adj_tx if use_tx_graph else None,
                tx_features,
                degree_feature=degree_feature,
            )
            tx_emb = model.build_tx_embedding(tx_features, degree_feature=degree_feature) if model.use_tx_features else None
            pos_logits = model.decode(z, pos_t, tx_emb=tx_emb, degree_feature=degree_feature)
            neg_logits = model.decode(z, neg_t, tx_emb=tx_emb, degree_feature=degree_feature)
            logits = torch.cat([pos_logits, neg_logits], dim=0)
            labels = torch.cat(
                [
                    torch.ones_like(pos_logits),
                    torch.zeros_like(neg_logits),
                ],
                dim=0,
            )

            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            batch_n = int(labels.shape[0])
            epoch_loss_sum += float(loss.item()) * batch_n
            epoch_samples += batch_n

        epoch_loss = epoch_loss_sum / max(1, epoch_samples)
        torch.cuda.empty_cache()

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val_metrics = evaluate_link_prediction(
                model,
                train_adj_follow,
                train_adj_tx if use_tx_graph else None,
                tx_features,
                val_edges,
                all_nbrs,
                device,
                seed=args.seed + epoch,
                hits_k=args.hits_k,
                eval_edge_batch_size=args.eval_edge_batch_size,
                negative_mode=args.eval_negative_mode,
                hard_pools=hard_pools,
                hard_ratio=args.hard_negative_ratio,
                rank_negatives=args.rank_negatives,
                degree_feature=degree_feature,
            )
            row = {"epoch": epoch, "loss": float(epoch_loss), **val_metrics}
            history.append(row)
            if val_metrics["auc"] > best_val_auc:
                best_val_auc = val_metrics["auc"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(
                f"[{name}] epoch={epoch:03d} loss={epoch_loss:.4f} val_auc="
                f"{val_metrics['auc']:.4f} val_ap={val_metrics['ap']:.4f} val_hits@"
                f"{args.hits_k}={val_metrics[f'hits@{args.hits_k}']:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate_link_prediction(
        model,
        train_adj_follow,
        train_adj_tx if use_tx_graph else None,
        tx_features,
        test_edges,
        all_nbrs,
        device,
        seed=args.seed + 999,
        hits_k=args.hits_k,
        eval_edge_batch_size=args.eval_edge_batch_size,
        negative_mode=args.eval_negative_mode,
        hard_pools=hard_pools,
        hard_ratio=args.hard_negative_ratio,
        rank_negatives=args.rank_negatives,
        degree_feature=degree_feature,
    )
    print(
        f"[{name}] TEST auc={test_metrics['auc']:.4f} ap="
        f"{test_metrics['ap']:.4f} hits@{args.hits_k}={test_metrics[f'hits@{args.hits_k}']:.4f}"
    )
    bucket_metrics = {}
    if train_out_degree is not None:
        bucket_metrics = evaluate_by_source_degree_buckets(
            model=model,
            adj_follow=train_adj_follow,
            adj_tx=train_adj_tx if use_tx_graph else None,
            tx_features=tx_features,
            pos_edges=test_edges,
            all_nbrs=all_nbrs,
            device=device,
            seed=args.seed + 12345,
            train_out_degree=train_out_degree,
            args=args,
            hard_pools=hard_pools,
            degree_feature=degree_feature,
        )

    return {"history": history, "test": test_metrics, "test_by_source_degree": bucket_metrics}


def save_analysis_plots(
    out_dir: Path,
    tx_feats: np.ndarray,
    follow_deg: np.ndarray,
    tx_deg: np.ndarray,
) -> None:
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    axs[0].hist(tx_feats[:, 0], bins=60, color="#4472C4", alpha=0.9)
    axs[0].set_title("Tx feature: log(1+tx_events) (z-score)")
    axs[0].set_xlabel("value")
    axs[0].set_ylabel("count")

    axs[1].hist(follow_deg, bins=60, color="#70AD47", alpha=0.9)
    axs[1].set_title("Follow out-degree distribution")
    axs[1].set_xlabel("out-degree")

    axs[2].hist(tx_deg, bins=60, color="#ED7D31", alpha=0.9)
    axs[2].set_title("Tx-sim graph degree distribution")
    axs[2].set_xlabel("degree")

    plt.tight_layout()
    fig.savefig(out_dir / "analysis_plots.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train tx-enhanced follow recommendation GNN.")
    parser.add_argument("--links-path", type=str, default="data/links.jsonl")
    parser.add_argument("--tx-path", type=str, default="fid_main_addr_all_transactions.jsonl")
    parser.add_argument("--out-dir", type=str, default="runs/tx_enhanced_gnn")
    parser.add_argument("--max-link-rows", type=int, default=15000000, help="0 means no limit")
    parser.add_argument("--max-tx-rows", type=int, default=0, help="0 means no limit")
    parser.add_argument("--max-follow-edges", type=int, default=4000000, help="0 means no limit")
    parser.add_argument("--counterparty-fid-cap", type=int, default=64)
    parser.add_argument("--min-shared-counterparty", type=int, default=2)
    parser.add_argument("--max-fids-per-counterparty-for-edges", type=int, default=30)
    parser.add_argument("--min-tx-pair-weight", type=int, default=2)
    parser.add_argument("--tx-shared-weight-scale", type=float, default=1.0)
    parser.add_argument("--tx-knn-k", type=int, default=24)
    parser.add_argument("--tx-knn-weight-scale", type=float, default=0.7)
    parser.add_argument("--tx-knn-min-tx-events", type=int, default=1)
    parser.add_argument("--tx-knn-max-edges", type=int, default=0, help="0 means no limit")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--split-mode", type=str, default="random", choices=["random", "temporal"])
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--residual", type=float, default=0.15)
    parser.add_argument("--tx-graph-weight", type=float, default=0.2)
    parser.add_argument("--tx-feature-gate", type=int, default=1, choices=[0, 1])
    parser.add_argument("--degree-aware-gate", type=int, default=1, choices=[0, 1])
    parser.add_argument("--degree-gate-alpha", type=float, default=1.2)
    parser.add_argument("--degree-gate-beta", type=float, default=1.0)
    parser.add_argument("--degree-gate-cutoff", type=float, default=0.35)
    parser.add_argument("--degree-gate-sharpness", type=float, default=10.0)
    parser.add_argument("--degree-gate-min", type=float, default=0.1)
    parser.add_argument("--residual-tx-decode", type=int, default=1, choices=[0, 1])
    parser.add_argument("--residual-tx-scale", type=float, default=0.35)
    parser.add_argument("--source-degree-tx-max", type=float, default=1.0)
    parser.add_argument("--enhanced-use-tx-features", type=int, default=1, choices=[0, 1])
    parser.add_argument("--enhanced-use-tx-graph", type=int, default=1, choices=[0, 1])
    parser.add_argument("--tx-feature-active-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hits-k", type=int, default=10)
    parser.add_argument("--train-edge-batch-size", type=int, default=200000)
    parser.add_argument("--eval-edge-batch-size", type=int, default=200000)
    parser.add_argument("--negative-mode", type=str, default="random", choices=["random", "two_hop"])
    parser.add_argument("--eval-negative-mode", type=str, default="random", choices=["random", "two_hop"])
    parser.add_argument("--rank-negatives", type=int, default=99)
    parser.add_argument("--hard-negative-ratio", type=float, default=0.7)
    parser.add_argument("--two-hop-mid-cap", type=int, default=48)
    parser.add_argument("--two-hop-pool-cap", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    print(f"Device: {device}")

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Artifacts -> {out_dir}")

    t0 = time.time()
    print("Step 1/5: Parse transaction file...")
    fid_stats, tx_pair_weights, tx_summary = stream_tx_features(
        tx_path=args.tx_path,
        max_tx_rows=args.max_tx_rows,
        counterparty_fid_cap=args.counterparty_fid_cap,
        min_shared_counterparty=args.min_shared_counterparty,
        max_fids_per_counterparty_for_edges=args.max_fids_per_counterparty_for_edges,
    )
    tx_fids = set(fid_stats.keys())
    if len(tx_fids) == 0:
        raise RuntimeError("No tx fids parsed. Check --tx-path or tx file format.")
    print(f"Tx fids: {len(tx_fids)} | tx similarity pairs(raw): {len(tx_pair_weights)}")

    print("Step 2/5: Parse follow file and build follow edges...")
    follow_edges_raw, follow_summary = stream_follow_edges(
        links_path=args.links_path,
        allowed_fids=tx_fids,
        max_link_rows=args.max_link_rows,
        max_follow_edges=args.max_follow_edges,
    )
    if len(follow_edges_raw) < 1000:
        raise RuntimeError(
            f"Too few follow edges retained ({len(follow_edges_raw)}). Increase --max-link-rows or check whether tx fids overlap with links."
        )
    print(f"Follow edges kept: {len(follow_edges_raw)}")

    follow_edge_items = list(follow_edges_raw.items())
    all_fids = sorted({u for (u, _), _ in follow_edge_items} | {v for (_, v), _ in follow_edge_items})
    fid_to_idx = {fid: i for i, fid in enumerate(all_fids)}
    num_nodes = len(all_fids)
    print(f"Graph nodes in overlap set: {num_nodes}")

    follow_edges_with_time = [((fid_to_idx[u], fid_to_idx[v]), ts) for (u, v), ts in follow_edge_items]
    edges_list = [e for e, _ in follow_edges_with_time]
    if args.split_mode == "temporal":
        train_e, val_e, test_e = split_edges_temporal(
            follow_edges_with_time,
            args.val_ratio,
            args.test_ratio,
            args.seed,
        )
    else:
        train_e, val_e, test_e = split_edges(edges_list, args.val_ratio, args.test_ratio, args.seed)
    print(f"Split -> train:{len(train_e)} val:{len(val_e)} test:{len(test_e)}")

    print("Step 3/5: Build tx-sim graph and node features...")
    tx_edges_shared = []
    for (a, b), w in tx_pair_weights.items():
        if w >= args.min_tx_pair_weight and a in fid_to_idx and b in fid_to_idx:
            u = fid_to_idx[a]
            v = fid_to_idx[b]
            weight = math.log1p(float(w)) * args.tx_shared_weight_scale
            tx_edges_shared.append((u, v, weight))
    print(f"Tx shared-counterparty edges: {len(tx_edges_shared)}")

    tx_feats_np = build_tx_feature_matrix(fid_stats, fid_to_idx)

    active_nodes = []
    for fid, idx in fid_to_idx.items():
        if fid_stats[fid].tx_events >= args.tx_knn_min_tx_events:
            active_nodes.append(idx)
    active_nodes_np = np.array(active_nodes, dtype=np.int64)
    if args.tx_feature_active_only and len(active_nodes) > 0:
        active_mask = np.zeros((num_nodes, 1), dtype=np.float32)
        active_mask[active_nodes_np, 0] = 1.0
        tx_feats_np = tx_feats_np * active_mask

    tx_feats = torch.tensor(tx_feats_np, dtype=torch.float32, device=device)

    tx_edges_knn = build_tx_knn_edges(
        tx_feats=tx_feats_np,
        active_nodes=active_nodes_np,
        knn_k=args.tx_knn_k,
        weight_scale=args.tx_knn_weight_scale,
        max_edges=args.tx_knn_max_edges,
    )
    print(f"Tx kNN edges: {len(tx_edges_knn)} (k={args.tx_knn_k}, active_nodes={len(active_nodes)})")

    # Fuse: shared-counterparty weights add up, kNN weights take the max.
    tx_edge_map = {}
    for u, v, w in tx_edges_shared:
        tx_edge_map[(u, v)] = tx_edge_map.get((u, v), 0.0) + float(w)
    for u, v, w in tx_edges_knn:
        prev = tx_edge_map.get((u, v), 0.0)
        tx_edge_map[(u, v)] = max(prev, float(w))
    tx_edges = [(u, v, w) for (u, v), w in tx_edge_map.items()]
    print(f"Tx fused edges total: {len(tx_edges)}")

    follow_nbrs_all = build_neighbor_map(edges_list, num_nodes)
    follow_nbrs_train = build_neighbor_map(train_e, num_nodes)
    follow_out_degree_train = build_out_degree(train_e, num_nodes)
    # Degree feature in [0, 1]: log1p(train out-degree) / max log1p(train out-degree).
    deg_log = np.log1p(follow_out_degree_train.astype(np.float32))
    deg_norm = deg_log / max(1e-6, float(np.max(deg_log)))
    degree_feature = torch.tensor(deg_norm.reshape(-1, 1), dtype=torch.float32, device=device)
    train_adj_follow = make_sparse_norm_adj(train_e, num_nodes, undirected=True, add_self_loops=True, device=device)
    train_adj_tx = make_sparse_norm_adj(tx_edges, num_nodes, undirected=True, add_self_loops=True, device=device)

    follow_deg = np.array([len(s) for s in follow_nbrs_all], dtype=np.int32)
    tx_edges_unweighted = [(u, v) for u, v, _ in tx_edges]
    tx_nbrs = build_neighbor_map(tx_edges_unweighted + [(b, a) for a, b in tx_edges_unweighted], num_nodes)
    tx_deg = np.array([len(s) for s in tx_nbrs], dtype=np.int32)
    save_analysis_plots(out_dir, tx_feats_np, follow_deg, tx_deg)

    train_e_np = np.array(train_e, dtype=np.int64)
    val_e_np = np.array(val_e, dtype=np.int64)
    test_e_np = np.array(test_e, dtype=np.int64)

    print("Step 4/5: Train baseline model (follow only)...")
    baseline_result = run_training(
        name="baseline",
        use_tx_features=False,
        use_tx_graph=False,
        train_edges=train_e_np,
        val_edges=val_e_np,
        test_edges=test_e_np,
        train_adj_follow=train_adj_follow,
        train_adj_tx=None,
        tx_features=tx_feats,
        all_nbrs=follow_nbrs_all,
        num_nodes=num_nodes,
        args=args,
        device=device,
        train_nbrs=follow_nbrs_train,
        train_out_degree=follow_out_degree_train,
        degree_feature=degree_feature,
    )

    print("Step 5/5: Train tx-enhanced model...")
    enhanced_result = run_training(
        name="tx_enhanced",
        use_tx_features=bool(args.enhanced_use_tx_features),
        use_tx_graph=bool(args.enhanced_use_tx_graph),
        train_edges=train_e_np,
        val_edges=val_e_np,
        test_edges=test_e_np,
        train_adj_follow=train_adj_follow,
        train_adj_tx=train_adj_tx,
        tx_features=tx_feats,
        all_nbrs=follow_nbrs_all,
        num_nodes=num_nodes,
        args=args,
        device=device,
        train_nbrs=follow_nbrs_train,
        train_out_degree=follow_out_degree_train,
        degree_feature=degree_feature,
    )

    delta = {
        k: enhanced_result["test"][k] - baseline_result["test"][k]
        for k in baseline_result["test"].keys()
    }

    report = {
        "config": vars(args),
        "device": device,
        "tx_summary": tx_summary,
        "follow_summary": follow_summary,
        "graph_stats": {
            "num_nodes": num_nodes,
            "follow_edges": len(edges_list),
            "tx_sim_edges": len(tx_edges),
            "tx_shared_edges": len(tx_edges_shared),
            "tx_knn_edges": len(tx_edges_knn),
            "tx_knn_active_nodes": len(active_nodes),
            "train_edges": len(train_e),
            "val_edges": len(val_e),
            "test_edges": len(test_e),
        },
        "baseline": baseline_result,
        "tx_enhanced": enhanced_result,
        "delta_test_metrics": delta,
        "elapsed_seconds": time.time() - t0,
    }

    with open(out_dir / "metrics_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    with open(out_dir / "run_summary.md", "w", encoding="utf-8") as f:
        f.write("# Tx-Enhanced Follow Recommendation Results\n\n")
        f.write(f"- Device: `{device}`\n")
        f.write(f"- Nodes: `{num_nodes}`\n")
        f.write(f"- Follow edges: `{len(edges_list)}`\n")
        f.write(f"- Tx-sim edges: `{len(tx_edges)}`\n\n")
        f.write("## Test Metrics\n\n")
        f.write("| Model | AUC | AP | Hits@K |\n")
        f.write("|---|---:|---:|---:|\n")
        f.write(
            f"| baseline | {baseline_result['test']['auc']:.4f} | "
            f"{baseline_result['test']['ap']:.4f} | "
            f"{baseline_result['test'][f'hits@{args.hits_k}']:.4f} |\n"
        )
        f.write(
            f"| tx_enhanced | {enhanced_result['test']['auc']:.4f} | "
            f"{enhanced_result['test']['ap']:.4f} | "
            f"{enhanced_result['test'][f'hits@{args.hits_k}']:.4f} |\n"
        )
        f.write(
            f"| delta | {delta['auc']:+.4f} | {delta['ap']:+.4f} | "
            f"{delta[f'hits@{args.hits_k}']:+.4f} |\n"
        )

        b_base = baseline_result.get("test_by_source_degree", {})
        b_tx = enhanced_result.get("test_by_source_degree", {})
        if b_base and b_tx:
            f.write("\n## Cold-Start / Degree Buckets (Test)\n\n")
            f.write("| Bucket | Baseline AUC | Tx AUC | Delta AUC | Baseline Hits@K | Tx Hits@K | Delta Hits@K | N |\n")
            f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
            common = sorted(set(b_base.keys()) & set(b_tx.keys()))
            for bk in common:
                auc_b = b_base[bk].get("auc", 0.0)
                auc_t = b_tx[bk].get("auc", 0.0)
                hk = f"hits@{args.hits_k}"
                hit_b = b_base[bk].get(hk, 0.0)
                hit_t = b_tx[bk].get(hk, 0.0)
                n = int(b_base[bk].get("num_edges", 0))
                f.write(
                    f"| {bk} | {auc_b:.4f} | {auc_t:.4f} | {auc_t - auc_b:+.4f} | "
                    f"{hit_b:.4f} | {hit_t:.4f} | {hit_t - hit_b:+.4f} | {n} |\n"
                )

    print(f"Done. Report -> {out_dir / 'metrics_report.json'}")
    print(f"Summary -> {out_dir / 'run_summary.md'}")
    print(f"Plots -> {out_dir / 'analysis_plots.png'}")


if __name__ == "__main__":
    main()
