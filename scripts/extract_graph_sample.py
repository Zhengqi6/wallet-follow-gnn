"""Run on i1 (read-only). Rebuilds the experiment's sample the same way the
training pipeline did and prints it as JSON on stdout:

  - per-fid transaction-event counts for the first 200,000 wallet records
  - the first 250,000 follow edges whose two endpoints are both in that set

Usage (from the Social_Network directory):
  python3 extract_graph_sample.py > sample.json
"""
import json
import sys

TX_PATH = "fid_main_addr_all_transactions.jsonl"
LINKS_PATH = "data/links.jsonl"
MAX_TX_ROWS = 200_000
MAX_LINK_ROWS = 1_000_000
MAX_FOLLOW_EDGES = 250_000

tx_count = {}
with open(TX_PATH) as f:
    for i, line in enumerate(f):
        if i >= MAX_TX_ROWS:
            break
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        txs = ((row.get("rpc_result") or {}).get("result") or {}).get("addressTransactions") or []
        tx_count[int(row["fid"])] = tx_count.get(int(row["fid"]), 0) + len(txs)
print(f"tx rows read: {i}, fids: {len(tx_count)}, events: {sum(tx_count.values())}", file=sys.stderr)

edges = {}
with open(LINKS_PATH) as f:
    for i, line in enumerate(f):
        if i >= MAX_LINK_ROWS or len(edges) >= MAX_FOLLOW_EDGES:
            break
        try:
            row = json.loads(line)
            data = row["message"]["data"]
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
        if data.get("type") != "MESSAGE_TYPE_LINK_ADD":
            continue
        body = data.get("linkBody") or {}
        if body.get("type") != "follow":
            continue
        src, dst = int(row["fid"]), int(body.get("targetFid") or -1)
        if src == dst or src not in tx_count or dst not in tx_count:
            continue
        ts = int(data.get("timestamp") or 0)
        if (src, dst) not in edges or ts < edges[(src, dst)]:
            edges[(src, dst)] = ts
print(f"link rows read: {i}, edges: {len(edges)}", file=sys.stderr)

nodes = sorted({u for e in edges for u in e})
json.dump({
    "tx_count": {str(u): tx_count[u] for u in nodes},
    "edges": [[u, v, t] for (u, v), t in edges.items()],
}, sys.stdout)
