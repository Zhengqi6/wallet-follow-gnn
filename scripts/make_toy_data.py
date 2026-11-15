"""Write a small synthetic dataset in the exact input format of the training script.

The real inputs (about 1.1 TB) cannot ship with the code. This generator
produces two small files with the same JSON layout (see docs/DATA.md), so the
full pipeline -- parsing, feature building, temporal split, training, evaluation
and the paired statistics -- can be checked end to end in about a minute
(on the A100 used for the thesis; the training script falls back to CPU when
no GPU is present).

The data are synthetic. Users belong to hidden communities; users in the same
community are more likely to follow each other, and active wallets in a
community pay a shared set of counterparties. About 80% of wallets are empty,
as in the real sample, so the wallet signal is sparse but real. A share of
users join late: all their follows fall in the last weeks, so the newest test
edges include cold-start users with no follows in the training graph.

Usage:
  python scripts/make_toy_data.py --out toy_data --users 1500 --seed 0
"""
import argparse
import json
import os
import random


def addr(rng):
    return "0x" + "".join(rng.choice("0123456789abcdef") for _ in range(40))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="toy_data")
    ap.add_argument("--users", type=int, default=1500)
    ap.add_argument("--communities", type=int, default=12)
    ap.add_argument("--follows-per-user", type=float, default=12.0)
    ap.add_argument("--active-share", type=float, default=0.2, help="share of wallets with any payment")
    ap.add_argument("--late-share", type=float, default=0.15, help="share of users who join in the last weeks")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)

    fids = list(range(1000, 1000 + args.users))
    community = {f: rng.randrange(args.communities) for f in fids}
    members = {c: [f for f in fids if community[f] == c] for c in range(args.communities)}
    shops = {c: [addr(rng) for _ in range(6)] for c in range(args.communities)}  # shared counterparties
    popular = fids[:15]  # a few accounts everyone tends to follow

    # Wallet transactions: one JSON line per (fid, address), newest transaction first.
    tx_path = os.path.join(args.out, "transactions.jsonl")
    with open(tx_path, "w") as f:
        for fid in fids:
            a = addr(rng)
            txs = []
            if rng.random() < args.active_share:
                for i in range(rng.randint(1, 25)):
                    cp = rng.choice(shops[community[fid]]) if rng.random() < 0.7 else addr(rng)
                    outgoing = rng.random() < 0.6
                    txs.append({
                        "hash": "0x%064x" % rng.getrandbits(256),
                        "status": "CONFIRMED",
                        "ethereum": {
                            "from": a if outgoing else cp,
                            "to": cp if outgoing else a,
                            "value": str(rng.randint(10**15, 5 * 10**18)),
                        },
                    })
            row = {"fid": fid, "address": a,
                   "rpc_result": {"result": {"total_count": len(txs), "addressTransactions": txs}}}
            f.write(json.dumps(row) + "\n")

    # Follow messages in the Farcaster hub layout (LinkAdd, type "follow").
    links_path = os.path.join(args.out, "links.jsonl")
    t0 = 1_700_000_000
    year = 365 * 24 * 3600
    late = set(rng.sample(fids[15:], int(args.late_share * len(fids))))
    edges = set()
    with open(links_path, "w") as f:
        for fid in fids:
            # Activity is heavy-tailed: some users follow many accounts, many follow few.
            k = min(len(fids) - 1, int(rng.paretovariate(1.6) * args.follows_per_user / 2.5))
            for _ in range(k):
                r = rng.random()
                if r < 0.55:
                    dst = rng.choice(members[community[fid]])
                elif r < 0.75:
                    dst = rng.choice(popular)
                else:
                    dst = rng.choice(fids)
                if dst == fid or (fid, dst) in edges:
                    continue
                edges.add((fid, dst))
                if fid in late:
                    ts = t0 + rng.randint(int(0.94 * year), year)
                else:
                    ts = t0 + rng.randint(0, year)
                msg = {"fid": fid, "message": {"data": {
                    "type": "MESSAGE_TYPE_LINK_ADD", "fid": fid, "timestamp": ts,
                    "linkBody": {"type": "follow", "targetFid": dst}}}}
                f.write(json.dumps(msg) + "\n")

    print(f"wrote {tx_path} ({len(fids)} wallets) and {links_path} ({len(edges)} follows)")


if __name__ == "__main__":
    main()
