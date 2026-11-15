# Data

## What the experiments use

The training script reads two JSON-lines files. Both are built from public
data: Farcaster hub messages and on-chain wallet activity.

| File | Content | Size used in the thesis |
|---|---|---|
| `links.jsonl` | Farcaster link messages (who follows whom, with timestamps) | first 1,000,000 rows → 250,000 follow edges |
| `fid_main_addr_all_transactions.jsonl` | one row per (account, verified wallet) with that wallet's transactions | first 200,000 rows → 14,801 accounts, 56,079 payments |

The full raw collection is about **1.1 TB**, too large for this repository.

**Download:** Baidu Netdisk, <https://pan.baidu.com/s/1KDDEf7P8fcx1kIZNC5WJog?pwd=iv97>
(extraction code `iv97`). Upload is still in progress; available so far:

| File | Size |
|---|---|
| `fid_main_addr_all_transactions.jsonl` | 180.6 GB |
| `fid_main_addr_all_transactions_part2.jsonl` | 478 GB |
| `fid_main_addr_all_transactions_part3.jsonl` | 75.6 GB |

The thesis experiments read the first 200,000 rows of
`fid_main_addr_all_transactions.jsonl`. The data can also be obtained on an
external drive from the author. Everything is public and can be re-collected
from a Farcaster hub (link messages) and a blockchain data provider (address
transactions).

## Format

### `links.jsonl`

One Farcaster hub message per line. Only `LinkAdd` messages of type `follow`
are used; everything else is skipped.

```json
{"fid": 1234,
 "message": {"data": {"type": "MESSAGE_TYPE_LINK_ADD",
                      "fid": 1234,
                      "timestamp": 98765432,
                      "linkBody": {"type": "follow", "targetFid": 5678}}}}
```

- `fid` follows `targetFid`. Self-follows are dropped.
- `timestamp` (Farcaster time) gives the follow's time. If the same follow
  appears twice, the earliest timestamp is kept. The temporal split sorts on it:
  oldest 80% train, next 10% validation, newest 10% test.
- Only follows between accounts that appear in the transaction file are kept.

### `fid_main_addr_all_transactions.jsonl`

One line per (account, verified wallet address). Transactions are listed
newest first.

```json
{"fid": 1234,
 "address": "0xabc...",
 "rpc_result": {"result": {"total_count": 2,
   "addressTransactions": [
     {"hash": "0x...", "status": "CONFIRMED",
      "ethereum": {"from": "0xabc...", "to": "0xdef...", "value": "5000000000000000"}},
     {"hash": "0x...", "status": "CONFIRMED",
      "ethereum": {"from": "0x999...", "to": "0xabc...", "value": "120000000000000000"}}]}}}
```

- An account with an empty `addressTransactions` list is kept (about 80% of
  accounts); the wallet gate learns to ignore it.
- `value` is in wei. Direction (in, out, self) is read from `from`/`to`
  relative to `address`; the other side is the counterparty.
- From these rows the script computes 20 features per account (counts,
  values, unique counterparties, recency, confirmed share). They are listed in
  Table 2.2 of the thesis.

## Running without the raw data

- **Check the pipeline:** `scripts/make_toy_data.py` writes small synthetic
  files in exactly this format, and `tests/smoke_test.sh` trains on them. The
  toy numbers have no meaning.
- **Rebuild the reported numbers and figures:** `data/raw/` holds every run's
  `metrics_report.json`; `scripts/collect_results.py`, `scripts/paired_stats.py`
  and `scripts/make_figures.py` work from those alone.

## Not included

- The raw files (size).
- `data/sample/graph_sample.json`, used for the network figure: it lists real
  Farcaster account IDs. `scripts/extract_graph_sample.py` rebuilds it from the
  raw files.
