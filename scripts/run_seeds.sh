#!/usr/bin/env bash
# Run the main experiment for several seeds, one after another.
#
# Usage:
#   LINKS=/data/links.jsonl TX=/data/fid_main_addr_all_transactions.jsonl \
#     scripts/run_seeds.sh [first_seed] [last_seed] [out_root] [args_file]
#
# Defaults: seeds 92..99, out_root runs/main, args_file configs/main_experiment.args.
# Each seed writes runs/main/tx200k_seed<S>/<timestamp>/metrics_report.json,
# which scripts/paired_stats.py and scripts/collect_results.py read.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FIRST="${1:-92}"
LAST="${2:-99}"
OUT_ROOT="${3:-runs/main}"
ARGS_FILE="${4:-$ROOT/configs/main_experiment.args}"
PYTHON="${PYTHON:-python}"
: "${LINKS:?set LINKS to the follow-link JSONL (see docs/DATA.md)}"
: "${TX:?set TX to the wallet-transaction JSONL (see docs/DATA.md)}"

# Read the flag file into an array, skipping comments and blank lines.
FLAGS=()
while read -r line; do
  [[ -z "$line" || "$line" == \#* ]] && continue
  read -r -a parts <<< "$line"
  FLAGS+=("${parts[@]}")
done < "$ARGS_FILE"

for seed in $(seq "$FIRST" "$LAST"); do
  echo "=== seed $seed ==="
  "$PYTHON" "$ROOT/artifact/train_tx_enhanced_gnn.py" \
    --links-path "$LINKS" --tx-path "$TX" \
    "${FLAGS[@]}" \
    --seed "$seed" --out-dir "$OUT_ROOT/tx200k_seed$seed"
done
echo "All seeds done. Next: python scripts/paired_stats.py $OUT_ROOT"
