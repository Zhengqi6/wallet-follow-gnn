#!/usr/bin/env bash
# End-to-end check on synthetic data, about one minute:
#   toy data -> 2 training seeds -> paired statistics -> sanity checks.
# Run it on the A100 node before launching the full experiment (the training
# script uses CUDA automatically; without a GPU it falls back to CPU).
# It checks that the pipeline runs and writes well-formed results; the toy
# numbers themselves carry no meaning.
#
# Usage: PYTHON=python3 tests/smoke_test.sh [work_dir]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${1:-$(mktemp -d)}"
PYTHON="${PYTHON:-python}"
export PYTHON

"$PYTHON" "$ROOT/scripts/make_toy_data.py" --out "$WORK/data" --seed 0
LINKS="$WORK/data/links.jsonl" TX="$WORK/data/transactions.jsonl" \
  bash "$ROOT/scripts/run_seeds.sh" 1 2 "$WORK/runs" "$ROOT/configs/toy.args" > "$WORK/train.log" 2>&1 \
  || { tail -40 "$WORK/train.log"; echo "FAIL: training"; exit 1; }
"$PYTHON" "$ROOT/scripts/paired_stats.py" "$WORK/runs" --out "$WORK/stats" > /dev/null

"$PYTHON" - "$WORK" <<'PY'
import glob, json, math, sys
work = sys.argv[1]
reports = sorted(glob.glob(f"{work}/runs/*seed*/*/metrics_report.json"))
assert len(reports) == 2, f"expected 2 runs, found {len(reports)}"
for path in reports:
    r = json.load(open(path))
    for model in ("baseline", "tx_enhanced"):
        test = r[model]["test"]
        for m in ("auc", "ap", "hits@10", "mrr"):
            assert 0.0 <= test[m] <= 1.0 and not math.isnan(test[m]), (path, model, m, test[m])
        assert r[model]["test_by_source_degree"], "missing degree buckets"
stats = json.load(open(f"{work}/stats/paired_stats.json"))
assert stats[0]["seeds"] == [1, 2] and "auc" in stats[0]["overall"]
print(f"PASS: 2 seeds trained, metrics valid, paired stats written ({work})")
PY
