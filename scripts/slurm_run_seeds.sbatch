#!/usr/bin/env bash
# Slurm array job: one seed per task (seeds 92..99), one GPU each.
#
#   sbatch --export=ALL,LINKS=/data/links.jsonl,TX=/data/fid_main_addr_all_transactions.jsonl \
#          scripts/slurm_run_seeds.sbatch
#
# Adjust --partition / --gres to your cluster. One seed takes about 2-3
# minutes on an A100 80GB; parsing the input files dominates the run time.
#SBATCH --job-name=wallet-follow
#SBATCH --array=92-99
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs
seed="${SLURM_ARRAY_TASK_ID}"
bash scripts/run_seeds.sh "$seed" "$seed" "${OUT_ROOT:-runs/main}"
