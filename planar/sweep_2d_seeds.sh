#!/bin/bash
# Measure per-controller control (planning) time in the 2D pedestrian benchmarks
# and rebuild the LaTeX results/ablation tables.
#
# Runs every (dataset, controller) pair SEQUENTIALLY in a single process so the
# per-step control timing (timing_ctrl_ms) is free of CPU contention. Each run
# writes planar/metric/<dataset>_<controller>.json; make_table_2d.py then
# aggregates the masked, per-scene control-time means into the paper tables.
#
# Usage:
#   conda run -n cp bash planar/sweep_2d_seeds.sh                 # all datasets
#   conda run -n cp bash planar/sweep_2d_seeds.sh --datasets zara1 eth
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-python}"   # invoke via `conda run -n cp bash ...` so this resolves to the cp env

DATASETS=("eth" "hotel" "univ" "zara1" "zara2")
CONTROLLERS=("cc" "ecp-mpc" "acp-mpc" \
             "fcp-hard-adaptive" "fcp-hard-nonadaptive" \
             "fcp-soft-adaptive" "fcp-soft-nonadaptive")

while [[ $# -gt 0 ]]; do
  case "$1" in
    --datasets) shift; DATASETS=(); while [[ $# -gt 0 && "$1" != --* ]]; do DATASETS+=("$1"); shift; done ;;
    --controllers) shift; CONTROLLERS=(); while [[ $# -gt 0 && "$1" != --* ]]; do CONTROLLERS+=("$1"); shift; done ;;
    *) echo "unknown arg: $1"; exit 1 ;;
  esac
done

echo "[sweep] datasets: ${DATASETS[*]}"
echo "[sweep] controllers: ${CONTROLLERS[*]}"

for dataset in "${DATASETS[@]}"; do
  for controller in "${CONTROLLERS[@]}"; do
    echo "==== ${dataset} / ${controller} ===="
    "$PY" "$SCRIPT_DIR/runner_2d.py" --dataset "$dataset" --controller "$controller"
  done
done

echo "[sweep] rebuilding 2D tables..."
cd "$SCRIPT_DIR"
"$PY" "$SCRIPT_DIR/make_table_2d.py"
echo "[sweep] done -> planar/tables/table_2d_results.tex , planar/tables/table_2d_ablation.tex"
