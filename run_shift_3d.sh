#!/bin/bash
set -euo pipefail

# Structural distribution-shift study (3D): calibrate the functional-CP envelope
# once on a CV-leaning mixture, freeze it, then sweep the deployment mixture
# toward turning / stop-and-go motion (beta) for the static vs. AFCP controllers.
# Finally build the coverage-vs-beta LaTeX table.

# defaults
BETAS="0,0.25,0.5,0.75,1.0"
METHODS="static,afcp"
SEED_FROM=20
SEED_TO=36
N_OBS=50
MAX_STEPS=400
I_COV=1
OUT_DIR="metric_3d/shift"
TABLE_OUT="tables/table_shift_3d.tex"
COV_FIELD="coverage"
REFIT=0

PY="python3"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --betas) BETAS="$2"; shift 2 ;;
    --methods) METHODS="$2"; shift 2 ;;
    --seed-from) SEED_FROM="$2"; shift 2 ;;
    --seed-to) SEED_TO="$2"; shift 2 ;;
    --n-obs) N_OBS="$2"; shift 2 ;;
    --max-steps) MAX_STEPS="$2"; shift 2 ;;
    --i-cov) I_COV="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --table-out) TABLE_OUT="$2"; shift 2 ;;
    --cov-field) COV_FIELD="$2"; shift 2 ;;
    --refit) REFIT=1; shift 1 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

mkdir -p "${OUT_DIR}"

echo "[run] betas=${BETAS} methods=${METHODS} seeds=${SEED_FROM}-${SEED_TO} n_obs=${N_OBS} max_steps=${MAX_STEPS} i_cov=${I_COV}"

EXTRA=()
if [[ "${REFIT}" -eq 1 ]]; then EXTRA+=(--refit); fi

${PY} quadrotor/run_shift_3d.py \
  --betas "${BETAS}" \
  --methods "${METHODS}" \
  --seed-from "${SEED_FROM}" \
  --seed-to "${SEED_TO}" \
  --n-obs "${N_OBS}" \
  --max-steps "${MAX_STEPS}" \
  --i-cov "${I_COV}" \
  --out-dir "${OUT_DIR}" \
  --cache "${OUT_DIR}/frozen_envelope.pkl" \
  "${EXTRA[@]}"

${PY} quadrotor/make_table_shift_3d.py \
  --json "${OUT_DIR}/shift_suite.json" \
  --out "${TABLE_OUT}" \
  --cov-field "${COV_FIELD}"

echo "[done] table -> ${TABLE_OUT}"
