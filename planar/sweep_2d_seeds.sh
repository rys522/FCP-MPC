#!/bin/bash
# 2D multi-seed sweep: run every (dataset, controller) over N MPPI sampler seeds so the
# result tables can report mean +/- std. Each run writes metric/{ds}_{ctrl}__s{seed}.json
# (suffix keeps per-seed runs from overwriting each other); aggregate with:
#   python make_table_2d_std.py --seeds 0-9 [--write-paper]
#
# Usage:  ./sweep_2d_seeds.sh [n_seeds]      (default 10 -> seeds 0..9)
# Env:    CONDA_ENV (default urban-nav)
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
source /home/jaeuk/anaconda3/etc/profile.d/conda.sh 2>/dev/null && conda activate "${CONDA_ENV:-urban-nav}"

NSEEDS="${1:-10}"
CTRLS="acp-mpc cc ecp-mpc fcp-hard-nonadaptive fcp-hard-adaptive fcp-soft-nonadaptive fcp-soft-adaptive"
DSETS="eth hotel univ zara1 zara2"
LOGDIR="$HERE/sweep_logs"; mkdir -p "$LOGDIR"
PROG="$LOGDIR/sweep_progress.log"; : > "$PROG"

t0=$(date +%s)
echo "2D seed sweep: ${NSEEDS} seeds x 5 datasets x 7 controllers" | tee -a "$PROG"
for ((sd=0; sd<NSEEDS; sd++)); do
  for ds in $DSETS; do
    for c in $CTRLS; do
      python runner_2d.py --dataset "$ds" --controller "$c" --seed "$sd" --out-suffix "__s$sd" \
        > "$LOGDIR/run_${ds}_${c}_s${sd}.log" 2>&1
      rc=$?
      [ $rc -ne 0 ] && echo "FAIL s$sd/$ds/$c rc=$rc :: $(tail -1 "$LOGDIR/run_${ds}_${c}_s${sd}.log")" >> "$PROG"
    done
  done
  echo "[seed $sd done] elapsed $(( $(date +%s)-t0 ))s" >> "$PROG"
done
echo "SWEEP COMPLETE elapsed $(( $(date +%s)-t0 ))s" | tee -a "$PROG"