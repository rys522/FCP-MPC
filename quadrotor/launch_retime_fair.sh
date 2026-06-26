#!/usr/bin/env bash
# Launch the fair steady-state 3D timing run, detached + screen-off-proof (macOS).
#   - caffeinate -ims : keep the Mac awake (no idle/disk/system sleep) while screen is off
#   - nohup + </dev/null + disown : survive the launching shell / socket closing
set -uo pipefail
cd "$(dirname "$0")/.."          # repo root
OUT=quadrotor/metric_3d/retime_fair
mkdir -p "$OUT"
LOG="$OUT/run.log"

nohup caffeinate -ims conda run --no-capture-output -n cp \
  python quadrotor/retime_fair_3d.py \
    --methods "CC-MPC" "ACP-MPC" "ECP-MPC" "FCP-MPC (hard)" "FCP-MPC (soft)" \
    --densities 10 50 100 150 200 280 \
    --seeds 20 21 22 23 24 \
    --total-steps 45 --warmup-exclude 30 --resume \
  > "$LOG" 2>&1 </dev/null &
disown
echo "launched detached: PID $! ; log -> $LOG"
