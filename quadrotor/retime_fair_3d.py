#!/usr/bin/env python3
"""Fair, contention-free, STEADY-STATE per-step control-time measurement for the 3D
quadrotor benchmark.

Why this exists
---------------
The previous timing pass (make_3d_results.run_timing_sequential) recorded ECP-MPC's
per-step control time from step 0 with ``warmup_steps=0`` and let episodes terminate
early on collision. But ECP's expensive online-calibration loop only engages once its
history exceeds the horizon, with the calibration batch ramping 1->15 over steps
~13-27 (see EgocentricCPMPC3D.evaluate_scores). So an ECP episode that is averaged over
cheap warmup/early-crash steps reports a per-step cost far below its steady state -- the
"ECP@50 is too fast" symptom. CC/ACP/FCP have constant per-step cost, so only ECP was
mismeasured; this also means the ACP~=CC near-equality (shared planner + identical
(P,T,M) distance array) is genuine, not an artifact.

What this does (identical treatment for every method => fair)
-------------------------------------------------------------
  * one episode at a time, single process, 1 BLAS thread  -> contention-free wall clock
  * early termination DISABLED (break_on_collision=False, goal stop disabled) so every
    method runs exactly --total-steps control steps
  * per-step control time is averaged over steps [--warmup-exclude:], i.e. the STEADY
    STATE after ECP's calibration set has filled (constant for CC/ACP/FCP, so excluding
    the warmup window does not change their numbers -- uniform, not ECP-special-cased)
  * full per-step arrays are stored so the ECP ramp is auditable

Outputs (written incrementally so a killed/--resumed run keeps partial progress):
  metric_3d/retime_fair/retime_fair.csv     one row per (method, n_obs, seed)
  metric_3d/retime_fair/retime_fair.json    same + full per-step ctrl arrays

Usage:
  python quadrotor/retime_fair_3d.py --methods CC-MPC ACP-MPC ECP-MPC \
      --densities 50 280 --seeds 20 --total-steps 45 --warmup-exclude 30
"""
from __future__ import annotations

# limit per-process BLAS threads BEFORE numpy import (fair single-thread timing)
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import inspect
import json
import time

import numpy as np

# reuse the EXACT shared config + run functions the paper pipeline uses
from make_3d_results import (METHOD_MAP, METHOD_LABELS, build_env, EXP_BASE)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "metric_3d", "retime_fair")
CSV_PATH = os.path.join(OUT_DIR, "retime_fair.csv")
JSON_PATH = os.path.join(OUT_DIR, "retime_fair.json")

CSV_COLS = ["method", "n_obs", "seed", "total_steps", "warmup_exclude",
            "n_steady", "ctrl_steady_mean_ms", "ctrl_steady_p50_ms",
            "ctrl_steady_p90_ms", "ctrl_all_mean_ms", "n_all", "elapsed_s"]


def _steady_stats(ctrl_ms, warmup_exclude):
    a = np.asarray(ctrl_ms, dtype=np.float64)
    steady = a[warmup_exclude:] if a.size > warmup_exclude else a
    if steady.size == 0:
        nan = float("nan")
        return dict(n_steady=0, mean=nan, p50=nan, p90=nan)
    return dict(n_steady=int(steady.size), mean=float(steady.mean()),
                p50=float(np.percentile(steady, 50)), p90=float(np.percentile(steady, 90)))


def run_episode(label, seed, n_obs, total_steps):
    """Run one episode with early termination disabled so it executes exactly
    `total_steps` control steps; return the per-step control-time array."""
    run_fn, extras = METHOD_MAP[label]
    env = build_env(seed, n_obs)
    exp = dict(EXP_BASE, **extras)
    exp["max_steps"] = int(total_steps)
    exp["break_on_collision"] = False      # do not stop on floor contact
    exp["goal_finish_dist"] = -1.0         # never register goal reach -> full step count
    allowed = set(inspect.signature(run_fn).parameters.keys())
    exp_clean = {k: v for k, v in exp.items() if k in allowed}
    t0 = time.perf_counter()
    result = run_fn(env, **exp_clean)
    elapsed = time.perf_counter() - t0
    return list(result.get("ctrl_times_ms", [])), elapsed


def load_done(json_path):
    if not os.path.isfile(json_path):
        return {}, []
    try:
        blob = json.load(open(json_path))
        rows = blob.get("rows", [])
        done = {(r["method"], r["n_obs"], r["seed"]) for r in rows}
        return done, rows
    except Exception:
        return {}, []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=list(METHOD_LABELS))
    ap.add_argument("--densities", type=int, nargs="+", default=[10, 50, 100, 150, 200, 280])
    ap.add_argument("--seeds", type=int, nargs="+", default=[20, 21, 22, 23, 24])
    ap.add_argument("--total-steps", type=int, default=45)
    ap.add_argument("--warmup-exclude", type=int, default=30,
                    help="steps excluded from the per-step mean (>= ECP calib fill ~27)")
    ap.add_argument("--resume", action="store_true",
                    help="skip (method,n_obs,seed) already present in the json")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    done, rows = (load_done(JSON_PATH) if args.resume else (set(), []))
    if args.resume and rows:
        print(f"[resume] {len(rows)} episodes already done; skipping those", flush=True)

    jobs = [(lab, no, s) for lab in args.methods
            for no in args.densities for s in args.seeds]
    print(f"[retime-fair] {len(jobs)} episodes | total_steps={args.total_steps} "
          f"warmup_exclude={args.warmup_exclude} | methods={args.methods}", flush=True)

    # (re)write CSV header fresh; JSON is the source of truth for --resume
    write_header = not (args.resume and os.path.isfile(CSV_PATH))
    csv_f = open(CSV_PATH, "a", newline="")
    writer = csv.DictWriter(csv_f, fieldnames=CSV_COLS)
    if write_header:
        writer.writeheader(); csv_f.flush()

    for i, (label, n_obs, seed) in enumerate(jobs, 1):
        if (label, n_obs, seed) in done:
            print(f"  [skip {i}/{len(jobs)}] {label} n_obs={n_obs} seed={seed}", flush=True)
            continue
        ctrl_ms, elapsed = run_episode(label, seed, n_obs, args.total_steps)
        st = _steady_stats(ctrl_ms, args.warmup_exclude)
        all_mean = float(np.mean(ctrl_ms)) if ctrl_ms else float("nan")
        row = dict(method=label, n_obs=int(n_obs), seed=int(seed),
                   total_steps=int(args.total_steps), warmup_exclude=int(args.warmup_exclude),
                   n_steady=st["n_steady"], ctrl_steady_mean_ms=st["mean"],
                   ctrl_steady_p50_ms=st["p50"], ctrl_steady_p90_ms=st["p90"],
                   ctrl_all_mean_ms=all_mean, n_all=len(ctrl_ms), elapsed_s=round(elapsed, 1))
        writer.writerow(row); csv_f.flush()
        rows.append({**row, "ctrl_ms": ctrl_ms})
        json.dump({"rows": rows}, open(JSON_PATH, "w"))
        print(f"  [{i:>3}/{len(jobs)}] {label:16s} n_obs={n_obs:>3} seed={seed} "
              f"steady_mean={st['mean']:8.1f}ms (n={st['n_steady']:>2})  "
              f"all_mean={all_mean:8.1f}ms (n={len(ctrl_ms):>2})  wall={elapsed:5.1f}s",
              flush=True)

    csv_f.close()
    print(f"[retime-fair] done -> {CSV_PATH}", flush=True)


if __name__ == "__main__":
    main()
