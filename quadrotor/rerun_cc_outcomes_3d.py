#!/usr/bin/env python3
"""Re-run the 3D CC-MPC OUTCOME metrics with the Lekeufack (2024) online conformal-
decision adaptation ENABLED (sim_cp_3d now calls update_conformal_var each step,
matching the 2D deployment). Outcomes are deterministic given (seed, config), so this
runs the 17 paper seeds x {50, 280} in parallel; per-step TIMING is NOT taken from here
(that comes from the contention-free retime_fair pass).

Writes the new CC records to outcomes_cc_adaptive.pkl and prints a before/after vs the
existing (fixed-lambda) CC rows in outcomes_3d_fixed.pkl. Does NOT overwrite the master
pkl -- a separate merge step does that once the change is confirmed.

  python quadrotor/rerun_cc_outcomes_3d.py
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import pickle
import numpy as np

from make_3d_results import EXP_BASE, run_jobs

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MASTER = os.path.join(ROOT, "outcomes_3d_fixed.pkl")
OUT = os.path.join(ROOT, "outcomes_cc_adaptive.pkl")

SEEDS = [20, 21, 22, 23, 24, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41]
DENSITIES = [50, 280]
MAX_STEPS = 250


def agg(records, N):
    rs = [r["metrics"] for r in records if r["n_obs"] == N]
    if not rs:
        return None
    reach = [m["reached_goal"] for m in rs]
    steps_reached = [m["steps"] for m in rs if m["reached_goal"]]
    return dict(
        n=len(rs), reached=float(np.mean(reach)),
        coll=float(np.mean([m["collision_rate"] for m in rs])),
        coll_std=float(np.std([m["collision_rate"] for m in rs])),
        infeas=float(np.mean([m["infeas_rate"] for m in rs])),
        steps=(float(np.mean(steps_reached)) if steps_reached else float("nan")),
    )


def main():
    eb = dict(EXP_BASE)
    eb["max_steps"] = MAX_STEPS
    eb["n_jobs"] = 1
    jobs = [("CC-MPC", s, N, eb, False) for N in DENSITIES for s in SEEDS]
    workers = max(1, (os.cpu_count() or 2) - 2)
    print(f"[cc-rerun] {len(jobs)} CC episodes (adaptation ON) on {workers} workers", flush=True)
    results = run_jobs(jobs, workers)
    records = [{"label": "CC-MPC", "seed": r["seed"], "n_obs": r["n_obs"],
                "metrics": r["metrics"]} for r in results]
    pickle.dump({"main": records, "seeds": SEEDS, "n_obs": DENSITIES}, open(OUT, "wb"))
    print(f"[cc-rerun] saved {len(records)} records -> {OUT}\n")

    old = pickle.load(open(MASTER, "rb"))["main"]
    old_cc = [r for r in old if r["label"] == "CC-MPC"]
    print(f"{'='*72}\nCC-MPC 3D outcomes: BEFORE (fixed lambda=1)  vs  AFTER (adaptation ON)\n{'='*72}")
    for N in DENSITIES:
        b, a = agg(old_cc, N), agg(records, N)
        print(f"\n N_obs={N}")
        print(f"   {'':10s} {'reached':>8s} {'coll':>14s} {'infeas':>8s} {'steps(reached)':>14s}")
        print(f"   {'BEFORE':10s} {b['reached']*100:7.0f}% {b['coll']:8.3f}±{b['coll_std']:.3f} "
              f"{b['infeas']:8.3f} {b['steps']:14.1f}")
        print(f"   {'AFTER':10s} {a['reached']*100:7.0f}% {a['coll']:8.3f}±{a['coll_std']:.3f} "
              f"{a['infeas']:8.3f} {a['steps']:14.1f}")


if __name__ == "__main__":
    main()
