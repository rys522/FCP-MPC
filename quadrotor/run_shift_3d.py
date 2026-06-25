from __future__ import annotations

"""
Driver for the 3D structural distribution-shift study.

  1. Calibrate the functional-CP envelope ONCE on the CV-leaning mixture (pi_cal)
     and freeze it (cached to disk so reruns/plots reuse the exact same fit).
  2. Sweep beta in [0, 1].  For each beta, deploy that frozen envelope on the
     shifted mixture pi(beta) with two controllers:
        static : adaptive=False  (frozen offline envelope -- expected to break)
        afcp   : adaptive=True   (online functional adaptation -- expected to recover)
     over multiple seeds.
  3. Dump a per-episode CSV and an aggregated JSON consumed by make_table_shift_3d.py.

Success criterion (headline): there exists a beta where static field coverage
falls below the 1-alpha target (ideally <= 0.80) while AFCP holds >= 0.90.

Example:
  python run_shift_3d.py --betas 0,0.25,0.5,0.75,1.0 \
      --seed-from 20 --seed-to 36 --n-obs 50 --max-steps 400
"""

import argparse
import csv
import json
import math
import os
import pickle
import time
from dataclasses import asdict, dataclass, fields
from typing import Dict, List

import numpy as np

from sim_shift_3d import (
    PI_CAL,
    PI_HARD,
    calibrate_envelope_3d,
    run_deploy_episode_3d,
)


@dataclass
class ShiftRow:
    method: str          # "static" | "afcp"
    beta: float
    seed: int
    coverage: float
    cell_coverage: float
    band_coverage: float
    ct_mean: float
    ct_tail: float
    n_cov_steps: int
    collisions: int
    infeasible_steps: int
    collision_rate: float
    infeas_rate: float
    reached_goal: int
    steps: int
    runtime_sec: float


def _safe_mkdir_for_file(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def append_csv(path: str, rows: List[ShiftRow]) -> None:
    if not rows:
        return
    _safe_mkdir_for_file(path)
    write_header = not os.path.exists(path)
    fieldnames = [f.name for f in fields(ShiftRow)]
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def _agg(vals: List[float]) -> Dict[str, float]:
    a = np.asarray([v for v in vals if v is not None and math.isfinite(float(v))], dtype=np.float64)
    if a.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {"mean": float(a.mean()), "std": float(a.std(ddof=0)), "n": int(a.size)}


def build_json(rows: List[ShiftRow], meta: Dict, betas: List[float], seeds: List[int]) -> Dict:
    methods = sorted({r.method for r in rows})
    out: Dict = {"meta": meta, "betas": list(betas), "seeds": list(seeds), "results": {}}
    for m in methods:
        out["results"][m] = {}
        for b in betas:
            sel = [r for r in rows if r.method == m and abs(r.beta - b) < 1e-9]
            out["results"][m][f"{b:g}"] = {
                "coverage": _agg([r.coverage for r in sel]),
                "cell_coverage": _agg([r.cell_coverage for r in sel]),
                "band_coverage": _agg([r.band_coverage for r in sel]),
                "ct_tail": _agg([r.ct_tail for r in sel]),
                "collision_rate": _agg([r.collision_rate for r in sel]),
                "infeas_rate": _agg([r.infeas_rate for r in sel]),
                "reached_goal": _agg([float(r.reached_goal) for r in sel]),
                "steps": _agg([float(r.steps) for r in sel]),
                "per_seed_coverage": [r.coverage for r in sel],
            }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--betas", type=str, default="0,0.25,0.5,0.75,1.0")
    ap.add_argument("--methods", type=str, default="static,afcp")
    ap.add_argument("--seed-from", type=int, default=20)
    ap.add_argument("--seed-to", type=int, default=36, help="inclusive")
    ap.add_argument("--n-obs", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--i-cov", type=int, default=1, help="horizon index for the coverage metric")
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--n-calib-samples", type=int, default=120)
    ap.add_argument("--calib-seed", type=int, default=7)
    ap.add_argument("--n-paths", type=int, default=2000)
    ap.add_argument("--out-dir", type=str, default="metric_3d/shift")
    ap.add_argument("--cache", type=str, default="metric_3d/shift/frozen_envelope.pkl")
    ap.add_argument("--refit", action="store_true", help="ignore the cached frozen envelope and recalibrate")
    args = ap.parse_args()

    betas = [float(x) for x in args.betas.split(",") if x.strip() != ""]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    method_adaptive = {"static": False, "afcp": True}
    for m in methods:
        if m not in method_adaptive:
            raise ValueError(f"Unknown method '{m}'. Known: {list(method_adaptive)}")
    seeds = list(range(args.seed_from, args.seed_to + 1))

    env_base = dict(
        dt=0.1, horizon=20, n_obs=args.n_obs,
        world_bounds_xyz=((-3, 7), (-3, 7), (0, 8)),
        pred_model_noise=0.20, obs_process_noise=0.22, gt_future_noise=0.20,
        mode_switch_p=0.95, mode_min_ttl=1, mode_max_ttl=6,
        turn_rate_std=3.0, stop_go_p=0.6, gui=False,
    )

    n_jobs = max(1, (os.cpu_count() or 4) - 2)

    # ---------------- Phase 1: calibrate (or load) the frozen envelope -------
    if (not args.refit) and os.path.exists(args.cache):
        print(f"[calib] loading frozen envelope from {args.cache}")
        with open(args.cache, "rb") as f:
            frozen = pickle.load(f)
    else:
        print("[calib] fitting frozen envelope on pi_cal (this happens ONCE)...")
        t0 = time.time()
        frozen = calibrate_envelope_3d(
            env_kwargs=env_base,
            time_horizon=12,
            alpha=args.alpha,
            n_calib_samples=args.n_calib_samples,
            calib_seed=args.calib_seed,
            n_jobs=n_jobs,
            pi_cal=PI_CAL,
        )
        _safe_mkdir_for_file(args.cache)
        with open(args.cache, "wb") as f:
            pickle.dump(frozen, f)
        print(f"[calib] done in {time.time()-t0:.1f}s -> cached {args.cache}")

    # ---------------- Phase 2: deploy across beta x method x seed ------------
    csv_path = os.path.join(args.out_dir, "shift_suite.csv")
    json_path = os.path.join(args.out_dir, "shift_suite.json")
    if os.path.exists(csv_path):
        os.remove(csv_path)

    all_rows: List[ShiftRow] = []
    for beta in betas:
        for m in methods:
            for seed in seeds:
                t0 = time.time()
                res = run_deploy_episode_3d(
                    frozen=frozen,
                    env_kwargs=env_base,
                    beta=beta,
                    adaptive=method_adaptive[m],
                    seed=seed,
                    i_cov=args.i_cov,
                    n_paths=args.n_paths,
                    max_steps=args.max_steps,
                    pi_cal=PI_CAL,
                    pi_hard=PI_HARD,
                )
                row = ShiftRow(
                    method=m, beta=beta, seed=seed,
                    coverage=res["coverage"],
                    cell_coverage=res["cell_coverage"],
                    band_coverage=res["band_coverage"],
                    ct_mean=res["ct_mean"], ct_tail=res["ct_tail"],
                    n_cov_steps=res["n_cov_steps"],
                    collisions=res["collisions"],
                    infeasible_steps=res["infeasible_steps"],
                    collision_rate=res["collision_rate"],
                    infeas_rate=res["infeas_rate"],
                    reached_goal=res["reached_goal"],
                    steps=res["steps"],
                    runtime_sec=time.time() - t0,
                )
                all_rows.append(row)
                append_csv(csv_path, [row])
                print(
                    f"[saved] {m:6s} beta={beta:<4g} seed={seed} "
                    f"cov={row.coverage:.3f} band={row.band_coverage:.3f} "
                    f"ct_tail={row.ct_tail:.3f} coll={row.collision_rate:.3f} "
                    f"reached={row.reached_goal} steps={row.steps}"
                )

    meta = {
        "alpha": args.alpha,
        "target_coverage": 1.0 - args.alpha,
        "n_obs": args.n_obs,
        "max_steps": args.max_steps,
        "i_cov": args.i_cov,
        "pi_cal": list(PI_CAL),
        "pi_hard": list(PI_HARD),
        "calib_seed": args.calib_seed,
        "n_calib_samples": args.n_calib_samples,
    }
    dump = build_json(all_rows, meta, betas, seeds)
    _safe_mkdir_for_file(json_path)
    with open(json_path, "w") as f:
        json.dump(dump, f, ensure_ascii=False, indent=2)
    print(f"[done] {len(all_rows)} episodes -> {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
