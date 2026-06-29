#!/usr/bin/env python3
"""Table III: full-episode vs feasible-step collision rate for FCP-MPC (hard).

Reproduces the per-step feasibility logging the paper's Table III reports. run_fcp_mpc
returns per-step `collisions` and `infeasible` arrays; we split the collision rate by
whether the certified hard filter returned a plan at that step (feasible) vs issued a
brake-to-hover fallback (infeasible).

Run at --evade-relax-scale 1.0 to reproduce the paper's stale Table III (validation), and
at 0.5 for the new default (consistent with the regenerated Table I).
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os, argparse, pickle
import numpy as np
from sims.sim_func_cp import run_fcp_mpc

EVAL_TASK = {
    "eth":   {"init_robot_pose": np.array([5.0, 1.0, np.pi / 2.0]), "goal_pos": np.array([3.0, 10.0])},
    "hotel": {"init_robot_pose": np.array([-1.5, 0.0, -np.pi / 2.0]), "goal_pos": np.array([2.0, -6.0])},
    "univ":  {"init_robot_pose": np.array([3.5, 2.0, np.pi / 4.0]), "goal_pos": np.array([11.5, 8.5])},
    "zara1": {"init_robot_pose": np.array([12.0, 5.0, np.pi]),      "goal_pos": np.array([3.0, 6.0])},
    "zara2": {"init_robot_pose": np.array([1.0, 6.0, 0.0]),         "goal_pos": np.array([14.0, 5.0])},
}
SCENARIOS = {"zara1": [100, 200, 300], "zara2": [100, 200, 300], "eth": [732, 339, 653],
             "hotel": [1001, 1245, 1582], "univ": [40, 140, 240]}
INIT_FRAME = {"zara1": 0, "zara2": 1, "eth": 78, "hotel": 0, "univ": 0}
MAX_N_STEPS = {"zara1": 100, "zara2": 100, "eth": 100, "hotel": 100, "univ": 300}
ORDER = ["eth", "hotel", "univ", "zara1", "zara2"]


def collision_split(metric):
    """Return (full, feasA, feasB), aggregated as mean-over-scenes to match Table I exactly
    (runner_2d stores a per-scene collision ratio; make_table_2d means over scenes).

    feasA: per-scene collision rate over steps with infeasible[k]==0 (collision at the feasible
           step), then averaged over scenes. This is the column the paper reports.
    feasB: same but conditioning on the PRIOR step being feasible (infeasible[k-1]==0); a brake
           at k-1 lets a moving pedestrian hit the stopped robot at k. Kept as a diagnostic.
    """
    full_s, fA_s, fB_s = [], [], []
    for sm in metric.values():
        c = np.asarray(sm["collisions"], dtype=float)
        inf = np.asarray(sm["infeasible"], dtype=float)
        full_s.append(float(c.mean()))
        feas = inf == 0
        if feas.sum() > 0:
            fA_s.append(float(c[feas].mean()))
        if c.size > 1 and (inf[:-1] == 0).sum() > 0:
            fB_s.append(float(c[1:][inf[:-1] == 0].mean()))
    full = float(np.mean(full_s)) if full_s else 0.0
    feasA = float(np.mean(fA_s)) if fA_s else 0.0
    feasB = float(np.mean(fB_s)) if fB_s else 0.0
    return full, feasA, feasB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evade-relax-scale", type=float, default=0.5)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--adaptive", type=int, default=0)
    args = ap.parse_args()

    rr, orad, dt = 0.4, 1.0 / np.sqrt(2.0), 0.4
    bounds = dict(max_linear_x=0.8, min_linear_x=-0.8, max_angular_z=0.7, min_angular_z=-0.7)

    print(f"FCP-hard  evade_relax_scale={args.evade_relax_scale}  adaptive={bool(args.adaptive)}  "
          f"seeds={args.seeds}")
    print(f"{'scene':6s} | {'full-episode':12s} | {'feasible(A:this)':16s} | {'feasible(B:prior)':17s}")
    print("-" * 60)
    rows = {}
    for ds in ORDER:
        preds = pickle.load(open(os.path.join(os.path.dirname(__file__), "predictions", f"{ds}.pkl"), "rb"))
        task = EVAL_TASK[ds]
        fulls, fAs, fBs = [], [], []
        for sd in args.seeds:
            np.random.seed(sd)
            metric, _ = run_fcp_mpc(
                dataset=ds, scenarios=SCENARIOS[ds], predictions=preds, dt=dt,
                init_frame=INIT_FRAME[ds], visualize=False, asset_dir=None, robot_img=None,
                max_n_steps=MAX_N_STEPS[ds], robot_rad=rr, obstacle_rad=orad,
                init_robot_pose=task["init_robot_pose"], goal_pos=task["goal_pos"],
                target_miscoverage_level=0.1, step_size=10.0, adaptive=bool(args.adaptive),
                safety_mode="hard", evade_relax_scale=args.evade_relax_scale, **bounds,
            )
            f, a, b = collision_split(metric)
            fulls.append(f); fAs.append(a); fBs.append(b)
        rows[ds] = (np.mean(fulls), np.mean(fAs), np.mean(fBs))
        print(f"{ds:6s} | {np.mean(fulls):.3f}        | {np.mean(fAs):.3f}            | {np.mean(fBs):.3f}")

    # LaTeX (uses feasible-A = collision rate on steps where the filter returned a plan)
    tag = f"de{int(round(args.evade_relax_scale*100)):03d}"
    out = os.path.join(os.path.dirname(__file__), "tables", f"table_2d_feasible_{tag}.tex")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    def cell(v, best):
        s = f"{v:.3f}"
        return f"\\textbf{{{s}}}" if best else s
    with open(out, "w") as f:
        f.write("\\begin{tabular}{lcc}\n\\hline\n")
        f.write("Scene & Coll. (full-episode) $\\downarrow$ & Coll. (feasible-step) $\\downarrow$ \\\\\n\\hline\n")
        for ds in ORDER:
            full, fa, _ = rows[ds]
            # bold the better (lower, since lower-is-better) of the two collision measures
            f.write(f"\\texttt{{{ds}}} & {cell(full, full < fa)} & {cell(fa, fa <= full)} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
