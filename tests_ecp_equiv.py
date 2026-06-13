#!/usr/bin/env python3
"""Equivalence harness for the ECP controller (original vs vectorized).

Drives EgocentricCPMPC3D with a deterministic synthetic obstacle stream in the
canonical loop order (update_observations -> __call__ -> update_predictions) and
records, per step, the quantile sum, the chosen next position, and feasibility.

Run BEFORE vectorizing to capture the reference, then AFTER to compare:
    conda run -n cp python tests_ecp_equiv.py            # prints + saves /tmp/ecp_run.pkl
    conda run -n cp python tests_ecp_equiv.py --compare  # compares to /tmp/ecp_ref.pkl
"""
from __future__ import annotations
import argparse, pickle
import numpy as np

from controllers.ecp_mpc_3d import EgocentricCPMPC3D, pred_dict_to_stacked


def run(n_steps=4, n_skip=2, n_obs=6, K=16, seed=0):
    rng = np.random.default_rng(seed)
    ctrl = EgocentricCPMPC3D(n_steps=n_steps, n_skip=n_skip, dt=0.4,
                             calibration_set_size=5, miscoverage_level=0.1,
                             step_size=0.05)
    H = n_steps
    pids = list(range(n_obs))
    pos = rng.uniform(-3, 3, size=(n_obs, 3)).astype(np.float32)
    vel = rng.uniform(-0.2, 0.2, size=(n_obs, 3)).astype(np.float32)
    robot = np.array([0, 0, 1], dtype=np.float32)
    yaw = 0.0
    goal = np.array([3, 3, 2], dtype=np.float32)

    hist_len = 5
    history = {pid: [pos[j].copy() for _ in range(hist_len)] for j, pid in enumerate(pids)}

    records = []
    for k in range(K):
        pos = pos + vel * 0.4
        for j, pid in enumerate(pids):
            history[pid].append(pos[j].copy())
            history[pid] = history[pid][-hist_len:]
        obs_history = {pid: np.asarray(history[pid], dtype=np.float32) for pid in pids}
        prediction = {pid: np.asarray([pos[j] + vel[j] * 0.4 * (h + 1) for h in range(H)],
                                      dtype=np.float32) for j, pid in enumerate(pids)}
        pred_xyz, pred_mask, _ = pred_dict_to_stacked(prediction, horizon=H)

        ctrl.update_observations(obs_history)
        act, info = ctrl(robot_xyz=robot, robot_yaw=yaw, goal_xyz=goal,
                         pred_xyz=pred_xyz, pred_mask=pred_mask, boxes_3d=[])
        ctrl.update_predictions(prediction)

        q = info.get("quantiles")
        qsum = float(np.nansum(np.asarray(q))) if q is not None else None
        nextpos = None if act is None else np.asarray(act[0], dtype=np.float64).round(8)
        records.append({"qsum": qsum, "next": None if nextpos is None else nextpos.tolist(),
                        "feasible": bool(info.get("feasible"))})
        if nextpos is not None:
            robot = nextpos.astype(np.float32)
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--save-ref", action="store_true")
    args = ap.parse_args()

    recs = run()
    for i, r in enumerate(recs):
        print(i, "qsum=%.6f" % (r["qsum"] or 0.0), "feas=%d" % r["feasible"],
              "next=", r["next"])

    pickle.dump(recs, open("/tmp/ecp_run.pkl", "wb"))
    if args.save_ref:
        pickle.dump(recs, open("/tmp/ecp_ref.pkl", "wb"))
        print("[saved reference -> /tmp/ecp_ref.pkl]")
    if args.compare:
        ref = pickle.load(open("/tmp/ecp_ref.pkl", "rb"))
        ok = True
        for i, (a, b) in enumerate(zip(ref, recs)):
            dq = abs((a["qsum"] or 0) - (b["qsum"] or 0))
            dn = 0.0
            if a["next"] and b["next"]:
                dn = float(np.max(np.abs(np.array(a["next"]) - np.array(b["next"]))))
            if dq > 1e-4 or dn > 1e-4 or a["feasible"] != b["feasible"]:
                ok = False
                print(f"  MISMATCH step {i}: dq={dq:.2e} dnext={dn:.2e} "
                      f"feas {a['feasible']}->{b['feasible']}")
        print("EQUIVALENT" if ok else "NOT EQUIVALENT")


if __name__ == "__main__":
    main()
