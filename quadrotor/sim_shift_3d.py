from __future__ import annotations

"""
Structural distribution-shift study (3D quadrotor).

Goal (the experiment the abstract/Theorem-3 headline needs):
  * Calibrate the functional-CP envelope ONCE on a CV-leaning obstacle-motion
    mixture (pi_cal), then FREEZE it.
  * Deploy that frozen envelope on increasingly shifted mixtures
    pi(beta) = (1-beta)*pi_cal + beta*pi_hard, where pi_hard is turn / stop-go
    heavy -- exactly the motion regimes the constant-velocity (CV) predictor
    systematically mis-predicts.
  * Show that a STATIC envelope under-covers as beta grows (field coverage drops
    below the 1-alpha target), while the ADAPTIVE envelope (AFCP) recovers
    coverage online.

Nothing about the predictor or the envelope basis changes across beta; the only
knob that moves is the obstacle motion-mode mixture (quad_env `mode_probs`).
This isolates "frozen offline envelope under-covers under shift" as the cause.

This module exposes two building blocks used by run_shift_3d.py:
  * calibrate_envelope_3d(...)   -> fit + freeze the envelope on pi_cal
  * run_deploy_episode_3d(...)   -> one deployment episode under pi(beta) that
                                    reuses the frozen envelope and logs realized
                                    field coverage (Appendix B definition), the
                                    online inflation trace c_t, and safety stats.
"""

import copy
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from quad_env import QuadWorldEnv3D, build_grid_3d, distance_field_points_3d
from cp.functional_cp import get_envelopes_value_and_function
from controllers.func_3d_mpc import FunctionalCPMPC3D

# Reuse the calibration/data helpers and the shared geometry constants so the
# shift study stays bit-for-bit consistent with the main FCP pipeline.
from sim_func_3d import (
    build_training_residuals_from_env_3d,
    stack_pred3d_from_p_dict,
    _get_obs_positions_from_history,
    _min_dist_robot_to_points,
    ROBOT_RAD,
    OBSTACLE_RAD,
    MAX_LINEAR_VEL,
    MAX_ANGULAR_Z,
    MIN_ANGULAR_Z,
    MAX_VZ,
)


# -----------------------------------------------------------------------------
# Mode-mixture knob: [CV, Turn, Wander, Stop-Go]
# -----------------------------------------------------------------------------
# pi_cal is intentionally CV-leaning so the offline envelope is *tight* and there
# is room for the deployment mixture to walk outside it. pi_hard concentrates on
# the regimes a CV predictor cannot follow (sharp turns + stop-and-go), which is
# what actually inflates the residual field -- not raw noise amplitude.
PI_CAL = np.array([0.70, 0.10, 0.10, 0.10], dtype=np.float64)
PI_HARD = np.array([0.00, 0.50, 0.00, 0.50], dtype=np.float64)


def mode_probs_at(beta: float,
                  pi_cal: np.ndarray = PI_CAL,
                  pi_hard: np.ndarray = PI_HARD) -> np.ndarray:
    """Linear interpolation between the calibration and hard mixtures."""
    beta = float(np.clip(beta, 0.0, 1.0))
    mix = (1.0 - beta) * np.asarray(pi_cal) + beta * np.asarray(pi_hard)
    return (mix / mix.sum()).astype(np.float64)


def _close_env(env) -> None:
    """Best-effort PyBullet disconnect to avoid leaking clients across a sweep."""
    try:
        env.close()
    except Exception:
        pass


# -----------------------------------------------------------------------------
# Phase 1: calibrate the envelope once on pi_cal, then freeze it.
# -----------------------------------------------------------------------------
def calibrate_envelope_3d(
    *,
    env_kwargs: Dict,
    nx: int = 40,
    ny: int = 40,
    nz: int = 40,
    time_horizon: int = 12,
    alpha: float = 0.10,
    p_base: int = 5,
    k_mix: int = 5,
    test_size: float = 0.30,
    random_state: int = 0,
    n_jobs: int = 4,
    backend: str = "loky",
    n_calib_samples: int = 120,
    calib_seed: int = 7,
    pi_cal: np.ndarray = PI_CAL,
) -> Dict:
    """Roll out the env under pi_cal, fit the functional-CP envelope, and return a
    frozen bundle (cp_params + grid axes + g_upper_grid + meta).

    The returned `cp_params` are what the deployment controller is seeded with;
    they are NEVER re-fit per beta -- that is the whole point of the study.
    """
    ek = dict(env_kwargs)
    ek["seed"] = calib_seed
    ek["mode_probs"] = list(np.asarray(pi_cal, dtype=float))

    env = QuadWorldEnv3D(**ek)
    try:
        xlim, ylim, zlim = env.xlim, env.ylim, env.zlim
        xs, ys, zs, X, Y, Z = build_grid_3d(xlim, ylim, zlim, nx, ny, nz)

        # Random-action rollouts collect prediction-vs-truth residual fields.
        np.random.seed(calib_seed)
        residuals, _ = build_training_residuals_from_env_3d(
            env,
            n_samples=n_calib_samples,
            X=X, Y=Y, Z=Z,
            time_horizon=time_horizon,
            episode_len=1000,
            group_by_episode=False,
            v_lim=(-MAX_LINEAR_VEL, MAX_LINEAR_VEL),
            yaw_rate_lim=(MIN_ANGULAR_Z, MAX_ANGULAR_Z),
            vz_lim=(-MAX_VZ, MAX_VZ),
        )
    finally:
        _close_env(env)

    g_upper_grid, cp_params = get_envelopes_value_and_function(
        residuals_train=residuals,
        p_base=p_base,
        K=k_mix,
        alpha=alpha,
        test_size=test_size,
        random_state=random_state,
        n_jobs=n_jobs,
        backend=backend,
    )

    return {
        "cp_params": cp_params,
        "g_upper_grid": np.asarray(g_upper_grid, dtype=np.float32),
        "xs": xs, "ys": ys, "zs": zs,
        "X": X, "Y": Y, "Z": Z,
        "nx": nx, "ny": ny, "nz": nz,
        "time_horizon": int(time_horizon),
        "alpha": float(alpha),
        "pi_cal": np.asarray(pi_cal, dtype=float),
        "n_calib_samples": int(n_calib_samples),
        "calib_seed": int(calib_seed),
        "env_kwargs": {k: v for k, v in env_kwargs.items() if k != "seed"},
    }


# -----------------------------------------------------------------------------
# Phase 2: one deployment episode under pi(beta) with the frozen envelope.
# -----------------------------------------------------------------------------
def run_deploy_episode_3d(
    *,
    frozen: Dict,
    env_kwargs: Dict,
    beta: float,
    adaptive: bool,
    seed: int,
    i_cov: int = 1,
    n_skip: int = 4,
    n_paths: int = 2000,
    max_steps: int = 400,
    goal_finish_dist: float = 0.8,
    band_factor: float = 4.0,
    safety_mode: str = "hard",
    w_safety: float = 300.0,
    aci_eta: float = 0.10,
    c_limits: Tuple[float, float] = (0.3, 5.0),
    warmup: Optional[int] = None,
    pi_cal: np.ndarray = PI_CAL,
    pi_hard: np.ndarray = PI_HARD,
) -> Dict:
    """Deploy the frozen envelope on the shifted mixture pi(beta).

    Two controllers share the SAME frozen support envelope U0:
      * static (adaptive=False): plans with U0 unchanged.
      * AFCP   (adaptive=True) : multiplies U0 by a single online scalar c_t,
        updated by adaptive conformal inference (ACI) from realized field
        coverage:  c_{t+1} = clip(c_t + eta * (err_t - alpha)),  err_t = 1[step
        t was uncovered].  c_t > 1 inflates the envelope to restore coverage;
        c_t < 1 tightens it when the frozen envelope over-covers.

    NOTE: this scalar inflation is applied directly here rather than through the
    controller's CPOnlineAdapter3D. The 3D adapter mutates `coeff_upper`, but the
    offline LRW support envelope (_support_envelope_at) is built from the GMM
    means/sigmas/radii and ignores `coeff_upper` entirely, so that path is a
    no-op for an offline-fit envelope. The scalar-c_t update is exactly the
    "single-scalar online update" the method section describes and provably
    drives the realized miscoverage to alpha.

    Coverage (Appendix B): at planning step t, the horizon-i lower distance field
    is  L = max(D_pred_i - c_t * U_i, 0).  The step is "covered" iff the realized
    distance field D_true (i steps later) satisfies D_true(x) >= L(x) for ALL
    cells x; a single violating cell makes the step uncovered.
    """
    cp_params = copy.deepcopy(frozen["cp_params"])  # never mutate the frozen fit
    xs, ys, zs = frozen["xs"], frozen["ys"], frozen["zs"]
    X, Y, Z = frozen["X"], frozen["Y"], frozen["Z"]
    nx, ny, nz = frozen["nx"], frozen["ny"], frozen["nz"]
    time_horizon = frozen["time_horizon"]

    safe_rad = ROBOT_RAD + OBSTACLE_RAD
    band_r = band_factor * safe_rad      # "near obstacle" shell for the diagnostic
    i_cov = int(np.clip(i_cov, 0, time_horizon - 1))

    ek = dict(env_kwargs)
    ek["seed"] = seed
    ek["mode_probs"] = list(mode_probs_at(beta, pi_cal, pi_hard))
    env = QuadWorldEnv3D(**ek)

    # Wall boxes match sim_func_3d so the planner sees the same static geometry.
    margin = 5.0
    cov_min, cov_max = -50.0, 50.0
    x_min, x_max = env.xlim
    y_min, y_max = env.ylim
    z_min, z_max = env.zlim
    wall_boxes = [
        [x_min - margin, x_min, cov_min, cov_max, cov_min, cov_max],
        [x_max, x_max + margin, cov_min, cov_max, cov_min, cov_max],
        [cov_min, cov_max, y_min - margin, y_min, cov_min, cov_max],
        [cov_min, cov_max, y_max, y_max + margin, cov_min, cov_max],
        [cov_min, cov_max, cov_min, cov_max, z_min - margin, z_min],
        [cov_min, cov_max, cov_min, cov_max, z_max, z_max + margin],
    ]

    # The controller's own CPOnlineAdapter3D is disabled (adaptive=False): it
    # mutates `coeff_upper`, which the LRW support envelope ignores. We drive the
    # paper's adaptation directly -- the SAME mechanism as the working 2D
    # CPOnlineAdapter: an ACI-updated scalar that multiplies the support radii
    # (the quantity _support_envelope_at actually consumes).
    ctrl = FunctionalCPMPC3D(
        cp_params=cp_params,
        xs=xs, ys=ys, zs=zs,
        n_steps=time_horizon,
        dt=env.dt,
        n_skip=n_skip,
        robot_rad=ROBOT_RAD,
        obstacle_rad=OBSTACLE_RAD,
        v_lim=(-MAX_LINEAR_VEL, MAX_LINEAR_VEL),
        yaw_rate_lim=(MIN_ANGULAR_Z, MAX_ANGULAR_Z),
        vz_lim=(-MAX_VZ, MAX_VZ),
        n_paths=n_paths,
        seed=0,
        CP=True,
        adaptive=False,
        safety_mode=safety_mode,
        w_safety=w_safety,
    )

    # Frozen base radii per horizon (what the online scalar scales). Mirrors the
    # 2D adapter's snapshot(): q.radii = scale * base_radii.
    base_params = sorted(copy.deepcopy(cp_params), key=lambda p: int(p.t_idx))
    base_radii = [
        (None if p.radii is None else np.asarray(p.radii, dtype=np.float32).copy())
        for p in base_params
    ]

    def _apply_scale(c: float) -> None:
        """Rebuild the controller envelope with all support radii scaled by c
        (ACI scalar). For c==1 this reproduces the frozen envelope exactly."""
        scaled = []
        for p, br in zip(base_params, base_radii):
            q = copy.deepcopy(p)
            if br is not None:
                q.radii = (float(c) * br).astype(np.float32)
            scaled.append(q)
        ctrl.set_cp_params(scaled)

    alpha = float(frozen.get("alpha", 0.10))
    c_t = 1.0                                    # ACI scalar (radii multiplier)
    c_min, c_max = float(c_limits[0]), float(c_limits[1])
    warmup = int(time_horizon if warmup is None else warmup)

    np.random.seed(seed)
    obs = env.reset()
    goal = np.asarray(obs.get("goal_xyz", [0, 0, 0]), dtype=np.float32).reshape(3,)

    # Pending lower-distance fields keyed by the realized frame they predict.
    pending: Dict[int, np.ndarray] = {}

    cov_steps: List[float] = []     # per-step covered indicator (whole field)
    cell_cov: List[float] = []      # per-step fraction of cells covered (smooth)
    band_steps: List[float] = []    # per-step covered indicator within obstacle band
    ct_trace: List[float] = []      # ACI scalar c_t (radii multiplier)
    ueff_trace: List[float] = []    # mean effective envelope at horizon i_cov

    n_collisions = 0
    n_infeasible = 0
    reached_goal = False
    steps = 0
    vx, vy, vz, yaw_rate = 0.0, 0.0, 0.0, 0.0

    for k in range(max_steps):
        robot = np.asarray(obs["robot_xyz"], dtype=np.float32).reshape(3,)
        yaw = float(obs["robot_yaw"])

        if np.linalg.norm(robot - goal) <= goal_finish_dist:
            reached_goal = True
            break

        obs_now = _get_obs_positions_from_history(obs)

        # --- realize coverage for any field that targeted the current frame ---
        if k in pending:
            L = pending.pop(k).astype(np.float32)
            if obs_now.size:
                D_true = distance_field_points_3d(obs_now, X, Y, Z)
                viol = D_true < L                      # cells the envelope failed
                covered = not bool(np.any(viol))
                cov_steps.append(1.0 if covered else 0.0)
                cell_cov.append(float(np.mean(~viol)))
                band = D_true <= band_r
                if bool(np.any(band)):
                    band_steps.append(0.0 if bool(np.any(viol & band)) else 1.0)

                # --- ACI scalar update (paper's online adaptation) -----------
                # err_t = functional violation indicator (1 if the realized field
                # broke the envelope anywhere). c <- clip(c + eta*(err - alpha)).
                if adaptive and k >= warmup:
                    err = 0.0 if covered else 1.0
                    c_new = float(np.clip(c_t + aci_eta * (err - alpha), c_min, c_max))
                    if abs(c_new - c_t) > 1e-9:
                        c_t = c_new
                        _apply_scale(c_t)

        # collision bookkeeping (realized min distance)
        dmin_now = _min_dist_robot_to_points(robot, obs_now) if obs_now.size else float("inf")
        if dmin_now < safe_rad:
            n_collisions += 1

        pred, pred_mask, _ = stack_pred3d_from_p_dict(obs.get("prediction", {}), horizon=time_horizon)

        act, info = ctrl(
            robot_xyz=robot,
            robot_yaw=yaw,
            goal_xyz=goal,
            pred_xyz=pred,
            pred_mask=pred_mask,
            boxes_3d=wall_boxes,
            robot_vel=(vx, vy, vz),
            observed_xyz=obs_now,
        )

        # Effective envelope AFTER any online adaptation this step (U_grid is
        # rebuilt by _apply_scale whenever c_t changes).
        U_grid = ctrl.U_grid  # (H, nz, ny, nx)
        ct_trace.append(float(c_t))
        if U_grid is not None:
            ueff_trace.append(float(np.mean(U_grid[i_cov])))

        # Stash the horizon-i lower field to be checked at frame k+i_cov+1.
        if U_grid is not None and pred.size:
            mask_i = pred_mask[i_cov]
            pts_i = pred[i_cov][mask_i] if mask_i.any() else np.zeros((0, 3), dtype=np.float32)
            D_pred_i = (
                distance_field_points_3d(pts_i, X, Y, Z)
                if pts_i.size else np.full((nz, ny, nx), np.inf, dtype=np.float32)
            )
            U_i = np.maximum(U_grid[i_cov], 0.0)
            L_new = np.maximum(D_pred_i - U_i, 0.0).astype(np.float16)  # cheap to buffer
            pending[k + i_cov + 1] = L_new

        if act is None:
            target = robot.copy()
            vx, vy, vz, yaw_rate = 0.0, 0.0, 0.0, 0.0
        else:
            pos, vel = act
            target = np.asarray(pos, dtype=np.float32).reshape(3,)
            vel = np.asarray(vel, dtype=np.float32).reshape(4,)
            vx, vy, vz, yaw_rate = map(float, vel)

        if not bool(info.get("feasible", False)):
            n_infeasible += 1

        obs = env.step(target, (vx, vy, vz, yaw_rate))
        steps += 1

    _close_env(env)

    def _mean(a: List[float]) -> float:
        return float(np.mean(a)) if a else float("nan")

    # c_t converged value = tail mean (last 20% of the trace), per the feedback's
    # "show AFCP converges" requirement; full mean kept for reference.
    tail = ct_trace[max(0, int(0.8 * len(ct_trace))):] if ct_trace else []

    return {
        "beta": float(beta),
        "adaptive": bool(adaptive),
        "seed": int(seed),
        "coverage": _mean(cov_steps),          # headline (Appendix B field cov.)
        "cell_coverage": _mean(cell_cov),      # smooth diagnostic
        "band_coverage": _mean(band_steps),    # washout-robust diagnostic
        "ct_mean": _mean(ct_trace),
        "ct_tail": _mean(tail),
        "ueff_tail": _mean(ueff_trace[max(0, int(0.8 * len(ueff_trace))):] if ueff_trace else []),
        "n_cov_steps": len(cov_steps),
        "collisions": int(n_collisions),
        "infeasible_steps": int(n_infeasible),
        "collision_rate": (n_collisions / steps) if steps else float("nan"),
        "infeas_rate": (n_infeasible / steps) if steps else float("nan"),
        "reached_goal": int(reached_goal),
        "steps": int(steps),
        "ct_trace": [float(x) for x in ct_trace],
    }
