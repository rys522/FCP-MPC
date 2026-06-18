"""Adaptive Conformal Prediction MPC for the 3D quadrotor setting.

A direct 3D port of the 2D ACP-MPC (Dixit et al., 2023): a per-horizon scalar
conformal radius is adapted online by ACI (Gibbs--Cand\`es) and used to inflate the
predicted obstacles inside a sampled, holonomic MPC. The ACI module itself is
dimension-agnostic and reused verbatim from the 2D controller.
"""
from __future__ import annotations

import math
import numpy as np

from controllers.acp_mpc import AdaptiveConformalPredictionModule
from controllers.utils import sample_goal_anchored_paths_3d


class AdaptiveCPMPC3D:
    def __init__(
        self,
        *,
        n_steps: int,
        dt: float,
        v_lim,
        vz_lim,
        yaw_rate_lim,
        n_skip: int,
        robot_rad: float,
        obstacle_rad: float,
        n_paths: int,
        seed: int = 0,
        target_miscoverage_level: float = 0.10,
        step_size: float = 0.05,
        **_ignore,
    ):
        self.n_steps = int(n_steps)
        self.dt = float(dt)
        self.n_skip = max(1, int(n_skip))
        self.vmin, self.vmax = map(float, v_lim)
        self.vzmin, self.vzmax = map(float, vz_lim)
        self.robot_rad = float(robot_rad)
        self.obstacle_rad = float(obstacle_rad)
        self.safe_rad = self.robot_rad + self.obstacle_rad
        self.n_paths = int(n_paths)
        self.rng = np.random.default_rng(int(seed))

        # Initial/fallback radius (before online data accrues). Kept modest so the
        # planner can move and let ACI adapt; an over-large value blocks the dense
        # 3D workspace outright. Grows gently with the horizon.
        max_interval_lengths = np.minimum(0.6, 0.1 + 0.05 * np.arange(self.n_steps))
        self.cp_module = AdaptiveConformalPredictionModule(
            target_miscoverage_level=float(target_miscoverage_level),
            step_size=float(step_size),
            n_scores=self.n_steps,
            max_interval_lengths=max_interval_lengths,
            sample_size=12,
            offline_calibration_set={i: [] for i in range(self.n_steps)},
        )
        self.w_inter, self.w_term, self.w_ctrl = 1.0, 10.0, 0.001

    # ------------------------------------------------------------------
    # Online ACI update from realized vs predicted obstacle positions
    # ------------------------------------------------------------------
    def update_cp(self, obs_hist, pred):
        return self.cp_module.update(obs_hist or {}, pred or {})

    # ------------------------------------------------------------------
    # Holonomic velocity rollouts (the quadrotor can move in any direction)
    # ------------------------------------------------------------------
    def _sample_paths(self, robot_xyz, goal_xyz):
        # Shared goal-anchored sampling-based planner (identical across all 3D
        # controllers) so the comparison isolates the conformal method, not the planner.
        return sample_goal_anchored_paths_3d(
            robot_xyz, goal_xyz, n_steps=self.n_steps, n_paths=self.n_paths, dt=self.dt,
            vmax=self.vmax, vzmin=self.vzmin, vzmax=self.vzmax, rng=self.rng)

    def __call__(self, *, robot_xyz, goal_xyz, pred_xyz, pred_mask, intervals, **_ignore):
        robot_xyz = np.asarray(robot_xyz, np.float32).reshape(3)
        goal = np.asarray(goal_xyz, np.float32).reshape(3)
        paths, vel = self._sample_paths(robot_xyz, goal)
        P, T1, _ = paths.shape
        T = T1 - 1

        # per-horizon conformal radii (online-adapted); pad/truncate to T
        r = np.asarray(intervals, np.float32) if intervals is not None and np.size(intervals) else np.zeros(T, np.float32)
        if r.size < T:
            pad = r[-1] if r.size else 0.0
            r = np.concatenate([r, np.full(T - r.size, pad, np.float32)])

        alive = np.ones(P, dtype=bool)
        if pred_xyz is not None and np.size(pred_xyz):
            pred = np.asarray(pred_xyz, np.float32)          # (H, M, 3)
            mask = np.asarray(pred_mask, bool) if pred_mask is not None else np.ones(pred.shape[:2], bool)
            Tu = min(T, pred.shape[0])
            if pred.shape[1] > 0 and Tu > 0:
                # Vectorized over the horizon (exact same per-element norm/min as the
                # per-step loop): masked obstacles -> inf so they never set the min.
                X = paths[:, 1:Tu + 1, :]                              # (P, Tu, 3)
                d = np.linalg.norm(X[:, :, None, :] - pred[None, :Tu, :, :], axis=-1)  # (P,Tu,M)
                d = np.where(mask[None, :Tu, :], d, np.inf)
                min_d = d.min(axis=-1)                                 # (P, Tu)
                alive = np.all(min_d >= (self.safe_rad + r[:Tu])[None, :], axis=1)

        if not np.any(alive):
            return None, {"feasible": False, "cost": None}

        sp, sv = paths[alive], vel[alive]
        cost = (self.w_inter * np.sum((sp[:, :-1, :] - goal) ** 2, axis=(-2, -1))
                + self.w_term * np.sum((sp[:, -1, :] - goal) ** 2, axis=-1)
                + self.w_ctrl * np.sum(sv ** 2, axis=(-2, -1)))
        b = int(np.argmin(cost))
        target = sp[b, 1, :].astype(np.float32)
        v0 = sv[b, 0, :]
        vel4 = np.array([v0[0], v0[1], v0[2], 0.0], np.float32)  # holonomic; zero yaw rate
        return (target, vel4), {"feasible": True, "cost": float(cost[b])}
