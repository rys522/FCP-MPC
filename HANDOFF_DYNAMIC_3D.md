# Handoff — Dynamic-env 3D experiments (run on desktop)

Local machine is slow for the 3D run (ECP is ~1 s/step at n_obs=280). Run the
commands below on the desktop (conda env `cp`), then commit the regenerated
tables/figures. Everything except the final 3D table/figure regeneration is
already done and committed.

## Goal
Produce the **dynamic-environment** 3D tables & figures: obstacles now include
goal-directed *crossing pedestrians* (traverse the workspace) instead of only a
random-walk-with-wall-bounce gas. This dynamic setting is the paper's main topic.

## Commands to run on desktop (env `cp`)

### 3D — dense table + trajectory figure + scalability (one-shot, 5 methods, 17 seeds)
```bash
conda run -n cp python make_3d_results.py \
  --seeds 20 21 22 23 24 30 31 32 33 34 35 36 37 38 39 40 41 \
  --traj-seeds 20 22 30
```
Outputs: `T_RO2026/table_3d_results.tex` (5 rows: ACP/CC/ECP/FCP-hard/FCP-soft),
`T_RO2026/traj_3d_seeds.png`, `T_RO2026/control_time_3d.png`, `metric_3d/*`.

### 3D — sparse table (N_obs=50)
```bash
conda run -n cp python run_sparse_3d.py
```
Outputs: `T_RO2026/table_3d_sparse.tex`.

### Sanity check first (fast)
```bash
conda run -n cp python make_3d_results.py --smoke
```

### 2D — already done & committed; only if you want to re-run
```bash
for ds in zara1 zara2 eth hotel univ; do
  for c in cc acp-mpc ecp-mpc fcp-hard-adaptive fcp-soft-adaptive \
           fcp-hard-nonadaptive fcp-soft-nonadaptive; do
    conda run -n cp python runner_2d.py --dataset $ds --controller $c
  done
done
conda run -n cp python make_table_2d.py
conda run -n cp python make_figs_2d.py
```

## What this session changed

### Already committed (3246739, 6d56b3e)
- **2D**: all CP baselines (`cc`, `acp_mpc`, `ecp_mpc`) switched from the fixed
  3x3 (v,w) grid to **MPPI-style sampling** via the shared
  `controllers/utils.py::sample_random_paths`. Conditions unified: `n_paths=1200`,
  `n_skip=2`, horizon=12 (= prediction horizon) for all; FCP keeps a 17-step
  planning horizon (extra steps are masked goal-seeking lookahead). 2D tables/figs
  regenerated.
- **3D ECP** (`ecp_mpc_3d.py`): grid → MPPI sampling
  (`controllers/utils.py::sample_random_paths_3d`), `n_paths=2000`.

### In the commit that ships with THIS file (code only)
- **`quad_env.py`** — goal-directed *crossing pedestrian* obstacle mode:
  `ObstacleAgent.goal/speed`, `_new_crossing_goal()`, a goal-directed branch in
  `_step_obstacles_logic()`, and env params
  `goal_directed_frac` / `goal_speed_range` / `goal_reach_thresh` / `goal_steer_gain`.
  Default `goal_directed_frac=0.0` preserves the old behavior.
- **`make_figs_3d.py`** `ENV_KWARGS`: `goal_directed_frac=0.5` (paper's dynamic
  setting — half crossing pedestrians, half random-walk).
- **`make_3d_results.py`** — refactored to a **5-method one-shot**
  (ACP / CC / ECP / FCP-hard / FCP-soft over the same 17 seeds) → the 5-row paper
  table + figure + scalability in a single run. This **removes the brittle
  `scan_traj_seeds` → `run_soft3d_full` → `build_final_3d` chain** that caused the
  seed-mismatch/duplication bug (baselines were aggregated over a different seed
  set than FCP). The trajectory figure maps FCP-soft to the "FCP-MPC (ours)"
  headline; FCP-hard is table-only.
- **`sim_cp_3d.py`** — CC-3D now accepts `n_paths`/`seed` and runs at **2000**
  (was the controller default 512), so CC matches ACP/ECP/FCP.

## Key settings / decisions
- **3D action search uniform across all methods**: `n_paths=2000`, `n_skip=4`,
  horizon=12 (from `make_figs_3d.py::EXP_BASE` + the per-method run wrappers).
- **FCP-3D is non-adaptive** (offline-calibrated field only). Online adaptation is
  optional and the 2D ablation already covers that story; skipped in 3D for speed.
- **Dynamic env**: `goal_directed_frac=0.5`. If metrics look too easy/hard, tune
  `goal_directed_frac` and `goal_speed_range` in `make_figs_3d.py::ENV_KWARGS`.

## After running on desktop — verify, then commit
1. `table_3d_results.tex` is 5 rows; baselines (ACP/ECP) should crash/timeout more
   under crossing traffic while FCP (hard/soft) still reaches the goal.
2. `traj_3d_seeds.png` panels = seeds [20,22,30]; if a panel looks empty (baseline
   died at step ~1), re-pick with `--traj-seeds <a> <b> <c>` choosing seeds where
   FCP reaches AND baselines visibly travel before crashing.
3. Commit `T_RO2026/table_3d_results.tex`, `table_3d_sparse.tex`,
   `traj_3d_seeds.png`, `control_time_3d.png` (metric_3d/* is gitignored).

## Notes
- ECP-3D control time is high (~1 s/step at n_obs=280) — inherent to its per-path
  online calibration, not a misconfiguration (sparse N_obs=50 → ~18 ms).
- Experiment data (`metric*/`, `*.pkl/json/csv/npy`) is gitignored; only code and
  `T_RO2026/` assets (`*.tex`, `*.png`) are tracked.
