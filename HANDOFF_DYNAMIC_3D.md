# Handoff — re-run 3D on desktop (overnight OK)

Continue via `git pull` (env `cp`). The 2D experiments + coverage are being re-run on the
laptop; **this handoff is only the 3D re-run** with the corrected FCP envelope.

## What changed (committed)
The FCP envelope was rewritten and is now shared by 2D and 3D:

- **LRW support-function envelope**
  `U_i(x) = ε_i + max_k { μ_k^T φ_i(x) + r_k (φ_i(x)^T Σ_k φ_i(x))^{1/2} }`
  — replaces the old per-coordinate box `mean + φ^T ξ̂ + ε`, which ignored the FPCA basis sign
  and was **not a valid upper bound**. (`cp/functional_cp.py`: `support_envelope_flat`; new
  `CPStepParameters` fields `{means,sigmas,radii,weights,lam}`; `split_alpha=False` so each of
  the two conformal steps uses level α; lower-tail λ quantile.)
- **Horizon-dependent clearance relaxation** re-added to the controllers (`func_cp_mpc.py`,
  `func_3d_mpc.py`; hard filter + soft penalty): the required clearance is relaxed by
  `Δ_t = ½·a_lat·(t·Δt)²` (`a_lat = v_max·max|yaw_rate|`). `Δ_0 = 0`, so the applied/1-step keeps
  full clearance (relaxation acts only for t≥1) and the **i=1 closed-loop guarantee is
  unaffected**. Rationale: rejecting a path because a probability-dependent bound flags it
  unsafe many steps ahead — a step that is re-planned and still evadable — is over-conservative.
  (Avoid the term "ICS", which usually denotes a tightening/avoid notion.)
- **Online AFCP** is now a scalar radius-multiplier `c` via ACI (2D only; 3D is offline, so its
  online adapter is unused).
- **p_base = 5** everywhere (diagnostic: higher p ⇒ *more* conservative support function; ε is
  small and ≈p-independent; 5 is a good middle, consistent with the L3 "5–7 PCs" story).

## Action (desktop, full 17-seed)
1. `git pull`; env `cp`.
2. **Invalidate stale envelope caches first** (old caches hold the box envelope / old p):
   `rm -f sims/cp_cache/*.pkl` and any 3D CP cache `*.pkl`.
3. Re-run the dense suite: `python make_3d_results.py` → regenerates
   `T_RO2026/table_3d_results.tex` with the **new envelope + clearance relaxation + mean±std**, plus
   `metric_3d/results_3d.{csv,json}`. Then `run_sparse_3d.py` + the scalability sweep.
   - `make_3d_results.py::write_latex_table` already emits `mean±std` (collision, infeasible,
     steps); mirror it in the sparse-table writer if that path doesn't share the code.
4. Numbers will shift (valid + larger envelope, p→5, clearance relaxation). Sanity-check: FCP should reach the
   goal; report new collision/infeasible/steps/ctrl. The clearance relaxation should keep hard
   feasible (in 2D it dropped hard infeasible from ~0.85 to ~0.2–0.6).
5. Regenerate envelope-dependent 3D figures (`Func_cp_3d_zoom`, `traj_3d_seeds`,
   `control_time_3d`) and set the two 3D table captions to "mean $\pm$ std over 17 seeds".

## Notes
- **Do NOT touch the commented-out Math-Setup blocks in `main.tex`** (owner is reconciling the
  theory section separately).
- 2D (laptop): coverage/reliability + 2D experiment tables are regenerated with the same
  envelope; the paper's 2D numbers come from there.
