from __future__ import annotations

"""
LaTeX table for the 3D structural distribution-shift study.

Reads metric_3d/shift/shift_suite.json (produced by run_shift_3d.py) and emits a
coverage-vs-beta table: for each shift level beta, the realized field coverage of
the STATIC (frozen) vs. ADAPTIVE (AFCP) envelope, plus collision rate. The story
the table must tell: static coverage drops below the 1-alpha target as beta grows
while AFCP stays at/above it.

Coverage cells below the target are NOT bolded; AFCP cells that meet the target
are bolded so the recovery reads at a glance.
"""

import argparse
import json
import math
import os
from typing import Dict, List


def read_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def fmt_pm(stat: Dict, nd: int = 3) -> str:
    m, s = stat.get("mean"), stat.get("std")
    if m is None or not math.isfinite(float(m)):
        return "N/A"
    if s is None or not math.isfinite(float(s)):
        return f"{m:.{nd}f}"
    return f"{m:.{nd}f}\\,$\\pm$\\,{s:.{nd}f}"


def bold(s: str) -> str:
    return r"\textbf{" + s + "}"


def build_table(suite: Dict, cov_field: str = "coverage") -> str:
    meta = suite.get("meta", {})
    target = float(meta.get("target_coverage", 0.90))
    betas = suite.get("betas", [])
    results = suite.get("results", {})

    have_static = "static" in results
    have_afcp = "afcp" in results
    n_obs = meta.get("n_obs", "?")

    lines: List[str] = []
    lines += [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        (r"\caption{Structural distribution shift (3D quadrotor, $N_\mathrm{obs}="
         + str(n_obs) + r"$). The envelope is calibrated once on a CV-leaning "
         r"motion mixture and \emph{frozen}; deployment mixes in turning / "
         r"stop-and-go motion with weight $\beta$. We report realized field "
         r"coverage (Appendix~B) at the $1-\alpha=" + f"{target:.2f}"
         + r"$ target. A frozen (static) envelope under-covers as $\beta$ grows, "
         r"while online adaptation (AFCP) restores coverage.}"),
        r"\label{tab:shift_3d}",
        r"\begin{tabular}{lcccc}",
        r"\hline",
        r"$\beta$ & Static cov.\ $\uparrow$ & AFCP cov.\ $\uparrow$ & "
        r"Static coll.\ $\downarrow$ & AFCP coll.\ $\downarrow$ \\",
        r"\hline",
    ]

    for b in betas:
        key = f"{b:g}"
        st = results.get("static", {}).get(key, {}) if have_static else {}
        af = results.get("afcp", {}).get(key, {}) if have_afcp else {}

        st_cov = st.get(cov_field, {})
        af_cov = af.get(cov_field, {})
        st_coll = st.get("collision_rate", {})
        af_coll = af.get("collision_rate", {})

        st_cov_s = fmt_pm(st_cov)
        af_cov_s = fmt_pm(af_cov)

        # Bold whichever coverage meets the target (highlights AFCP recovery and
        # the low-beta in-distribution regime where static is still fine).
        if math.isfinite(float(st_cov.get("mean", float("nan")))) and st_cov["mean"] >= target - 1e-9:
            st_cov_s = bold(st_cov_s)
        if math.isfinite(float(af_cov.get("mean", float("nan")))) and af_cov["mean"] >= target - 1e-9:
            af_cov_s = bold(af_cov_s)

        lines.append(
            f"{b:g} & {st_cov_s} & {af_cov_s} & {fmt_pm(st_coll)} & {fmt_pm(af_coll)} \\\\"
        )

    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, default="metric_3d/shift/shift_suite.json")
    ap.add_argument("--out", type=str, default="tables/table_shift_3d.tex")
    ap.add_argument("--cov-field", type=str, default="coverage",
                    choices=["coverage", "band_coverage", "cell_coverage"],
                    help="which coverage definition to tabulate")
    args = ap.parse_args()

    suite = read_json(args.json)
    tex = build_table(suite, cov_field=args.cov_field)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(tex)
    print(f"[saved] LaTeX table -> {args.out}  (coverage field: {args.cov_field})")
    print(f"[info]  read suite json -> {args.json}")


if __name__ == "__main__":
    main()
