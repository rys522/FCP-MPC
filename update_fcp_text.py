#!/usr/bin/env python3
"""Update the FCP prose numbers in §VI(3D) of main.tex after Table III aggregation.

Usage:
    python update_fcp_text.py [--csv metric_3d/table3d.csv] [--tex T_RO2026/main.tex]
"""
from __future__ import annotations
import argparse, csv, math, re

def safe_float(x):
    try: return float(x)
    except: return math.nan

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="metric_3d/table3d.csv")
    ap.add_argument("--tex", default="T_RO2026/main.tex")
    args = ap.parse_args()

    rows = []
    with open(args.csv) as f:
        for r in csv.DictReader(f):
            if r["method"] == "fcp":
                rows.append(r)

    if len(rows) < 5:
        print(f"ERROR: only {len(rows)} FCP rows, need 5.")
        return

    coll   = [safe_float(r["collision_rate"]) for r in rows]
    infeas = [safe_float(r["infeas_rate"]) for r in rows]
    steps_raw = [safe_float(r["steps"]) for r in rows]
    max_steps = 250  # same as --max-steps
    steps_for_avg = [s for s in steps_raw if math.isfinite(s) and s < max_steps]

    mean_coll   = sum(c for c in coll   if math.isfinite(c)) / len(coll)
    mean_infeas = sum(i for i in infeas if math.isfinite(i)) / len(infeas)
    mean_steps  = sum(steps_for_avg) / len(steps_for_avg) if steps_for_avg else float('nan')

    print(f"FCP 5-seed results:")
    print(f"  collision_rate = {mean_coll:.3f}")
    print(f"  infeas_rate    = {mean_infeas:.3f}")
    print(f"  steps          = {mean_steps:.1f}  (from {len(steps_for_avg)}/5 seeds that reached goal)")

    # Patch the body text  (lines ~1366-1368 in main.tex)
    OLD_PAT = re.compile(
        r'It reduces the collision rate to \$[0-9.]+\$ and the infeasible rate to \$[0-9.]+\$,\s*'
        r'while reaching the goal in \$[0-9.]+\$ steps on average'
    )
    new_text = (
        f"It reduces the collision rate to ${mean_coll:.3f}$ and the infeasible rate to ${mean_infeas:.3f}$,\n"
        f"while reaching the goal in ${mean_steps:.1f}$ steps on average"
    )

    with open(args.tex) as f:
        src = f.read()

    m = OLD_PAT.search(src)
    if not m:
        print("WARNING: could not find FCP prose pattern in main.tex — check manually.")
        print("  Expected pattern: 'It reduces the collision rate to $X.XXX$ ...'")
        return

    new_src = src[:m.start()] + new_text + src[m.end():]
    with open(args.tex, "w") as f:
        f.write(new_src)
    print(f"[patched] FCP prose in {args.tex}")

if __name__ == "__main__":
    main()
