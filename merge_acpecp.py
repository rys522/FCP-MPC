#!/usr/bin/env python3
"""Merge the shared-sampler ACP/ECP re-run into outcomes_3d_fixed.pkl, replacing the
old ACP/ECP entries and keeping CC/FCP (unchanged — they already use the shared
goal-anchored planner)."""
import pickle, os
HERE = os.path.dirname(os.path.abspath(__file__))
base = pickle.load(open(os.path.join(HERE, "outcomes_3d_fixed.pkl"), "rb"))
new = pickle.load(open(os.path.join(HERE, "outcomes_acpecp.pkl"), "rb"))
keep = [r for r in base["main"] if r["label"] not in ("ACP-MPC", "ECP-MPC")]
base["main"] = keep + new["main"]
pickle.dump(base, open(os.path.join(HERE, "outcomes_3d_fixed.pkl"), "wb"))
n = {}
for r in base["main"]:
    n[r["label"]] = n.get(r["label"], 0) + 1
print("merged; per-label episode counts:", n)
