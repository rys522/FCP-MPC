#!/usr/bin/env python3
from __future__ import annotations

import os
import json
import math
import numpy as np

# ----------------------------
# Config
# ----------------------------
METRIC_DIR = "metric"          # runner_2d.py가 json 저장한 폴더
OUT_DIR = "tables"
OUT_TEX = "table_2d_results.tex"

DATASETS = ["zara1", "zara2", "eth", "univ"]
CONTROLLERS = ["cc", "ecp-mpc", "acp-mpc", "fcp-mpc"]

METHOD_NAME = {
    "cc": r"CC-MPC",
    "ecp-mpc": r"ECP-MPC",
    "acp-mpc": r"ACP-MPC",
    "fcp-mpc": r"FCP-MPC (ours)",
}

MAX_N_STEPS = {
    "zara1": 100,
    "zara2": 100,
    "eth": 100,
    "univ": 300,
}

TIMEOUT_EPS = 0.5


# ----------------------------
# Helpers
# ----------------------------
def safe_float(x):
    try:
        f = float(x)
        if not math.isfinite(f):
            return float("nan")
        return f
    except Exception:
        return float("nan")


def mean_or_nan(arr):
    if arr is None or len(arr) == 0:
        return float("nan")
    a = np.asarray(arr, dtype=np.float64)
    if a.size == 0:
        return float("nan")
    return float(np.nanmean(a))


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ctrl_time_mean(d):
    """
    timing_ctrl_ms: [ {mean, p50, ...}, None, ... ]
    → scene별 mean을 다시 평균
    """
    lst = d.get("timing_ctrl_ms", [])
    vals = []
    for item in lst:
        if isinstance(item, dict) and "mean" in item:
            vals.append(safe_float(item["mean"]))
    return mean_or_nan(vals)


def format_steps(val, max_steps):
    if not math.isfinite(val):
        return "N/A"
    if val >= max_steps - TIMEOUT_EPS:
        return "timeout"
    return f"{val:.1f}"


def fmt(x, nd=3):
    if not math.isfinite(x):
        return "N/A"
    return f"{x:.{nd}f}"


def bold(s):
    return r"\textbf{" + s + "}"


# ----------------------------
# Main logic
# ----------------------------
def build_table():
    os.makedirs(OUT_DIR, exist_ok=True)

    lines = []
    lines.append(r"\begin{tabular}{llcccc}")
    lines.append(r"\hline")
    lines.append(
        r"Dataset & Method & Collision rate $\downarrow$ & "
        r"Infeasible rate $\downarrow$ & "
        r"Steps to goal $\downarrow$ & "
        r"Ctrl.\ time (ms) $\downarrow$ \\"
    )
    lines.append(r"\hline")

    for dataset in DATASETS:
        results = {}

        # ---- load all controllers for this dataset ----
        for ctrl in CONTROLLERS:
            path = os.path.join(METRIC_DIR, f"{dataset}_{ctrl}.json")
            if not os.path.isfile(path):
                results[ctrl] = None
                continue

            d = read_json(path)
            results[ctrl] = {
                "collision": mean_or_nan(d.get("collision", [])),
                "infeasible": mean_or_nan(d.get("infeasible", [])),
                "steps": mean_or_nan(d.get("time", [])),
                "ctrl_ms": ctrl_time_mean(d),
            }

        # ---- mins for bold ----
        def min_val(key, ignore_timeout=False):
            vals = []
            for c in CONTROLLERS:
                r = results.get(c)
                if r is None:
                    continue
                v = r[key]
                if not math.isfinite(v):
                    continue
                if ignore_timeout and v >= MAX_N_STEPS[dataset] - TIMEOUT_EPS:
                    continue
                vals.append(v)
            return min(vals) if vals else float("nan")

        min_collision = min_val("collision")
        min_infeasible = min_val("infeasible")
        min_steps = min_val("steps", ignore_timeout=True)
        min_ctrl = min_val("ctrl_ms")

        # ---- rows ----
        for j, ctrl in enumerate(CONTROLLERS):
            r = results.get(ctrl)
            name = METHOD_NAME[ctrl]

            if r is None:
                lines.append(
                    f"{dataset} & {name} & N/A & N/A & N/A & N/A \\\\"
                )
                continue

            c, i, s, t = r["collision"], r["infeasible"], r["steps"], r["ctrl_ms"]

            c_str = fmt(c, 3)
            i_str = fmt(i, 3)
            s_str = format_steps(s, MAX_N_STEPS[dataset])
            t_str = fmt(t, 2)

            if math.isfinite(c) and abs(c - min_collision) < 1e-9:
                c_str = bold(c_str)
            if math.isfinite(i) and abs(i - min_infeasible) < 1e-9:
                i_str = bold(i_str)
            if s_str not in ["N/A", "timeout"] and math.isfinite(s) and abs(s - min_steps) < 1e-9:
                s_str = bold(s_str)
            if math.isfinite(t) and abs(t - min_ctrl) < 1e-9:
                t_str = bold(t_str)

            dataset_str = dataset if j == 0 else ""
            lines.append(
                f"{dataset_str} & {name} & {c_str} & {i_str} & {s_str} & {t_str} \\\\"
            )

        lines.append(r"\hline")

    lines.append(r"\end{tabular}")

    out_path = os.path.join(OUT_DIR, OUT_TEX)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[saved] LaTeX table -> {out_path}")


if __name__ == "__main__":
    build_table()