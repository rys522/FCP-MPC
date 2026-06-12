#!/usr/bin/env python3
"""Composite the per-seed 3D trajectory PNGs into one paper figure.

The 3D trajectory PNGs are produced by:
    python runner_3d.py --methods fcp --seed-from 25 --seed-to 27 \
        --save-traj-img --traj-img-dir <SRC_DIR> --traj-img-max-seeds 3
each saved as ``<SRC_DIR>/fcp_seed<N>.png``. This script crops the debug title
strip off the top of each panel, lays them out 1xK with clean "seed N" titles,
and writes a single figure into the paper folder:

    T_RO2026/traj_3d_seeds.png   (override dirs via FCP_3D_SRC / FCP_PAPER_DIR)
"""
from __future__ import annotations

import os
import re
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.environ.get("FCP_3D_SRC", "/tmp/traj3d_seeds")
PAPER_DIR = os.environ.get("FCP_PAPER_DIR", os.path.join(HERE, "T_RO2026"))
OUT_PATH = os.path.join(PAPER_DIR, "traj_3d_seeds.png")

TOP_CROP = 0.07   # fraction of height to trim off the top (removes the debug title)
MAX_PANELS = 4    # 4 seeds -> 2x2 grid


def _seed_of(path):
    m = re.search(r"seed(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def _tight_crop(img):
    """Drop the baked title strip and trim surrounding white margin so the 3D
    plot fills its panel (matplotlib 3D leaves a lot of whitespace)."""
    h = img.shape[0]
    img = img[int(TOP_CROP * h):, :, :]            # remove the debug title
    rgb = img[..., :3]
    scale = 1.0 if rgb.max() <= 1.0 else 255.0
    mask = np.any(rgb < 0.985 * scale, axis=-1)     # non-(near-white) content
    ys, xs = np.where(mask)
    if ys.size == 0:
        return img
    pad = 4
    y0, y1 = max(0, ys.min() - pad), min(img.shape[0], ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(img.shape[1], xs.max() + pad)
    return img[y0:y1, x0:x1, :]


def main():
    paths = sorted(glob.glob(os.path.join(SRC_DIR, "fcp_seed*.png")), key=_seed_of)
    paths = [p for p in paths if _seed_of(p) >= 0][:MAX_PANELS]
    if not paths:
        raise SystemExit(f"No fcp_seed*.png found in {SRC_DIR}")

    os.makedirs(PAPER_DIR, exist_ok=True)
    imgs = [_tight_crop(mpimg.imread(p)) for p in paths]
    n = len(imgs)

    # 4 seeds -> 2x2 grid; otherwise a single row. Square-ish keeps panels large.
    import math
    ncols = 2 if n >= 4 else n
    nrows = math.ceil(n / ncols)

    ar = np.mean([im.shape[1] / im.shape[0] for im in imgs])  # width/height
    panel_w = 4.2
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(panel_w * ncols, (panel_w / ar) * nrows))
    axes = np.array(axes).reshape(-1)

    for ax, im, p in zip(axes, imgs, paths):
        ax.imshow(im)
        ax.set_title(f"seed {_seed_of(p)}", fontsize=15)
        ax.axis("off")
    for ax in axes[len(imgs):]:  # hide any unused cells
        ax.axis("off")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01, wspace=0.04, hspace=0.08)
    fig.savefig(OUT_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] 3D trajectory figure ({n} seeds) -> {OUT_PATH}")


if __name__ == "__main__":
    main()
