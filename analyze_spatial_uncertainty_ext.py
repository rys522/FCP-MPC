"""Spatial-uncertainty analysis on external trajectory datasets (SDD / rounD).

Shows that prediction uncertainty is spatially structured and concentrated at
turning / decision regions (roundabouts, intersections, entrances) -- the
justification for a *functional, space-dependent* conformal bound.

No learned predictor needed: a constant-velocity (CV) predictor residual is used as
the uncertainty proxy. Per scene we grid the workspace, compute per-cell mean ADE
error + mean turning + density, correlate them, and overlay the error field on the
scene's reference image.

Datasets
--------
SDD (Stanford Drone Dataset) -- free, pixel coords + reference.jpg per video:
    data_dir/annotations/<scene>/video<k>/annotations.txt
    data_dir/annotations/<scene>/video<k>/reference.jpg
  `deathCircle` is a roundabout; `hyang`/`gates` are intersections.
  Download: https://cvgl.stanford.edu/projects/uav_data/  (annotations + reference imgs)

rounD (levelXdata) -- LICENSE-GATED, you must download it yourself (meters):
    data_dir/<rec>_tracks.csv, <rec>_recordingMeta.csv, <rec>_background.png
  Roundabout vehicles/VRUs; strong turning signal on the circle.

Usage
-----
  conda run -n cp python analyze_spatial_uncertainty_ext.py \
      --dataset sdd --data-dir /path/to/SDD --scene deathCircle --video video0
  conda run -n cp python analyze_spatial_uncertainty_ext.py \
      --dataset round --data-dir /path/to/rounD/data --recording 00
"""
import os, glob, argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# prediction convention (pedestrian standard: 0.4 s step, 8 obs / 12 pred)
OBS_LEN, PRED_LEN = 8, 12


# ---------------------------------------------------------------- shared analysis
def total_heading_change(traj):
    d = np.diff(traj, axis=0); n = np.linalg.norm(d, axis=1)
    d = d[n > 1e-6]
    if len(d) < 2:
        return 0.0
    ang = np.arctan2(d[:, 1], d[:, 0])
    return float(np.sum(np.abs((np.diff(ang) + np.pi) % (2 * np.pi) - np.pi)))


def cv_windows(traj):
    """Slide an 8/12 window; CV-predict from the obs velocity. Yields (pos, ade, turn)."""
    T = len(traj)
    for t in range(OBS_LEN - 1, T - 2):
        obs = traj[t - OBS_LEN + 1:t + 1]
        fut = traj[t + 1:t + 1 + PRED_LEN]
        k = len(fut)
        if k < 2:
            continue
        vel = (obs[-1] - obs[0]) / (OBS_LEN - 1)              # mean obs velocity
        pred = obs[-1] + vel * np.arange(1, k + 1)[:, None]   # (k, 2)
        ade = float(np.mean(np.linalg.norm(pred - fut[:k], axis=1)))
        yield obs[-1], ade, total_heading_change(fut[:k])


def grid_mean(pos, val, x0, y0, cell, nx, ny, min_count):
    ix = np.clip(((pos[:, 0] - x0) / cell).astype(int), 0, nx - 1)
    iy = np.clip(((pos[:, 1] - y0) / cell).astype(int), 0, ny - 1)
    s = np.zeros((ny, nx)); c = np.zeros((ny, nx))
    np.add.at(s, (iy, ix), val); np.add.at(c, (iy, ix), 1)
    g = np.full((ny, nx), np.nan); m = c >= min_count; g[m] = s[m] / c[m]
    return g, c


def pearson(a, b):
    m = np.isfinite(a) & np.isfinite(b); a, b = a[m], b[m]
    return (float(np.corrcoef(a, b)[0, 1]) if len(a) >= 5 else float("nan")), int(len(a))


# ---------------------------------------------------------------- dataset loaders
def load_sdd(data_dir, scene, video):
    """Return (trajectories[list of (T,2) px], background RGB, coord->px = identity)."""
    base = os.path.join(data_dir, "annotations", scene, video)
    ann = os.path.join(base, "annotations.txt")
    if not os.path.isfile(ann):
        raise FileNotFoundError(f"SDD annotations not found: {ann}")
    # cols: trackId xmin ymin xmax ymax frame lost occluded generated label
    cols = np.genfromtxt(ann, dtype=None, encoding="utf-8")
    rows = {}
    for r in cols:
        tid, xmin, ymin, xmax, ymax, frame, lost = (int(r[0]), float(r[1]), float(r[2]),
                                                    float(r[3]), float(r[4]), int(r[5]), int(r[6]))
        if lost:
            continue
        rows.setdefault(tid, []).append((frame, 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)))
    trajs = []
    for tid, pts in rows.items():
        pts.sort()
        pts = np.array([[p[1], p[2]] for p in pts], np.float32)
        # SDD ~30 fps; downsample to ~2.5 fps (every 12th frame) for the 0.4s convention
        pts = pts[::12]
        if len(pts) >= OBS_LEN + 2:
            trajs.append(pts)
    bg = cv2.cvtColor(cv2.imread(os.path.join(base, "reference.jpg")), cv2.COLOR_BGR2RGB)
    return trajs, bg, (lambda P: P)  # already pixel coords


def load_round(data_dir, rec):
    """Return (trajectories[list of (T,2) m], background RGB, coord->px mapping)."""
    import csv
    tracks = os.path.join(data_dir, f"{rec}_tracks.csv")
    meta = os.path.join(data_dir, f"{rec}_recordingMeta.csv")
    bgp = os.path.join(data_dir, f"{rec}_background.png")
    if not os.path.isfile(tracks):
        raise FileNotFoundError(f"rounD tracks not found: {tracks}")
    rows = {}
    with open(tracks) as f:
        for r in csv.DictReader(f):
            tid = int(r["trackId"])
            rows.setdefault(tid, []).append((int(r["frame"]), float(r["xCenter"]), float(r["yCenter"])))
    trajs = []
    for tid, pts in rows.items():
        pts.sort()
        pts = np.array([[p[1], p[2]] for p in pts], np.float32)
        pts = pts[::25]  # rounD ~25 fps -> ~1 Hz; coarse but fine for spatial stats
        if len(pts) >= OBS_LEN + 2:
            trajs.append(pts)
    # meters -> pixels via orthoPxToMeter (image y is flipped)
    px2m = None
    with open(meta) as f:
        m = next(csv.DictReader(f))
        px2m = float(m.get("orthoPxToMeter", "1") or 1.0)
    bg = cv2.cvtColor(cv2.imread(bgp), cv2.COLOR_BGR2RGB) if os.path.isfile(bgp) else None

    def to_px(P):
        # rounD: x_px = x_m / px2m ; y_px = -y_m / px2m  (UTM-like, y up)
        out = np.empty_like(P)
        out[:, 0] = P[:, 0] / px2m
        out[:, 1] = -P[:, 1] / px2m
        return out
    return trajs, bg, to_px


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["sdd", "round"], required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--scene", default="deathCircle")     # SDD
    ap.add_argument("--video", default="video0")          # SDD
    ap.add_argument("--recording", default="00")          # rounD
    ap.add_argument("--cell", type=float, default=None, help="grid cell (px for sdd, m for round)")
    ap.add_argument("--min-count", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.55)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.dataset == "sdd":
        trajs, bg, to_px = load_sdd(args.data_dir, args.scene, args.video)
        cell = args.cell or 40.0   # pixels
        tag = f"SDD/{args.scene}/{args.video}"
    else:
        trajs, bg, to_px = load_round(args.data_dir, args.recording)
        cell = args.cell or 2.0    # metres
        tag = f"rounD/{args.recording}"

    pos, err, turn = [], [], []
    for tr in trajs:
        for p, e, t in cv_windows(tr):
            pos.append(p); err.append(e); turn.append(t)
    pos, err, turn = np.array(pos), np.array(err), np.array(turn)
    if len(pos) < 50:
        raise RuntimeError(f"too few windows ({len(pos)}) -- check data/downsampling")

    x0, y0 = pos[:, 0].min(), pos[:, 1].min()
    nx = int((pos[:, 0].max() - x0) / cell) + 1
    ny = int((pos[:, 1].max() - y0) / cell) + 1
    err_g, cnt = grid_mean(pos, err, x0, y0, cell, nx, ny, args.min_count)
    turn_g, _ = grid_mean(pos, turn, x0, y0, cell, nx, ny, args.min_count)

    r_turn, n1 = pearson(err_g.ravel(), turn_g.ravel())
    r_dens, _ = pearson(err_g.ravel(), np.log10(np.where(cnt > 0, cnt, np.nan)).ravel())
    print(f"=== {tag} ===  windows={len(pos)}  cells={n1}")
    print(f"  corr(error, turning) = {r_turn:+.3f}")
    print(f"  corr(error, density) = {r_dens:+.3f}")

    # ---- overlay on reference image (if available) ----
    fig, ax = plt.subplots(figsize=(8, 7))
    if bg is not None:
        h, w = bg.shape[:2]
        ax.imshow(bg)
        # back-project: for each pixel, find its world cell. We have coord->px; invert by
        # sampling cell centres -> px and scattering (robust for both identity and affine).
        ys, xs = np.mgrid[0:ny, 0:nx]
        wx = x0 + (xs + 0.5) * cell; wy = y0 + (ys + 0.5) * cell
        P = np.stack([wx.ravel(), wy.ravel()], 1)
        Ppx = to_px(P)
        vmin, vmax = np.nanpercentile(err, 10), np.nanpercentile(err, 95)
        norm = plt.Normalize(vmin, vmax)
        ev = err_g.ravel()
        good = np.isfinite(ev)
        ax.scatter(Ppx[good, 0], Ppx[good, 1], c=ev[good], cmap="turbo", norm=norm,
                   s=(cell * (h / max(1, ny * cell)))**2 * 0.0 + 120, marker="s",
                   alpha=args.alpha, edgecolors="none")
        ax.set_xlim(0, w); ax.set_ylim(h, 0)
        sm = cm.ScalarMappable(norm=norm, cmap="turbo")
        plt.colorbar(sm, ax=ax, fraction=0.046, label="CV-predictor ADE")
    else:
        im = ax.imshow(err_g, origin="lower", extent=[x0, x0+nx*cell, y0, y0+ny*cell],
                       cmap="turbo")
        plt.colorbar(im, ax=ax, label="CV-predictor ADE")
    ax.set_title(f"{tag}: prediction uncertainty (CV ADE)\n"
                 f"corr(error,turning)={r_turn:+.2f}  corr(error,density)={r_dens:+.2f}",
                 fontsize=10)
    ax.axis("off")
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   f"spatial_uncertainty_{args.dataset}.png")
    fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
