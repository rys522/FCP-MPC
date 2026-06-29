import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# CSV produced by run_3d_obs.sh; output figure goes straight into the paper folder.
CSV_PATH = os.environ.get("FCP_NOBS_CSV", "metric_3d/ablation_nobs.csv")
OUT_PATH = os.environ.get("FCP_NOBS_OUT", "T_RO2026/control_time_3d.png")

def main():
    df = pd.read_csv(CSV_PATH)

    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.edgecolor": "0.3",
        "grid.color": "0.85",
    })
    plt.figure(figsize=(3.6, 2.8))

    # normalize legacy short names; the unified driver already uses display names
    short = {'acp': 'ACP-MPC', 'cc': 'CC-MPC', 'ecp': 'ECP-MPC',
             'fcp': 'FCP-MPC (ours)', 'nocp': 'Nominal MPC'}
    df['method_display'] = df['method'].map(lambda m: short.get(m, m))
    # The 5-method driver emits "FCP-MPC (hard)"/"FCP-MPC (soft)". Show soft as the
    # headline "FCP-MPC (ours)" line (consistent with the traj figure); hard is
    # table-only. Without this remap FCP is dropped from the scalability plot entirely.
    df['method_display'] = df['method_display'].replace({'FCP-MPC (soft)': 'FCP-MPC (ours)'})

    # colours/markers consistent with the trajectory figure (make_figs_3d); ours emphasized.
    # CC and ACP land within ~5% of each other, so they visually merge. To separate them we
    # draw CC as a thick, slightly transparent gray "band" underneath and ACP as a thin,
    # opaque purple line with a higher zorder on top -- the purple reads cleanly through the
    # gray, and the distinct markers (o vs ^) disambiguate at each point.
    # tuple: (color, linestyle, marker, linewidth, markersize, zorder, alpha)
    # Colors matched to the paper's Fig. 10 (sampled from the PDF): CC=magenta, ECP=red,
    # ACP=blue, FCP=teal. With distinct hues, CC and ACP are separable even where their
    # values nearly coincide; FCP (ours, teal) is emphasized.
    style = {
        "CC-MPC":         ("#c84898", "-",  "o", 2.0, 5.0, 3, 1.0),
        "ECP-MPC":        ("#d85838", "-",  "s", 2.0, 5.0, 3, 1.0),
        "ACP-MPC":        ("#2868a8", "-",  "^", 2.0, 5.0, 4, 1.0),
        "Nominal MPC":    ("#8c564b", "--", "D", 1.8, 5.0, 2, 1.0),
        "FCP-MPC (ours)": ("#28a8a8", "-",  "*", 3.0, 9.5, 6, 1.0),
    }
    # Nominal MPC is shown in the table but excluded from the scalability plot.
    order = [m for m in ["CC-MPC", "ECP-MPC", "ACP-MPC", "FCP-MPC (ours)"]
             if m in set(df['method_display'])]
    def _per_step_mean(g):
        # pooled per-step mean: weight each episode's mean by its number of timed control
        # steps (ctrl_n) so the curve is a per-step average, consistent with the table and
        # robust to episodes contributing different step counts (e.g. ECP after warmup).
        d = g.dropna(subset=['ctrl_mean_ms'])
        if 'ctrl_n' in d and d['ctrl_n'].sum() > 0:
            return (d['ctrl_mean_ms'] * d['ctrl_n']).sum() / d['ctrl_n'].sum()
        return d['ctrl_mean_ms'].mean()

    for m in order:
        sub = (df[df['method_display'] == m]
               .groupby('n_obs').apply(_per_step_mean).reset_index(name='ctrl_mean_ms')
               .sort_values('n_obs'))
        c, ls, mk, lw, ms, zo, al = style.get(m, ("0.5", "-", "o", 1.8, 5, 2, 1.0))
        plt.plot(sub['n_obs'], sub['ctrl_mean_ms'], color=c, linestyle=ls,
                 marker=mk, linewidth=lw, markersize=ms, zorder=zo, alpha=al,
                 label=m, markeredgecolor=c)

    plt.yscale('log')

    plt.xlabel(r"Number of dynamic obstacles $N_{\mathrm{obs}}$", fontsize=11)
    plt.ylabel("Mean control time [ms]", fontsize=11)

    plt.xticks([10, 50, 100, 150, 200, 280], fontsize=10)
    plt.yticks(fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=10)
    # legend outside the axes (above the plot), 2 columns
    plt.legend(fontsize=9.5, loc='lower center', bbox_to_anchor=(0.5, 1.02),
               ncol=2, columnspacing=1.3, handletextpad=0.5, frameon=False)
    plt.tight_layout()

    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
    plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    print(f"[saved] scalability figure -> {OUT_PATH}")

if __name__ == "__main__":
    main()
