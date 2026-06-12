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

    method_mapping = {
        'acp': 'ACP-MPC',
        'cc': 'CC-MPC',
        'ecp': 'ECP-MPC',
        'fcp': 'Proposed (FCP-MPC)'
    }
    df['method_display'] = df['method'].map(method_mapping).fillna(df['method'])
    # Ordered by development year: ACP, CC, ECP, FCP (ours).
    order = [m for m in ['ACP-MPC', 'CC-MPC', 'ECP-MPC', 'Proposed (FCP-MPC)']
             if m in set(df['method_display'])]

    sns.lineplot(
        data=df,
        x='n_obs',
        y='ctrl_mean_ms',
        hue='method_display',
        style='method_display',
        hue_order=order,
        style_order=order,
        markers=True,
        dashes=False,
        linewidth=2.0,
        markersize=6
    )

    plt.yscale('log')

    plt.xlabel(r"Number of dynamic obstacles $N_{\mathrm{obs}}$", fontsize=11)
    plt.ylabel("Mean control time [ms]", fontsize=11)

    plt.xticks([10, 50, 100, 150, 200, 280], fontsize=10)
    plt.yticks(fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.legend(fontsize=9.5, loc='upper left', frameon=True, framealpha=0.9)
    plt.tight_layout()

    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
    plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    print(f"[saved] scalability figure -> {OUT_PATH}")

if __name__ == "__main__":
    main()
