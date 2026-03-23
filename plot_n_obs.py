import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    df = pd.read_csv("metric_3d/ablation_nobs_scratch.csv")

    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(8, 6))

    method_mapping = {
        'cc': 'CC-MPC',
        'ecp': 'ECP-MPC',
        'fcp': 'Proposed (FCP-MPC)'
    }
    df['method_display'] = df['method'].map(method_mapping).fillna(df['method'])

    sns.lineplot(
        data=df, 
        x='n_obs', 
        y='ctrl_mean_ms',  
        hue='method_display', 
        style='method_display',
        markers=True, 
        dashes=False,
        linewidth=2.5,
        markersize=10
    )

    plt.yscale('log')
    
    plt.xlabel("Number of Dynamic Obstacles ($N_{obs}$)", fontsize=14)
    plt.ylabel("Mean Control Time [ms]", fontsize=14)
    plt.title("Scalability Test in Highly Dense 3D Environments", fontsize=16)
    
    #plt.axhline(y=100, color='red', linestyle='--', label='10Hz Real-time Deadline')
    
    plt.xticks([10, 50, 100, 150, 200, 280], fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12, loc='upper left')
    plt.tight_layout()

    plt.savefig("control_time_3d.png", dpi=600)
    plt.show()

if __name__ == "__main__":
    main()
