"""Bar chart of the 30-replication assessment of each method's best schedule."""

import pandas as pd
import matplotlib.pyplot as plt
import os


def generate_robustness_plot(csv_path="results/benchmark_results.csv", output_dir="results"):
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # Keep only the rows holding the final robust evaluation
    robust_df = df.dropna(subset=['Robust_Mean']).copy()

    if robust_df.empty:
        print("Error: No robust evaluation data found in the CSV.")
        return

    # Flip the sign to report positive waitlists
    robust_df['Waitlist_Mean'] = -robust_df['Robust_Mean']
    robust_df['Waitlist_CI_Lower'] = -robust_df['Robust_CI_Upper']
    robust_df['Waitlist_CI_Upper'] = -robust_df['Robust_CI_Lower']

    algorithms = robust_df['Optimizer'].tolist()
    means = robust_df['Waitlist_Mean'].tolist()
    ci_lower = robust_df['Waitlist_CI_Lower'].tolist()
    ci_upper = robust_df['Waitlist_CI_Upper'].tolist()

    # The historical schedule is shown by its 30-replication mean only
    # (its confidence interval is not stored in the CSV).
    baseline_val = robust_df['Hospital_Baseline'].iloc[0]
    algorithms = ['Historical (Baseline)'] + algorithms
    means = [baseline_val] + means

    # Asymmetric error bars
    yerr_lower = [0] + [m - l for m, l in zip(means[1:], ci_lower)]
    yerr_upper = [0] + [u - m for m, u in zip(means[1:], ci_upper)]
    yerr = [yerr_lower, yerr_upper]

    plt.figure(figsize=(10, 6))

    colors = ['#cccccc', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bars = plt.bar(algorithms, means, yerr=yerr, capsize=8, color=colors, alpha=0.8, edgecolor='black')

    plt.title('Robustness of Optimized Schedules (30 Replications)', fontsize=14, pad=15)
    plt.ylabel('Expected Waitlist Queue (Patients)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 100,
                 f'{height:,.0f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.xticks(rotation=15, ha='right')

    # Truncated y-axis to make the differences visible
    plt.ylim(24000, 30000)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'robustness_plot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Robustness plot saved to: {plot_path}")


if __name__ == '__main__':
    generate_robustness_plot()
