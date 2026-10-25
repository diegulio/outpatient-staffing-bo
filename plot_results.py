"""Convergence plot and summary table from results/benchmark_results.csv."""

import pandas as pd
import matplotlib.pyplot as plt
import os


def generate_paper_plots(csv_path="results/benchmark_results.csv", output_dir="results"):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run benchmark.py first.")
        return

    df = pd.read_csv(csv_path)

    # 1. Convergence plot (iteration vs. best-so-far waitlist)
    plt.figure(figsize=(10, 6))

    optimizers = df['Optimizer'].unique()

    # Line styles that remain distinguishable in black and white
    linestyles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'D']

    for i, opt in enumerate(optimizers):
        opt_data = df[df['Optimizer'] == opt]
        plt.plot(opt_data['Iteration'],
                 # Flip the sign to plot the (positive) waitlist
                 -opt_data['Best_Objective'],
                 label=opt,
                 linestyle=linestyles[i % len(linestyles)],
                 marker=markers[i % len(markers)],
                 markersize=6,
                 linewidth=2)

    # Historical-schedule reference line
    if 'Hospital_Baseline' in df.columns:
        baseline_val = df['Hospital_Baseline'].iloc[0]
        plt.axhline(y=baseline_val, color='black', linestyle='--', linewidth=2, label='Hospital Baseline (Historical)')

    plt.title('Convergence of Optimization Algorithms', fontsize=14, pad=15)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Best Objective Value (Expected Waitlist)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10, loc='upper right', bbox_to_anchor=(1, 1))

    convergence_path = os.path.join(output_dir, 'convergence_plot.png')
    plt.savefig(convergence_path, dpi=300, bbox_inches='tight')
    print(f"Convergence plot saved to: {convergence_path}")
    plt.close()

    # 2. Summary table
    print("\n--- Summary table (Markdown) ---")
    summary = []

    for opt in optimizers:
        opt_data = df[df['Optimizer'] == opt]
        best_val = -opt_data['Best_Objective'].max()  # best (lowest) waitlist
        time_secs = opt_data['ExecutionTime_Secs'].iloc[0]

        robust_mean = "N/A"
        robust_ci = "N/A"
        if 'Robust_Mean' in df.columns:
            robust_rows = opt_data.dropna(subset=['Robust_Mean'])
            if not robust_rows.empty:
                r_row = robust_rows.iloc[-1]
                # Flip the sign (and the interval bounds) to report positive waitlists
                r_mean = -r_row['Robust_Mean']
                r_ci_lower = -r_row['Robust_CI_Upper']
                r_ci_upper = -r_row['Robust_CI_Lower']

                robust_mean = f"{r_mean:.2f}"
                robust_ci = f"[{r_ci_lower:.2f}, {r_ci_upper:.2f}]"

        summary.append({
            "Algorithm": opt,
            "Best Objective (Min Waitlist)": best_val,
            "Execution Time (s)": round(time_secs, 2),
            "Robust Mean (30 sims)": robust_mean,
            "95% CI": robust_ci
        })

    df_summary = pd.DataFrame(summary)
    print(df_summary.to_markdown(index=False))

    print("\n--- Summary table (LaTeX) ---")
    print(df_summary.to_latex(index=False, caption="Algorithm Comparison Results", label="tab:results"))


if __name__ == '__main__':
    os.makedirs("results", exist_ok=True)
    generate_paper_plots()
