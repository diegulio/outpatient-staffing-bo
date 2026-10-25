"""Horizontal bar chart of online R^2 per GP kernel.

By default, the values plotted are the recorded 50-step online kernel-sensitivity
results reported in the manuscript (kernel-sensitivity table). Pass ``--csv`` to
plot the output of a fresh run of kernel_analysis.py instead.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os

# Recorded results reported in the manuscript
REPORTED = {
    'Kernel': [
        'Rational Quadratic + White',
        'Constant * Dot Product + White',
        'Matern + White',
        'RBF + White',
        'Dot Product + White',
        'Complex Historical Kernel'
    ],
    'Online_R2': [
        -0.1204,
        -0.1327,
        -0.2010,
        -0.3960,
        -470.7192,
        -729.3276
    ]
}

# Display names for the kernel labels written by kernel_analysis.py
DISPLAY_NAMES = {
    'RationalQuadratic + White': 'Rational Quadratic + White',
    'Constant * DotProduct + White': 'Constant * Dot Product + White',
    'Matern + White': 'Matern + White',
    'RBF + White': 'RBF + White',
    'DotProduct + White': 'Dot Product + White',
    'Complex (Historical Winner)': 'Complex Historical Kernel',
}


def plot_kernel_sensitivity(csv_path=None, output_dir='results'):
    if csv_path is None:
        df = pd.DataFrame(REPORTED)
    else:
        df = pd.read_csv(csv_path)[['Kernel', 'Online_R2']].dropna()
        df['Kernel'] = df['Kernel'].map(lambda k: DISPLAY_NAMES.get(k, k))

    # Best (closest to zero) first
    df = df.sort_values(by='Online_R2', ascending=False)

    plt.figure(figsize=(10, 6))

    # Color by order of magnitude: stable kernels, then the two overfitting ones
    colors = ['#1f77b4' if r2 > -10 else ('#ff7f0e' if r2 > -500 else '#d62728') for r2 in df['Online_R2']]
    bars = plt.barh(df['Kernel'], df['Online_R2'], color=colors, edgecolor='black', alpha=0.8)

    # Symlog scale because of the large differences in magnitude
    plt.xscale('symlog', linthresh=1.0)

    plt.title('Kernel Sensitivity: Online R² Performance (Higher is better)', fontsize=14, pad=15)
    plt.xlabel('Online R² (Symlog Scale)', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    plt.gca().invert_yaxis()

    for bar in bars:
        width = bar.get_width()
        plt.text(width - (abs(width)*0.1 if abs(width) > 1 else 0.1), bar.get_y() + bar.get_height()/2,
                 f'{width:.2f}',
                 ha='right', va='center', fontsize=10, color='white' if abs(width) > 10 else 'black', fontweight='bold')

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'kernel_r2_plot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Kernel sensitivity plot saved to: {plot_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot online R^2 per GP kernel')
    parser.add_argument('--csv', default=None,
                        help='CSV written by kernel_analysis.py (default: values reported in the manuscript)')
    parser.add_argument('--output-dir', default='results')
    args = parser.parse_args()
    plot_kernel_sensitivity(csv_path=args.csv, output_dir=args.output_dir)
