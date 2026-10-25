"""Online kernel-sensitivity analysis of the Gaussian-process surrogate.

Starting from 50 feasible schedules, each candidate kernel is used in a 50-step
sequential BO loop. At each step the GP predicts the objective of the selected
candidate before it is simulated; the online R^2 between predictions and
observed objectives is reported per kernel.

Note: the kernels below use scikit-learn's default hyperparameter
initializations (e.g. Matern() has nu=1.5), with normalize_y=True and two
optimizer restarts, whereas the optimizer in optimizers/bayesian_optimizer.py
uses Matern(nu=2.5) + WhiteKernel(noise_level=0.1), alpha=1e-2 and ten restarts.
"""

import pandas as pd
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              DotProduct, ConstantKernel, WhiteKernel)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import r2_score

from core.problem import HospitalProblem
from core.simulator import simulator_objective
from core.gurobi_space import sample_feasible_space
from optimizers.bayesian_optimizer import BayesianOptimizer


def run_kernel_sensitivity():
    print("Starting kernel sensitivity analysis (online learning)...")
    problem = HospitalProblem(seed=42)

    print("1. Generating initial feasible sample (N=50)...")
    initial_X, _ = sample_feasible_space(problem, 50)

    if len(initial_X) == 0:
        print("Error: could not generate the initial feasible sample.")
        return

    initial_y = []
    print("2. Evaluating the objective on the initial sample...")
    for x in initial_X:
        val = simulator_objective(problem, x)
        initial_y.append(val)

    X_initial = np.array(initial_X)
    y_initial = np.array(initial_y)

    print("\n3. Testing kernels with online iterative learning...")
    N_TEST = 50

    kernels_to_test = {
        "DotProduct + White": DotProduct() + WhiteKernel(),
        "RBF + White": RBF() + WhiteKernel(),
        "Matern + White": Matern() + WhiteKernel(),
        "RationalQuadratic + White": RationalQuadratic() + WhiteKernel(),
        "Constant * DotProduct + White": ConstantKernel() * DotProduct() + WhiteKernel(),
        "Complex (Historical Winner)": (DotProduct() * RationalQuadratic()) + (DotProduct() * Matern()) + (RBF() * DotProduct()) + (Matern() * RBF()) + WhiteKernel()
    }

    results = []
    bo = BayesianOptimizer(problem, simulator_objective)
    plot_dir = os.path.join("results", "kernel_online")
    os.makedirs(plot_dir, exist_ok=True)

    for name, kernel in kernels_to_test.items():
        print(f"\nEvaluating kernel: {name}")
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, normalize_y=True)

        X_train = X_initial.copy()
        y_train = y_initial.copy()

        y_real = []
        y_pred = []
        stds = []

        start_time = time.time()

        try:
            for i in range(N_TEST):
                # 1. Fit the model on the data accumulated so far (reduced space)
                X_train_red = bo._reduce_X(X_train)
                model.fit(X_train_red, y_train)

                # 2. Select the next candidate with the acquisition function
                x_next = bo.opt_acquisition(X_train, y_train, model)

                if x_next is not None:
                    # 3. Predict BEFORE evaluating
                    x_next_red = bo._reduce_X([x_next])
                    mu, std = model.predict(x_next_red, return_std=True)
                    y_pred.append(mu[0])
                    stds.append(std[0])

                    # 4. Evaluate with the simulator
                    actual_val = simulator_objective(problem, x_next)
                    y_real.append(actual_val)

                    # 5. Online update of the training set
                    X_train = np.vstack([X_train, x_next])
                    y_train = np.append(y_train, actual_val)

                    if (i+1) % 10 == 0:
                        print(f"   Iteration {i+1}/{N_TEST}...")
                else:
                    break

            y_real = np.array(y_real)
            y_pred = np.array(y_pred)
            score = r2_score(y_real, y_pred)

            # Observed vs. predicted (online)
            plt.figure(figsize=(10, 6))
            plt.subplot(211)
            plt.title(f"Kernel: {name} (Online R^2: {score:.4f})", fontsize=12)
            plt.plot(range(len(y_real)), y_real, label='Observed', marker='o', markersize=4)
            plt.plot(range(len(y_pred)), y_pred, label='Predicted', marker='s', markersize=4)
            plt.legend(fontsize=10)
            plt.ylabel("Objective (negative waitlist)", fontsize=10)

            plt.subplot(212)
            plt.plot(range(len(stds)), stds, color='orange', label='Uncertainty', linestyle='--')
            plt.xlabel("Iterations", fontsize=10)
            plt.ylabel("Standard Deviation", fontsize=10)
            plt.legend(fontsize=10)

            plt.tight_layout()
            safe_name = name.replace(" ", "_").replace("*", "x").replace("+", "plus").replace("(", "").replace(")", "")
            plot_path = os.path.join(plot_dir, f"plot_online_{safe_name}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   Saved plot to {plot_path}")

        except Exception as e:
            print(f"Error evaluating {name}: {e}")
            score = np.nan

        execution_time = time.time() - start_time
        results.append({
            "Kernel": name,
            "Online_R2": score,
            "Time_Secs": execution_time
        })
        print(f"  -> Online R^2: {score:.4f} | Total Time: {execution_time:.2f}s")

    print("\n4. Saving results...")
    df_results = pd.DataFrame(results).sort_values(by="Online_R2", ascending=False)
    print("\nFinal results summary:")
    print(df_results.to_string(index=False))

    csv_path = os.path.join("results", "kernel_sensitivity_online.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")


if __name__ == '__main__':
    run_kernel_sensitivity()
