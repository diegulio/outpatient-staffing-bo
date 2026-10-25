"""Full-case benchmark: BO vs. random search, genetic algorithm and simulated annealing.

Reproduces the full-case benchmark of the manuscript (Computational results). All methods start
from the same initial sample of feasible schedules (and the same objective
observations), run for the same number of iterations, and their best schedule is
then re-evaluated over 30 fresh stochastic replications.
"""

import pandas as pd
import time
import os
import argparse
from core.problem import HospitalProblem
from core.baseline import build_historical_schedule
from core.simulator import simulator_objective, evaluate_solution_robust
from core.gurobi_space import sample_feasible_space
from optimizers.bayesian_optimizer import BayesianOptimizer
from optimizers.random_optimizer import RandomOptimizer
from optimizers.genetic_optimizer import GeneticOptimizer
from optimizers.sa_optimizer import SimulatedAnnealingOptimizer


def run_benchmark(only_bo=False, inject_baseline=False, iterations=100,
                  initial_samples=500, seed=42, csv_path="results/benchmark_results.csv"):
    print("Starting benchmark...")
    problem = HospitalProblem(seed=seed)

    print(f"1. Generating the initial feasible sample ({initial_samples} schedules)...")
    initial_X, _ = sample_feasible_space(problem, initial_samples)

    if len(initial_X) == 0:
        print("Error: could not generate the initial feasible sample (check your Gurobi license).")
        return

    # Historical (baseline) schedule of the hospital
    print("-> Evaluating the historical schedule (30 replications)...")
    X_orig = build_historical_schedule(problem)
    mean_baseline, _ = evaluate_solution_robust(problem, X_orig, n_sims=30)
    print(f"Historical schedule (waitlist): {mean_baseline:.2f}")

    if inject_baseline:
        print("-> Adding the historical schedule to the initial sample...")
        initial_X.append(list(X_orig.flatten()))

    initial_y = []
    print("2. Evaluating the objective on the initial sample...")
    for x in initial_X:
        val = simulator_objective(problem, x)
        initial_y.append(val)

    if only_bo:
        optimizers_to_run = [BayesianOptimizer(problem, simulator_objective)]
    else:
        optimizers_to_run = [
            RandomOptimizer(problem, simulator_objective),
            BayesianOptimizer(problem, simulator_objective),
            GeneticOptimizer(problem, simulator_objective),
            SimulatedAnnealingOptimizer(problem, simulator_objective)
        ]

    results = []

    for optimizer in optimizers_to_run:
        print(f"\n3. Running optimizer: {optimizer.name}")
        start_time = time.time()

        best_X, best_y, history_y = optimizer.optimize(initial_X, initial_y, iterations=iterations)

        execution_time = time.time() - start_time
        print(f"{optimizer.name} finished in {execution_time:.2f} seconds.")
        print(f"Best objective (3-replication estimate): {best_y}")

        # Final assessment over 30 fresh replications
        print("Robust evaluation (30 replications)...")
        mean_le, ci = evaluate_solution_robust(problem, best_X, n_sims=30)

        # Reported in objective (negative) form
        print(f"Robust evaluation -> mean: {-mean_le:.2f}, 95% CI: [{-ci[1]:.2f}, {-ci[0]:.2f}]")

        for i, val in enumerate(history_y):
            results.append({
                "Optimizer": optimizer.name,
                "Iteration": i + 1,
                "Best_Objective": val,
                "ExecutionTime_Secs": execution_time,
                "Robust_Mean": -mean_le if i == len(history_y) - 1 else None,
                "Robust_CI_Lower": -ci[1] if i == len(history_y) - 1 else None,
                "Robust_CI_Upper": -ci[0] if i == len(history_y) - 1 else None,
                "Hospital_Baseline": mean_baseline
            })

    print("\n4. Saving results...")
    df_results = pd.DataFrame(results)
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    df_results.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the full-case optimizer benchmark')
    parser.add_argument('--bo', action='store_true', help='Run only Bayesian optimization')
    parser.add_argument('--baseline', action='store_true',
                        help='Add the historical schedule to the initial sample (not used in the manuscript)')
    parser.add_argument('--iterations', type=int, default=100, help='Iterations per optimizer (default: 100)')
    parser.add_argument('--initial-samples', type=int, default=500,
                        help='Size of the shared initial feasible sample (default: 500)')
    parser.add_argument('--seed', type=int, default=42, help='NumPy seed (default: 42)')
    parser.add_argument('--output', default='results/benchmark_results.csv', help='Output CSV path')
    args = parser.parse_args()

    run_benchmark(only_bo=args.bo, inject_baseline=args.baseline, iterations=args.iterations,
                  initial_samples=args.initial_samples, seed=args.seed, csv_path=args.output)
