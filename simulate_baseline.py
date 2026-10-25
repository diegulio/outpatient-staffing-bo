"""Evaluates the historical schedule with the queue model (no Gurobi required)."""

from core.problem import HospitalProblem
from core.baseline import build_historical_schedule
from core.simulator import evaluate_solution_robust, N_MONTHS


def main():
    problem = HospitalProblem(seed=42)
    X_original = build_historical_schedule(problem)

    print("==================================================")
    print(" HISTORICAL SCHEDULE (NO OPTIMIZATION)")
    print("==================================================")
    print(f"- Horizon: {N_MONTHS} months")
    print(f"- Initial total queue (month 0): {sum(problem.queue_length)}")
    print("- Evaluating over 30 stochastic replications...")

    mean_le, ci = evaluate_solution_robust(problem, X_original, n_sims=30)

    print("--------------------------------------------------")
    print(f"-> Expected waitlist (month {N_MONTHS}): {mean_le:.2f} patients")
    print(f"-> 95% confidence interval: [{ci[0]:.2f}, {ci[1]:.2f}]")
    print("==================================================")


if __name__ == "__main__":
    main()
