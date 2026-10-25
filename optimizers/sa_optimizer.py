import numpy as np
import random
import math
from .base_optimizer import BaseOptimizer
from core.gurobi_space import repair_solution


class SimulatedAnnealingOptimizer(BaseOptimizer):
    """Simulated annealing that evaluates five repaired neighbors per iteration
    and applies the Metropolis acceptance rule to the best of them."""

    def __init__(self, problem, objective_func, initial_temp=1000.0, cooling_rate=0.95):
        super().__init__(problem, objective_func)
        self.name = "Simulated Annealing"
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate

    def optimize(self, initial_X, initial_y, iterations=50):
        # Start from the best schedule of the initial sample
        best_idx = np.argmax(initial_y)
        current_x = list(initial_X[best_idx])
        current_y = initial_y[best_idx]

        best_overall_x = current_x
        best_overall_y = current_y
        history_y = []

        T = self.initial_temp

        print(f"Starting Simulated Annealing with T={T}, alpha={self.cooling_rate}")

        for i in range(iterations):
            # Explore 5 neighbors and keep the best one
            best_neighbor_x = None
            best_neighbor_y = -float('inf')

            for _ in range(5):
                neighbor_x = self._get_neighbor(current_x)
                repaired_x = repair_solution(self.problem, neighbor_x)
                if repaired_x is not None:
                    repaired_x = list(repaired_x)
                    val = self.objective_func(self.problem, repaired_x)
                    if val > best_neighbor_y:
                        best_neighbor_y = val
                        best_neighbor_x = repaired_x

            if best_neighbor_x is not None:
                delta_y = best_neighbor_y - current_y

                if delta_y > 0:
                    # Improving move: always accept
                    current_x = best_neighbor_x
                    current_y = best_neighbor_y
                    if current_y > best_overall_y:
                        best_overall_y = current_y
                        best_overall_x = current_x
                else:
                    # Worsening move: accept with probability exp(delta / T)
                    prob = math.exp(delta_y / T) if T > 1e-5 else 0
                    if random.random() < prob:
                        current_x = best_neighbor_x
                        current_y = best_neighbor_y

            T *= self.cooling_rate

            history_y.append(best_overall_y)
            print(f"> SA Iter {i+1}: current = {current_y:.3f}, best so far = {best_overall_y:.3f}, T = {T:.2f}")

        return best_overall_x, best_overall_y, history_y

    def _get_neighbor(self, x):
        """
        Neighborhood operators, inspired by Meersman and Maenhout (2022):
        - capacity: increase one variable and decrease another by the same amount
        - interchange: swap two variables
        - shift: move a few hours from one variable to a nearby one
        """
        neighbor = list(x)
        operator_type = random.choice(["capacity", "interchange", "shift"])

        if operator_type == "capacity":
            idx1 = random.randint(0, len(neighbor) - 1)
            idx2 = random.randint(0, len(neighbor) - 1)
            delta = random.randint(1, 10)
            neighbor[idx1] = max(0, neighbor[idx1] + delta)
            neighbor[idx2] = max(0, neighbor[idx2] - delta)

        elif operator_type == "interchange":
            idx1 = random.randint(0, len(neighbor) - 1)
            idx2 = random.randint(0, len(neighbor) - 1)
            neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]

        elif operator_type == "shift":
            idx1 = random.randint(0, len(neighbor) - 1)
            idx2 = (idx1 + random.randint(1, 5)) % len(neighbor)
            delta = random.randint(1, 5)
            neighbor[idx1] = max(0, neighbor[idx1] - delta)
            neighbor[idx2] = max(0, neighbor[idx2] + delta)

        return neighbor
