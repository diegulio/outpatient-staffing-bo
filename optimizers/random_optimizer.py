import numpy as np
from .base_optimizer import BaseOptimizer


class RandomOptimizer(BaseOptimizer):
    """Random search: evaluates one new feasible schedule per iteration."""

    def __init__(self, problem, objective_func):
        super().__init__(problem, objective_func)
        self.name = "Random Search (Baseline)"

    def optimize(self, initial_X, initial_y, iterations=50):
        from core.gurobi_space import sample_feasible_space

        X = np.array(initial_X)
        y = np.array(initial_y)

        history_y = []
        best_current_y = np.max(y)

        for i in range(iterations):
            Xsamples, _ = sample_feasible_space(self.problem, 1)
            if len(Xsamples) == 0:
                print("No more feasible solutions found at iteration", i)
                break

            x_next = Xsamples[0]
            actual = self.objective_func(self.problem, x_next)

            X = np.concatenate((X, [x_next]))
            y = np.concatenate((y, [actual]))

            if actual > best_current_y:
                best_current_y = actual

            history_y.append(best_current_y)
            print(f"> Random Iter {i+1}: observed = {actual:.3f}, best so far = {best_current_y:.3f}")

        ix = np.argmax(y)
        return X[ix], y[ix], history_y
