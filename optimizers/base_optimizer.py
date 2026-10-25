class BaseOptimizer:
    def __init__(self, problem, objective_func):
        self.problem = problem
        self.objective_func = objective_func
        self.name = "BaseOptimizer"

    def optimize(self, initial_X, initial_y, iterations=50):
        """
        Must be implemented by every algorithm.

        All optimizers MAXIMIZE the objective (the negative expected waitlist).

        Returns:
        - X_best: best schedule found (flat list of length i*j*k).
        - y_best: objective value of X_best.
        - history_y: best-so-far objective value after each iteration.
        """
        raise NotImplementedError("optimize must be implemented by each optimizer")
