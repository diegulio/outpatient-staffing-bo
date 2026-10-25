import numpy as np
import random
from .base_optimizer import BaseOptimizer
from core.gurobi_space import repair_solution


class GeneticOptimizer(BaseOptimizer):
    """Genetic algorithm with elitism, tournament selection, single-point
    crossover, mutation, and an L1 repair MILP to restore feasibility."""

    def __init__(self, problem, objective_func, population_size=20, mutation_rate=0.1):
        super().__init__(problem, objective_func)
        self.name = "Genetic Algorithm"
        self.pop_size = population_size
        self.mutation_rate = mutation_rate

    def optimize(self, initial_X, initial_y, iterations=50):
        # The initial population is the pop_size best schedules of the initial sample
        sorted_indices = np.argsort(initial_y)[::-1]
        X_pop = [list(initial_X[i]) for i in sorted_indices[:self.pop_size]]
        y_pop = [initial_y[i] for i in sorted_indices[:self.pop_size]]

        history_y = []
        best_overall_x = X_pop[0]
        best_overall_y = y_pop[0]

        print(f"Starting GA with population size {self.pop_size}")

        for gen in range(iterations):
            new_population = []

            # Elitism: keep the best individual found so far
            new_population.append(best_overall_x)

            while len(new_population) < self.pop_size:
                # Tournament selection
                p1 = self._tournament_selection(X_pop, y_pop)
                p2 = self._tournament_selection(X_pop, y_pop)

                child = self._crossover(p1, p2)

                if random.random() < self.mutation_rate:
                    child = self._mutate(child)

                # Repair (Gurobi guarantees feasibility)
                repaired_child = repair_solution(self.problem, child)
                if repaired_child is not None:
                    new_population.append(list(repaired_child))

            # Evaluate the whole population
            X_pop = new_population
            y_pop = []
            for indiv in X_pop:
                val = self.objective_func(self.problem, indiv)
                y_pop.append(val)

            current_best_idx = np.argmax(y_pop)
            if y_pop[current_best_idx] > best_overall_y:
                best_overall_y = y_pop[current_best_idx]
                best_overall_x = X_pop[current_best_idx]

            history_y.append(best_overall_y)
            print(f"> GA Gen {gen+1}: generation best = {y_pop[current_best_idx]:.3f}, best so far = {best_overall_y:.3f}")

        return best_overall_x, best_overall_y, history_y

    def _tournament_selection(self, X, y, k=3):
        k = min(k, len(X))
        selected_indices = random.sample(range(len(X)), k)
        best_idx = selected_indices[0]
        for idx in selected_indices[1:]:
            if y[idx] > y[best_idx]:
                best_idx = idx
        return X[best_idx]

    def _crossover(self, p1, p2):
        # Single-point crossover
        point = random.randint(0, len(p1) - 1)
        child = p1[:point] + p2[point:]
        return child

    def _mutate(self, child):
        # Perturb one gene by a small random integer; the repair step restores feasibility
        idx = random.randint(0, len(child) - 1)
        child[idx] = max(0, child[idx] + random.randint(-10, 10))
        return child
