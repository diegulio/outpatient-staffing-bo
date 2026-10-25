"""Feasible scheduler module (Gurobi MILP).

Generates feasible integer schedules x[i, j, k] (monthly hours of physician i in
activity j and room k) and repairs infeasible schedules to the nearest feasible
one in L1 distance. The constraints follow the feasible-schedule appendix of the manuscript.
"""

import numpy as np
from gurobipy import Model, GRB, quicksum


def _add_hospital_constraints(m, problem, x):
    """Adds the hospital scheduling constraints to Gurobi model ``m``."""
    # Room capacity
    m.addConstrs((x.sum('*', '*', k) <= problem.o[k] for k in problem.K), 'rest1')
    # Physician contract cap
    m.addConstrs((x.sum(i, '*', '*') <= problem.c[i] for i in problem.I), 'rest2')
    # Minimum hours per activity (inactive in the full case: a = 0)
    m.addConstrs((x.sum('*', j, '*') >= problem.a[j] for j in problem.J), 'rest3')

    # Activities not allowed for a specialty (inactive in the full case: U is empty)
    m.addConstrs(((quicksum(quicksum(quicksum(x[i,j,k] for k in problem.K) for j in problem.U[s]) for i in problem.E[s]) == 0 for s in problem.S)), 'rest5')
    # Admissions share of each specialty's hours is at most p
    m.addConstrs(((quicksum(quicksum(quicksum(x[i,j,k] for k in problem.K) for j in problem.C[s]) for i in problem.E[s]) - problem.p*(quicksum(quicksum(quicksum(x[i,j,k] for k in problem.K) for j in problem.J) for i in problem.E[s])) <= 0 for s in problem.S)), 'rest6')
    # Activities not allowed in a room (inactive in the full case: R is empty)
    m.addConstrs((quicksum(quicksum(x[i,j,k] for i in problem.E[s]) for j in problem.R[k]) == 0 for s in problem.S for k in problem.K), 'rest7')

    # Specialty-hour budget: total monthly hours of a specialty's physicians may not
    # exceed four times its observed weekly Admissions + Checkups hours.
    if hasattr(problem, 'current_adm_hrs') and hasattr(problem, 'current_chk_hrs'):
        m.addConstrs(
            (quicksum(quicksum(quicksum(x[i,j,k] for k in problem.K) for j in problem.J) for i in problem.E[s])
             <= (problem.current_adm_hrs[s] + problem.current_chk_hrs[s]) * 4 for s in problem.S),
            'specialty_hour_budget'
        )


def sample_feasible_space(problem, n_samples):
    """Draws ``n_samples`` feasible schedules from a Gurobi solution pool.

    The model is solved once with independent uniform random objective
    coefficients on every variable, collecting a diverse pool of at most 1,000
    feasible schedules; ``n_samples`` schedules are then drawn from the pool
    (with replacement). Returns (flat_schedules, schedule_matrices).
    """
    X = list()
    X_matrix = list()
    pool_size = 1000  # at most 1,000 pooled solutions

    m = Model('feasible_schedule_sampling')
    m.setParam('OutputFlag', 0)
    m.Params.PoolSolutions = pool_size
    m.Params.PoolSearchMode = 2

    x = m.addVars(problem.I, problem.J, problem.K, lb=0, vtype=GRB.INTEGER)

    _add_hospital_constraints(m, problem, x)

    # Randomized linear objective to diversify the solution pool
    for i in problem.I:
        for j in problem.J:
            for k in problem.K:
                x[i,j,k].Obj = np.random.random()

    m.ModelSense = GRB.MAXIMIZE

    try:
        m.optimize()
    except Exception as e:
        print("Error while optimizing the Gurobi feasible space:", e)
        return [], []

    for _ in range(n_samples):
        try:
            # Pick a random solution from the pool
            m.setParam('SolutionNumber', np.random.randint(min(pool_size-1, m.SolCount)))
            xn = m.getAttr("Xn")
            x_matrix = np.array(xn).reshape(len(problem.I), len(problem.J), len(problem.K))
            X.append(xn)
            X_matrix.append(x_matrix)
        except Exception:
            break

    return X, X_matrix


def repair_solution(problem, target_X):
    """Returns the feasible schedule closest (L1 distance) to ``target_X``.

    Used by the genetic algorithm and simulated annealing, whose crossover,
    mutation, and neighborhood moves may violate the scheduling constraints.
    ``target_X`` is a flat list of length i*j*k. Returns None if no optimal
    repair is found.
    """
    m = Model('repair_solution')
    m.setParam('OutputFlag', 0)

    x = m.addVars(problem.I, problem.J, problem.K, lb=0, vtype=GRB.INTEGER)

    _add_hospital_constraints(m, problem, x)

    # Linearized L1 distance: z >= x - target, z >= target - x
    z = m.addVars(problem.I, problem.J, problem.K, lb=0)

    t_idx = 0
    for i in problem.I:
        for j in problem.J:
            for k in problem.K:
                m.addConstr(z[i,j,k] >= x[i,j,k] - target_X[t_idx])
                m.addConstr(z[i,j,k] >= target_X[t_idx] - x[i,j,k])
                t_idx += 1

    m.setObjective(z.sum(), GRB.MINIMIZE)

    try:
        m.optimize()
        if m.Status == GRB.OPTIMAL:
            return list(m.getAttr("X", x).values())
    except Exception as e:
        print("Error in repair_solution:", e)

    return None
