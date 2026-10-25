"""Historical (baseline) schedule of the CRS Hospital."""

import numpy as np


def build_historical_schedule(problem):
    """Builds an integer schedule x[i, j, k] reproducing the observed hours.

    Each specialty's observed weekly Admissions and Checkups hours are scaled to
    a month (x4) and split as evenly as possible among its physicians; hours are
    then placed in rooms in index order, respecting room capacity. Only the
    per-specialty Admissions/Checkups totals matter to the queue evaluator.
    """
    X_orig = np.zeros((problem.i, problem.j, problem.k), dtype=int)
    room_capacity = list(problem.o)

    def assign_hours(medic, activity, hours_to_assign):
        for k in problem.K:
            if hours_to_assign == 0: break
            available = min(hours_to_assign, room_capacity[k])
            if available > 0:
                X_orig[medic, activity, k] += available
                room_capacity[k] -= available
                hours_to_assign -= available

    for s in problem.S:
        adm_hrs = problem.current_adm_hrs[s] * 4
        chk_hrs = problem.current_chk_hrs[s] * 4
        medics = problem.E[s]
        n_medics = len(medics)
        base_adm = adm_hrs // n_medics
        rem_adm = adm_hrs % n_medics
        for idx, m in enumerate(medics):
            assign_hours(m, 0, base_adm + (1 if idx < rem_adm else 0))
        base_chk = chk_hrs // n_medics
        rem_chk = chk_hrs % n_medics
        for idx, m in enumerate(medics):
            assign_hours(m, 1, base_chk + (1 if idx < rem_chk else 0))

    return X_orig
