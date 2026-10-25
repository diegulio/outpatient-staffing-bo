"""Stochastic, discrete-time macroscopic queue model (Simulation module).

For each specialty, the model tracks two queues (Admissions and Checkups) over
monthly steps. Uncertainty enters through physician absences, patient no-shows,
discharge probabilities, and a reduced-form referral inflow. See the
Simulation module section of the manuscript for the full description.
"""

import numpy as np
import scipy.stats as st

# Planning horizon used in the manuscript (months)
N_MONTHS = 10


def simulate_specialty_waitlist(n_months, consult_hrs, p_admissions, specialty_idx, problem):
    """Simulates one specialty and returns its total queue at month ``n_months``.

    consult_hrs  : total monthly consultation hours of the specialty (H_A + H_C)
    p_admissions : share of those hours assigned to Admissions, H_A / (H_A + H_C)
    """
    # Service yields (patients served per hour)
    yield_adm = problem.mu[specialty_idx, 0]
    yield_chk = problem.mu[specialty_idx, 1]

    # -------------------------------------------------------------------------
    # Initial state at t=0
    # -------------------------------------------------------------------------
    total_q = problem.queue_length[specialty_idx]
    adm_queue_0 = int(total_q * p_admissions)        # patients waiting for Admissions
    chk_queue_0 = int(total_q * (1 - p_admissions))  # patients waiting for Checkups

    # Monthly referral inflow: uniform draw in [80%, 120%] of a reference equal to
    # 10% of the initial Admissions queue. These are NEW referrals arriving each
    # month (from month 2 onward); they are not part of the initial queue, which
    # already contains all patients waiting at t=0.
    referrals_base = int(adm_queue_0 * 0.1)
    referrals_low = max(0, int(referrals_base * 0.8))
    referrals_high = max(referrals_low, int(referrals_base * 1.2))

    # External arrivals enter the Admissions queue only (weekly rate x 4 weeks/month)
    external_arrivals = int(problem.lambdas_external[specialty_idx] * 4)

    # -------------------------------------------------------------------------
    # Stochastic parameters
    # -------------------------------------------------------------------------
    # Patient no-show probability: uniform draw in [9.804%, 14.828%]
    noshow_low = 0.09804349
    noshow_high = 0.14827611

    # Discharge probability: uniform draw +/-2% around the historical estimate
    d_prob = problem.discharge_prob[specialty_idx]
    discharge_low = max(0.01, d_prob - 0.02)
    discharge_high = min(0.99, d_prob + 0.02)

    # Physician absence rate: uniform draw in [1%, 7%]
    absence_low = 0.01
    absence_high = 0.07

    def _draw_noshow():
        return float((noshow_high - noshow_low) * np.random.random() + noshow_low)

    def _draw_absence():
        return float((absence_high - absence_low) * np.random.random() + absence_low)

    def _draw_discharge():
        return float((discharge_high - discharge_low) * np.random.random() + discharge_low)

    # -------------------------------------------------------------------------
    # Realized service and discharges for one queue in one month.
    #
    # Physician absence reduces the scheduled supply first; patient no-shows
    # then reduce the patients actually seen. Applying both multiplicatively
    # avoids double-counting their joint effect. Counts are clamped at zero so
    # that stochastic draws never produce negative realized service.
    # -------------------------------------------------------------------------
    def _serve_queue(demand, capacity):
        """Returns (served, discharged) for a single queue in one month.

        demand   : patients in the queue this month
        capacity : nominal throughput (hours x service yield)
        """
        scheduled = min(demand, capacity)
        effective_supply = scheduled * (1.0 - _draw_absence())
        served = max(0, int(effective_supply * (1.0 - _draw_noshow())))
        discharged = int(_draw_discharge() * served)
        return served, discharged

    # The schedule is fixed over the horizon, so monthly capacity is constant.
    capacity_adm = consult_hrs * p_admissions * yield_adm
    capacity_chk = consult_hrs * (1 - p_admissions) * yield_chk

    # -------------------------------------------------------------------------
    # Month 1: initial backlog plus external arrivals (no referral inflow yet)
    # -------------------------------------------------------------------------
    demand_adm_1 = adm_queue_0 + external_arrivals
    demand_chk_1 = chk_queue_0

    served_adm_1, discharged_adm_1 = _serve_queue(demand_adm_1, capacity_adm)
    served_chk_1, discharged_chk_1 = _serve_queue(demand_chk_1, capacity_chk)

    # State lists (index 0 unused; index t = month t)
    demand_adm = [0, demand_adm_1]
    demand_chk = [0, demand_chk_1]
    served_adm = [0, served_adm_1]
    discharged_adm = [0, discharged_adm_1]
    discharged_chk = [0, discharged_chk_1]

    # -------------------------------------------------------------------------
    # Months 2..n_months
    # -------------------------------------------------------------------------
    for month in range(2, n_months + 1):
        referrals = int((referrals_high - referrals_low) * np.random.random() + referrals_low)

        # Admissions: previous demand - served + external arrivals + referrals
        d_adm = max(0, int(
            demand_adm[month - 1]
            - served_adm[month - 1]
            + external_arrivals
            + referrals
        ))

        # Checkups: previous demand
        #   + (Admissions served - Admissions discharged)  -> patients moving to Checkups
        #   - Checkups discharged                          -> patients leaving the system
        # (Checkups patients seen but not discharged return to the same queue,
        #  so Checkups service cancels out except for its discharges.)
        d_chk = max(0, int(
            demand_chk[month - 1]
            + served_adm[month - 1] - discharged_adm[month - 1]
            - discharged_chk[month - 1]
        ))

        s_adm, dis_adm = _serve_queue(d_adm, capacity_adm)
        s_chk, dis_chk = _serve_queue(d_chk, capacity_chk)

        demand_adm.append(d_adm)
        demand_chk.append(d_chk)
        served_adm.append(s_adm)
        discharged_adm.append(dis_adm)
        discharged_chk.append(dis_chk)

    # Total waitlist = Admissions demand + Checkups demand at the final month
    return demand_adm[n_months] + demand_chk[n_months]


def _simulate_total_waitlist(problem, X):
    """One stochastic replication of the hospital-wide waitlist at month N_MONTHS.

    The detailed schedule X[i][j][k] is aggregated into Admissions and Checkups
    hours per specialty; physician and room identities do not enter the recursion.
    """
    total = 0
    for s_idx in range(problem.s):
        adm_hrs = sum(X[i][0][k] for i in problem.E[s_idx] for k in problem.K)
        chk_hrs = sum(X[i][1][k] for i in problem.E[s_idx] for k in problem.K)
        consult_hrs = adm_hrs + chk_hrs
        p_admissions = adm_hrs / consult_hrs if consult_hrs > 0 else 0.5
        total += simulate_specialty_waitlist(N_MONTHS, consult_hrs, p_admissions, s_idx, problem)
    return total


def simulator_objective(problem, X, n_sims=3):
    """Objective used by the optimizers.

    Returns the NEGATIVE mean total waitlist across ``n_sims`` independent
    replications (default 3), so that maximizing the objective minimizes the
    expected waitlist.
    """
    if len(np.shape(X)) == 1:
        X = np.array(X).reshape(problem.i, problem.j, problem.k)

    results = [_simulate_total_waitlist(problem, X) for _ in range(n_sims)]
    return -np.mean(results)


def evaluate_solution_robust(problem, X, n_sims=30):
    """Final assessment of a schedule over ``n_sims`` fresh replications.

    Returns (mean_waitlist, (ci_lower, ci_upper)) as POSITIVE patient counts,
    where the interval is a Student-t 95% confidence interval.
    """
    if len(np.shape(X)) == 1:
        X = np.array(X).reshape(problem.i, problem.j, problem.k)

    results = [_simulate_total_waitlist(problem, X) for _ in range(n_sims)]

    mean_val = np.mean(results)
    std_err = st.sem(results)
    ci = st.t.interval(0.95, len(results) - 1, loc=mean_val, scale=std_err)

    return mean_val, ci
