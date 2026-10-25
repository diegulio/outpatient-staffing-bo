"""Case-study configuration: CRS Hospital Provincia Cordillera (SEMHS, Chile), 2019.

All numerical inputs below are the aggregated figures reported in the manuscript
(specialty, routing-probability, and demand tables) together with the observed weekly consultation hours per specialty.
No patient-level data are included in this repository.

Index sets (notation follows the feasible-schedule appendix of the manuscript):
    I : physicians (medics),        |I| = i = 44
    J : activities,                 |J| = j = 2   (0 = Admissions, 1 = Checkups)
    K : physical resources (rooms), |K| = k = 20
    S : specialties,                |S| = s = 15
"""

import random

import numpy as np


class HospitalProblem:
    def __init__(self, seed=0):
        # Seeds NumPy's global RNG, which drives the feasible-pool objective
        # coefficients, pool draws, the stochastic queue evaluator and the GP
        # hyperparameter restarts, and Python's RNG, which drives the genetic
        # algorithm and simulated annealing operators.
        np.random.seed(seed)
        random.seed(seed)
        self.setup_full_case()

    def setup_full_case(self):
        # ------------------------------------------------------------------
        # Index sets
        # ------------------------------------------------------------------
        self.i = 44  # physicians
        self.j = 2   # activities (0: Admissions, 1: Checkups)
        self.k = 20  # rooms
        self.s = 15  # specialties

        self.specialty_names = [
            'Cardiology', 'General surgery', 'Dermatology', 'Endocrinology',
            'Gastroenterology', 'Gynaecology', 'Physical medicine and rehab',
            'Internal medicine', 'Nephrology', 'Neurology', 'Ophthalmology',
            'Otorhinolaryngology', 'Pediatrics', 'Traumatology and orthopaedics', 'Urology'
        ]

        # Number of physicians per specialty; E[s] lists the physician indices of specialty s.
        medic_counts = [1, 4, 2, 1, 3, 3, 2, 8, 1, 1, 4, 3, 3, 5, 3]

        self.E = []
        curr = 0
        for count in medic_counts:
            self.E.append(list(range(curr, curr + count)))
            curr += count

        # ------------------------------------------------------------------
        # Scheduler parameters (monthly scale: 45 h/week x 4 weeks = 180 h)
        # ------------------------------------------------------------------
        self.o = [45*4 for _ in range(self.k)]  # monthly capacity of room k (O_r)
        self.a = [0 for _ in range(self.j)]     # minimum monthly hours of activity j (h_a); inactive in the full case
        self.c = [45*4 for _ in range(self.i)]  # monthly contract cap of physician i (C_m)

        self.U = [[] for _ in range(self.s)]    # activities not allowed for specialty s (empty in the full case)
        self.C = [[0] for _ in range(self.s)]   # Admissions activities of specialty s
        self.R = [[] for _ in range(self.k)]    # activities not allowed in room k (empty in the full case)

        self.S = range(self.s)
        self.K = range(self.k)
        self.I = range(self.i)
        self.J = range(self.j)
        self.p = 0.3  # maximum Admissions share of each specialty's hours

        # ------------------------------------------------------------------
        # Queue-model inputs (specialty table of the manuscript)
        # ------------------------------------------------------------------
        # Patients in queue at the beginning of January 2019
        self.queue_length = np.array([756, 978, 609, 23, 91, 573, 1299, 264, 16, 297, 1065, 1141, 274, 978, 455])
        # Historical specialty discharge probability
        self.discharge_prob = np.array([0.1034, 0.1533, 0.0605, 0.1018, 0.1619, 0.1249, 0.1817, 0.0952, 0.0507, 0.1135, 0.1013, 0.1493, 0.0299, 0.1653, 0.1283])

        # Observed (historical) weekly consultation hours per specialty
        self.current_adm_hrs = np.array([6, 14, 8, 13, 12, 8, 26, 39, 6, 8, 18, 17, 18, 17, 7])
        self.current_chk_hrs = np.array([10, 7, 9, 13, 18, 9, 24, 64, 8, 8, 14, 13, 25, 19, 11])

        # Patients served per week at each activity
        self.patients_served_adm = np.array([12, 52, 24, 26, 24, 24, 52, 78, 12, 16, 54, 51, 36, 51, 21])
        self.patients_served_chk = np.array([30, 35, 36, 52, 72, 36, 72, 192, 32, 24, 56, 52, 75, 76, 44])

        # Service yield (patients per hour) for each specialty and activity
        self.mu_adm = self.patients_served_adm / self.current_adm_hrs
        self.mu_chk = self.patients_served_chk / self.current_chk_hrs
        self.mu = np.column_stack((self.mu_adm, self.mu_chk))

        # ------------------------------------------------------------------
        # Routing probabilities between specialties (routing table of the manuscript),
        # estimated from 2018 SEMHS referral records. Stored for reference and for
        # the stability analysis (Proposition 1); NOT used by the queue evaluator.
        # ------------------------------------------------------------------
        self.routing_probs = np.array([
            [0, 0.0264, 0.0132, 0.2772, 0.1452, 0.0132, 0.1122, 0.0726, 0.0528, 0.1254, 0.0264, 0.0165, 0, 0.0660, 0.0528],
            [0.2651, 0, 0.0066, 0.0546, 0.3846, 0.0189, 0.0102, 0.0685, 0.0102, 0.0087, 0.0109, 0.0175, 0, 0.0830, 0.0612],
            [0, 0.0400, 0, 0.2400, 0.4000, 0, 0, 0.0400, 0, 0.1600, 0.0400, 0, 0, 0.0800, 0],
            [0.2311, 0.1873, 0.0398, 0, 0.0876, 0.0159, 0, 0.0040, 0.0239, 0.0637, 0.0558, 0.0279, 0, 0.2470, 0.0159],
            [0.3019, 0.1094, 0.0038, 0.1132, 0, 0.0906, 0.0075, 0.0377, 0.2189, 0, 0.0075, 0.0453, 0.0038, 0.0075, 0.0528],
            [0.1326, 0.0375, 0.0231, 0.2853, 0.2075, 0, 0.0231, 0.0720, 0.0058, 0.0173, 0.0202, 0.0086, 0, 0.0576, 0.1095],
            [0.0462, 0.0018, 0.0089, 0.0533, 0.0320, 0.0107, 0, 0.0622, 0.0249, 0.0710, 0.0107, 0.0231, 0.0515, 0.5968, 0.0071],
            [0.1167, 0.0397, 0.0068, 0.1343, 0.1654, 0.0227, 0.0986, 0, 0.0805, 0.0884, 0.0725, 0.0232, 0.0006, 0.0997, 0.0510],
            [0.1340, 0.0101, 0, 0.3970, 0.1608, 0.0335, 0.0469, 0.0201, 0, 0, 0.0101, 0.0067, 0, 0.0402, 0.1407],
            [0.1108, 0.0058, 0.0058, 0.0875, 0.0117, 0.0175, 0.5015, 0.0029, 0, 0, 0.0612, 0.0671, 0, 0.1166, 0.0117],
            [0.1242, 0, 0.0373, 0.0932, 0, 0, 0, 0.3789, 0, 0.2112, 0, 0.1242, 0.0062, 0.0248, 0],
            [0.2152, 0.0074, 0.0130, 0.2059, 0.2115, 0.0111, 0.0186, 0.0390, 0.0445, 0.0779, 0.0371, 0, 0.0371, 0.0705, 0.0111],
            [0.0186, 0.0124, 0.0464, 0.0093, 0.0186, 0.0124, 0.1981, 0.0217, 0.0124, 0.0433, 0.1548, 0.2972, 0, 0.1548, 0],
            [0.0072, 0.0069, 0.0014, 0.0076, 0.0080, 0.0065, 0.8919, 0.0098, 0.0007, 0.0470, 0.0040, 0.0047, 0.0007, 0, 0.0036],
            [0.2056, 0.1682, 0.0187, 0, 0.0935, 0.0748, 0.0935, 0.1028, 0.0374, 0, 0.0280, 0.0093, 0, 0.1682, 0]
        ])

        # Fixed weekly external-arrival rates per specialty (weekly-demand column of the demand table)
        self.lambdas_external = np.array([5, 45, 28, 25, 31, 29, 54, 72, 10, 23, 44, 58, 27, 55, 21])
