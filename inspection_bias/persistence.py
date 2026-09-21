# AI-assisted research code; see README.md and the repository AI_NOTICE.md.
"""
Candidate persistence functions w(k) and functional information computation.

The effective weight w(k) combines reaching a configuration of complexity k
and its persistence. Functional information I = -log2(w_bar), with w_bar the
possibility-space-weighted average. The fitting model is provisional.

Data from Hazen & Wong (2024), "Open-ended versus bounded evolution:
Mineral evolution as a case study", PNAS Nexus 3(7), Table 1.
"""

import numpy as np
from math import lgamma


# ---------------------------------------------------------------
# Hazen-Wong Table 1: (k_max, m, n, M_observed, I_observed_bits)
# ---------------------------------------------------------------
HAZEN_WONG_DATA = [
    (4,  16, 19,   27,   22.5),   # I: Stellar
    (7,  23, 19,   59,   40.0),   # II: Nebular
    (7,  29, 36,   96,   49.3),   # III: Chondrite
    (8,  34, 36,  141,   57.2),   # IV: Achondrite
    (8,  42, 71,  296,   67.3),   # V: Sec. Achondrite
    (10, 48, 72,  442,   84.5),   # VI: Hadean
    (11, 61, 72,  883,   95.5),   # VIB: Late Hadean
    (12, 68, 123, 3759, 113.3),   # VII/VIII: Ign+Meta
    (15, 72, 135, 9300, 138.6),   # IX-XII: Modern
]

STAGE_NAMES = [
    "I: Stellar", "II: Nebular", "III: Chondrite", "IV: Achondrite",
    "V: Sec. Achondrite", "VI: Hadean", "VIB: Late Hadean",
    "VII/VIII: Ign+Meta", "IX-XII: Modern",
]


# ---------------------------------------------------------------
# Combinatorial building blocks (in log2 space for stability)
# ---------------------------------------------------------------

def log2_binom(k, m):
    """log2(C(k, m)) = log2(m choose k)."""
    return (sum(np.log2(m - i) for i in range(k))
            - sum(np.log2(i + 1) for i in range(k)))


def log2_Pprime(k, n):
    """
    log2(P'(k, n)): nonequivalent coefficient permutations.

    P'(k,n) = P(k,n) * correction(k) where P(k,n) = n!/(n-k)!
    Corrections from Hazen-Wong account for equivalent permutations.
    """
    log_P = sum(np.log2(n - i) for i in range(k))
    corrections = {1: 1.0, 2: 0.64, 3: 0.836, 4: 0.924, 5: 0.965, 6: 0.983}
    return log_P + np.log2(corrections.get(k, 1.0))


def log2_Phi(k, m, n):
    """log2(Phi(k)) = log2(C(k,m) * P'(k,n)), the possibility-space weight."""
    return log2_binom(k, m) + log2_Pprime(k, n)


def log2_sum_exp2(terms):
    """Numerically stable log2(sum(2^t for t in terms))."""
    terms = list(terms)
    if not terms:
        return -np.inf
    mx = max(terms)
    if not np.isfinite(mx):
        return mx
    return mx + np.log2(sum(2 ** (t - mx) for t in terms))


# ---------------------------------------------------------------
# Functional information from persistence function
# ---------------------------------------------------------------

def compute_I(k_max, m, n, log2_w):
    """
    Compute functional information I = log2(N/M) = -log2(w_bar).

    Parameters
    ----------
    k_max : int
        Maximum formula complexity.
    m : int
        Number of available elements.
    n : int
        Maximum stoichiometric coefficient.
    log2_w : callable
        log2(w(k)) for the candidate persistence function.

    Returns
    -------
    float
        Functional information in bits.
    """
    log2_N_terms = []
    log2_M_terms = []
    for k in range(1, k_max + 1):
        base = log2_Phi(k, m, n)
        log2_N_terms.append(base)
        log2_M_terms.append(base + log2_w(k))
    return log2_sum_exp2(log2_N_terms) - log2_sum_exp2(log2_M_terms)


def compute_log2_N(k_max, m, n):
    """Compute log2 of the possibility space size N."""
    terms = [log2_Phi(k, m, n) for k in range(1, k_max + 1)]
    return log2_sum_exp2(terms)


# ---------------------------------------------------------------
# Candidate persistence function factories
# ---------------------------------------------------------------
# Each factory takes a parameter vector and returns log2(w(k)).

def exponential_factory(params):
    """w(k) = A * exp(-beta * k). Params: [log2_A, beta]."""
    log2A, beta = params
    return lambda k: log2A - beta * k / np.log(2)


def power_law_factory(params):
    """w(k) = A * k^(-gamma). Params: [log2_A, gamma]."""
    log2A, gamma = params
    return lambda k: log2A - gamma * np.log2(k)


def gaussian_factory(params):
    """w(k) = A * exp(-beta * k^2). Params: [log2_A, beta]."""
    log2A, beta = params
    return lambda k: log2A - beta * k ** 2 / np.log(2)


def factorial_factory(params):
    """w(k) = A / (k!)^alpha. Params: [log2_A, alpha]."""
    log2A, alpha = params
    return lambda k: log2A - alpha * lgamma(k + 1) / np.log(2)


MODELS = {
    "exponential": {
        "factory": exponential_factory,
        "formula": "w(k) = A * exp(-beta * k)",
        "p0": [0.0, 6.0],
        "param_names": ["log2_A", "beta"],
    },
    "power_law": {
        "factory": power_law_factory,
        "formula": "w(k) = A * k^(-gamma)",
        "p0": [0.0, 10.0],
        "param_names": ["log2_A", "gamma"],
    },
    "gaussian": {
        "factory": gaussian_factory,
        "formula": "w(k) = A * exp(-beta * k^2)",
        "p0": [0.0, 0.5],
        "param_names": ["log2_A", "beta"],
    },
    "factorial": {
        "factory": factorial_factory,
        "formula": "w(k) = A / (k!)^alpha",
        "p0": [0.0, 1.0],
        "param_names": ["log2_A", "alpha"],
    },
}
