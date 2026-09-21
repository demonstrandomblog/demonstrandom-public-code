# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# AI-assisted experimental code; full human and mathematical review is not established.

"""Orthogonal potential, harmonic, and nonstrategic decomposition for square games.

A row player and a column player each have k strategies. Flatten payoffs as
A.ravel(), then B.ravel(). The standard Euclidean payoff inner product is
used; square equal-action games have equal player weights. Numerical SVD
uses a fixed 1e-10 rank threshold on the integer difference operators.
"""

import numpy as np
from .game_classes import validate_k, validate_game

def build_difference_operator(k):
    """Build the difference operator D for a k x k bimatrix game.

    For each player, D maps payoffs to incentive differences.
    Player A's payoff is a k x k matrix A. For each column j (opponent
    strategy), the incentive to play row i vs row i' is A[i,j] - A[i',j].

    D maps R^{2k^2} -> R^{2 * C(k,2) * k}, where C(k,2)*k counts the
    number of (strategy pair, opponent strategy) triples per player.

    Returns D as a matrix.
    """
    k = validate_k(k)
    n_payoffs = 2 * k * k

    # Number of strategy pairs per player: C(k,2)
    pairs = [(i, ip) for i in range(k) for ip in range(i+1, k)]
    n_diffs_per_player = len(pairs) * k
    n_diffs = 2 * n_diffs_per_player

    D = np.zeros((n_diffs, n_payoffs))

    row = 0
    # Player A: for each pair (i, i') and each column j
    for (i, ip) in pairs:
        for j in range(k):
            # Incentive: A[i,j] - A[i',j]
            D[row, i * k + j] = 1.0
            D[row, ip * k + j] = -1.0
            row += 1

    # Player B: for each pair (j, j') and each row i
    for (j, jp) in pairs:
        for i in range(k):
            # Incentive: B[i,j] - B[i,j']
            D[row, k * k + i * k + j] = 1.0
            D[row, k * k + i * k + jp] = -1.0
            row += 1

    assert row == n_diffs
    return D


def build_potential_operator(k):
    """Build the 'potential test' operator for a k x k bimatrix game.

    A game (A, B) is a potential game if there exists Phi such that:
      A[i,j] - A[i',j] = Phi[i,j] - Phi[i',j]  for all i,i',j
      B[i,j] - B[i,j'] = Phi[i,j] - Phi[i,j']  for all i,j,j'

    Equivalently: the "curl" of the incentive field is zero.
    The curl operator C maps incentive differences to their circulations
    around 4-cycles (i,j) -> (i',j) -> (i',j') -> (i,j') -> (i,j).

    Potential games = ker(C * D).
    """
    k = validate_k(k)
    pairs = [(i, ip) for i in range(k) for ip in range(i+1, k)]
    col_pairs = [(j, jp) for j in range(k) for jp in range(j+1, k)]

    n_payoffs = 2 * k * k
    # Number of 4-cycles: C(k,2) * C(k,2)
    n_cycles = len(pairs) * len(col_pairs)

    # The curl checks: for each (i,i') x (j,j'):
    #   [A[i,j] - A[i',j]] - [A[i,j'] - A[i',j']]
    #   should equal
    #   [B[i,j] - B[i,j']] - [B[i',j] - B[i',j']]
    #
    # Rearranging: A[i,j]-A[i',j]-A[i,j']+A[i',j'] = B[i,j]-B[i,j']-B[i',j]+B[i',j']
    # i.e. the 2D discrete derivative of A (rows) = 2D discrete derivative of B (cols)
    #
    # This is the "curl = 0" condition. We encode it as:
    #   A[i,j] - A[i',j] - A[i,j'] + A[i',j'] - B[i,j] + B[i,j'] + B[i',j] - B[i',j'] = 0

    C = np.zeros((n_cycles, n_payoffs))
    row = 0
    for (i, ip) in pairs:
        for (j, jp) in col_pairs:
            # A interaction term
            C[row, i * k + j] = 1.0
            C[row, ip * k + jp] = 1.0
            C[row, ip * k + j] = -1.0
            C[row, i * k + jp] = -1.0
            # B interaction term (opposite sign)
            C[row, k * k + i * k + j] = -1.0
            C[row, k * k + ip * k + jp] = -1.0
            C[row, k * k + ip * k + j] = 1.0
            C[row, k * k + i * k + jp] = 1.0
            row += 1

    assert row == n_cycles
    return C


def candogan_projectors(k):
    """Compute the three orthogonal Candogan projectors for k x k bimatrix games.

    Returns (P_pot, P_harm, P_ns) where:
      P_ns:   projection onto non-strategic subspace (ker D)
      P_pot:  projection onto potential games (image of D^T restricted to curl-free)
      P_harm: projection onto harmonic games (strategic but not potential)

    The decomposition: V = im(D^T) ⊕ ker(D) splits into strategic vs non-strategic.
    Within strategic: im(D^T) = potential ⊕ harmonic, where potential = im(D^T) ∩ ker(C).
    """
    k = validate_k(k)
    n = 2 * k * k
    D = build_difference_operator(k)
    C = build_potential_operator(k)

    # Non-strategic = ker(D)
    # Compute via SVD
    U, s, Vt = np.linalg.svd(D, full_matrices=True)
    rank_D = np.sum(s > 1e-10)
    # Columns of V corresponding to zero singular values span ker(D)
    V_null = Vt[rank_D:].T  # n x (n - rank_D)

    P_ns = V_null @ V_null.T

    # Strategic = im(D^T) = complement of ker(D)
    V_strat = Vt[:rank_D].T  # n x rank_D
    P_strat = V_strat @ V_strat.T

    # Within strategic subspace, find potential games
    # Potential games satisfy C @ v = 0 AND are in im(D^T) (i.e., are strategic)
    # So: potential = ker(C) ∩ im(D^T)
    #
    # Project C onto strategic subspace: C_strat = C @ V_strat, then find its kernel
    C_strat = C @ V_strat  # n_cycles x rank_D
    Uc, sc, Vtc = np.linalg.svd(C_strat, full_matrices=True)
    rank_C_strat = np.sum(sc > 1e-10)

    # Potential subspace within strategic coordinates
    V_pot_strat = Vtc[rank_C_strat:].T  # rank_D x dim_pot
    # Map back to full coordinates
    V_pot = V_strat @ V_pot_strat  # n x dim_pot
    P_pot = V_pot @ V_pot.T

    # Harmonic = strategic - potential
    P_harm = P_strat - P_pot

    return P_pot, P_harm, P_ns

def decompose(A, B):
    """Return ((A_p,B_p), (A_h,B_h), (A_ns,B_ns)) in payoff coordinates."""
    A,B = validate_game(A,B)
    v = np.r_[A.ravel(),B.ravel()]
    return tuple((P @ v).reshape(2,len(A),len(A)) for P in candogan_projectors(len(A)))
