# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# AI-assisted experimental code; full human and mathematical review is not established.

"""Canonical unique-best-response maps for square bimatrix games.

Maps need not be bijections. Ties are rejected explicitly; choosing an
arbitrary tied index would destroy relabeling invariance. Canonicalization
uses 2(k!)^2 transformations, so this is intended for small games.
"""

from itertools import permutations
import numpy as np
from .game_classes import validate_game


def best_response_maps(A, B):
    A, B = validate_game(A, B)
    if np.any((A == A.max(axis=0)).sum(axis=0) != 1) or np.any(
            (B == B.max(axis=1, keepdims=True)).sum(axis=1) != 1):
        raise ValueError("Unique-best-response maps are undefined when maxima are tied")
    return tuple(map(int, A.argmax(axis=0))), tuple(map(int, B.argmax(axis=1)))


def canonical_maps(sigma, tau):
    """Canonical pair under row/column relabeling and (sigma,tau)->(tau,sigma)."""
    k = len(sigma)
    if k < 2 or len(tau) != k or any(
        not isinstance(x, (int, np.integer)) or not 0 <= x < k for x in (*sigma,*tau)
    ):
        raise ValueError("Expected two maps from range(k) to range(k), k >= 2")
    candidates = []
    for s,t in ((sigma,tau),(tau,sigma)):
        for rows in permutations(range(k)):
            inv_rows = np.argsort(rows)
            for cols in permutations(range(k)):
                inv_cols = np.argsort(cols)
                candidates.append((
                    tuple(rows[s[inv_cols[j]]] for j in range(k)),
                    tuple(cols[t[inv_rows[i]]] for i in range(k)),
                ))
    return min(candidates)


def br_type(A, B):
    return canonical_maps(*best_response_maps(A, B))


def br_type_properties(sigma, tau):
    """Count mutual best responses and cycles of tau o sigma, excluding transients."""
    from .cycles import find_cycles_in_dynamics
    canonical_maps(sigma,tau)  # Validate maps, including non-bijective ones.
    dynamics = {j: tau[sigma[j]] for j in range(len(sigma))}
    cycles = find_cycles_in_dynamics(dynamics)
    return {
        "ne_count": sum(dynamics[j] == j for j in dynamics),
        "cycle_structure": tuple(sorted(map(len, cycles))),
        "sigma_perm": len(set(sigma)) == len(sigma),
        "tau_perm": len(set(tau)) == len(tau),
    }
