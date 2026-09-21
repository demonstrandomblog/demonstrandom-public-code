# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# AI-assisted experimental code; full human and mathematical review is not established.

"""Square bimatrix games and exact monomial orbit averages.

Coordinates are A.ravel() followed by B.ravel(); A chooses rows, B columns.
The symmetry includes independent strategy relabelings and player exchange
(A,B)->(B.T,A.T). Orbit averages are invariant but a degree cutoff does not
establish a generating or separating set.
"""

from fractions import Fraction
from itertools import permutations, combinations_with_replacement
import numpy as np


def validate_game(A, B):
    A, B = np.asarray(A, dtype=float), np.asarray(B, dtype=float)
    if A.ndim != 2 or A.shape != B.shape or A.shape[0] != A.shape[1] or len(A) < 2:
        raise ValueError("A and B must be matching square matrices with k >= 2")
    if not np.isfinite(A).all() or not np.isfinite(B).all():
        raise ValueError("Payoffs must be finite")
    return A, B


def validate_k(k):
    if not isinstance(k, (int, np.integer)) or k < 2:
        raise ValueError("k must be an integer at least two")
    return int(k)


def build_group_perms(k):
    """Return all 2(k!)^2 permutations mapping old coordinates to new ones."""
    k = validate_k(k)
    group = []
    for rows in permutations(range(k)):
        for cols in permutations(range(k)):
            for swap in (False, True):
                perm = []
                for player in range(2):
                    for i in range(k):
                        for j in range(k):
                            if swap:
                                perm.append((1-player)*k*k + rows[j]*k + cols[i])
                            else:
                                perm.append(player*k*k + rows[i]*k + cols[j])
                group.append(tuple(perm))
    return group


def monomial_orbits(k, degree):
    """Enumerate monomial orbits; factorial growth limits this to small games."""
    if not isinstance(degree, int) or degree < 0:
        raise ValueError("degree must be a nonnegative integer")
    group = build_group_perms(k)
    remaining = set(combinations_with_replacement(range(2*k*k), degree))
    result = {}
    while remaining:
        rep = min(remaining)
        orbit = {tuple(sorted(g[i] for i in rep)) for g in group}
        result[rep] = tuple(sorted(orbit))
        remaining.difference_update(orbit)
    return result


def eval_invariant(game, orbit_members, order=None):
    """Exact Reynolds average over distinct monomials.

    Each distinct orbit member has equal stabilizer size, so divide by the
    orbit size. Optional order is accepted for source-call compatibility;
    it is not the denominator of a distinct-member sum.
    """
    if not orbit_members:
        raise ValueError("An orbit must contain at least one monomial")
    values = [Fraction(x) for x in game]
    total = Fraction(0)
    for member in orbit_members:
        product = Fraction(1)
        for index in member:
            product *= values[index]
        total += product
    return total / len(orbit_members)


def class_diagnostics(A, B, atol=1e-10):
    """Return overlapping class predicates in the supplied player/strategy labels.

    Symmetric means B=A.T in these labels, not symmetry after an unknown
    relabeling. Zero-sum/common-interest are literal payoff equalities.
    Dominance means strictly better against every opposing pure strategy.
    """
    from .potential_subvariety import is_potential_game
    A, B = validate_game(A, B)
    if not np.isfinite(atol) or atol < 0:
        raise ValueError("atol must be finite and nonnegative")
    k = len(A)
    dominant_rows = [i for i in range(k)
                     if all(np.all(A[i] > A[j] + atol) for j in range(k) if j != i)]
    dominant_cols = [j for j in range(k)
                     if all(np.all(B[:, j] > B[:, l] + atol) for l in range(k) if l != j)]
    return {
        "potential": is_potential_game(np.r_[A.ravel(), B.ravel()], k, atol),
        "zero_sum": bool(np.allclose(A, -B, atol=atol, rtol=0)),
        "common_interest": bool(np.allclose(A, B, atol=atol, rtol=0)),
        "symmetric_in_given_labels": bool(np.allclose(A, B.T, atol=atol, rtol=0)),
        "strictly_dominant_rows": dominant_rows,
        "strictly_dominant_columns": dominant_cols,
    }
