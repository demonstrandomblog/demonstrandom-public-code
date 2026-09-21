# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Experimental AI-assisted research code; human review is incomplete.
"""Compare a contrast criterion with strict dominance in two-player 2x2 games.
Eight scalars give player A's four row-major payoffs followed by player B's.
Contrasts are unnormalized row, column, and interaction sums. The random
experiment compares its Boolean condition against explicit best responses."""
import numpy as np


def mean_zero(a1, a2, a3, a4, b1, b2, b3, b4):
    rA = a1 + a2 - a3 - a4
    cA = a1 - a2 + a3 - a4
    dA = a1 - a2 - a3 + a4
    rB = b1 + b2 - b3 - b4
    cB = b1 - b2 + b3 - b4
    dB = b1 - b2 - b3 + b4
    return rA, cA, dA, rB, cB, dB


def is_solvable_brute(a1, a2, a3, a4, b1, b2, b3, b4):
    A = np.array([[a1, a2], [a3, a4]])
    B = np.array([[b1, b2], [b3, b4]])
    p1 = [True, True]; p2 = [True, True]
    changed = True
    while changed:
        changed = False
        a1 = [i for i in range(2) if p1[i]]; a2 = [j for j in range(2) if p2[j]]
        if len(a1) == 2:
            if all(A[0, j] > A[1, j] for j in a2):
                p1[1] = False; changed = True
            elif all(A[1, j] > A[0, j] for j in a2):
                p1[0] = False; changed = True
        a1 = [i for i in range(2) if p1[i]]; a2 = [j for j in range(2) if p2[j]]
        if len(a2) == 2:
            if all(B[i, 0] > B[i, 1] for i in a1):
                p2[1] = False; changed = True
            elif all(B[i, 1] > B[i, 0] for i in a1):
                p2[0] = False; changed = True
    return sum(p1) == 1 and sum(p2) == 1


def solvable_condition(rA, cA, dA, rB, cB, dB):
    """Solvable iff at least one player has a dominant strategy."""
    return rA**2 > dA**2 or cB**2 > dB**2


if __name__ == '__main__':
    rng = np.random.default_rng(42)
    n_mismatch = 0
    for _ in range(100000):
        payoffs = tuple(rng.standard_normal(8))
        brute = is_solvable_brute(*payoffs)
        cond = solvable_condition(*mean_zero(*payoffs))
        if brute != cond:
            n_mismatch += 1
    print(f"100,000 random games: {n_mismatch} mismatches "
          f"({'VERIFIED' if n_mismatch == 0 else 'FAIL'})")
