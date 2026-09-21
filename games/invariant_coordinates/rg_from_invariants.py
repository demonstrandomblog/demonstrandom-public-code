# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Experimental AI-assisted research code; human review is incomplete.
"""Strict ordinal classes of two-player 2x2 games with players distinguished.
Payoffs a and b list cells in row-major order. mean_zero returns unnormalized
Hadamard contrasts (row,column,interaction) for each player. rg_type ranks
each player's payoffs and minimizes over the four joint row/column permutations.
The sign-feature experiment uses the 144 representatives with payoffs 1,2,3,4;
it does not establish a rule for arbitrary cardinal payoffs or tied games."""
import numpy as np
from itertools import permutations


def mean_zero(a1, a2, a3, a4, b1, b2, b3, b4):
    rA = a1 + a2 - a3 - a4
    cA = a1 - a2 + a3 - a4
    dA = a1 - a2 - a3 + a4
    rB = b1 + b2 - b3 - b4
    cB = b1 - b2 + b3 - b4
    dB = b1 - b2 - b3 + b4
    return rA, cA, dA, rB, cB, dB


def eval_generators(rA, cA, dA, rB, cB, dB):
    deg2 = [rA**2, rA*rB, cA**2, cA*cB, dA**2, dA*dB, rB**2, cB**2, dB**2]
    deg3 = [cA*dA*rA, cA*dB*rA, cB*dA*rA, cB*dB*rA,
            cA*dA*rB, cA*dB*rB, cB*dA*rB, cB*dB*rB]
    return deg2 + deg3


def ordinal_type(payoffs):
    """Strict ordinal ranking as a tuple, or None if there is a tie."""
    sorted_idx = sorted(range(len(payoffs)), key=lambda i: payoffs[i])
    for i in range(len(payoffs) - 1):
        if payoffs[sorted_idx[i]] == payoffs[sorted_idx[i + 1]]:
            return None
    ranks = [0] * len(payoffs)
    for rank, idx in enumerate(sorted_idx):
        ranks[idx] = rank
    return tuple(ranks)


def rg_type(a, b):
    """Canonical R-G type: ordinal-pair orbit under S_2 x S_2."""
    ot_a, ot_b = ordinal_type(a), ordinal_type(b)
    if ot_a is None or ot_b is None:
        return None
    # S_2 x S_2 acts on 2x2 entries by row swap and column swap
    e = [0, 1, 2, 3]; s1 = [2, 3, 0, 1]; s2 = [1, 0, 3, 2]; s12 = [3, 2, 1, 0]
    orbit = {(tuple(ot_a[g[i]] for i in range(4)),
              tuple(ot_b[g[i]] for i in range(4)))
             for g in [e, s1, s2, s12]}
    return min(orbit)


def sign(x, tol=1e-10):
    return 1 if x > tol else (-1 if x < -tol else 0)


def enumerate_rg_types():
    """One game per R-G type, payoffs in {1, 2, 3, 4}."""
    rg_map = {}
    for pa in permutations(range(4)):
        for pb in permutations(range(4)):
            a = tuple(float(pa[i] + 1) for i in range(4))
            b = tuple(float(pb[i] + 1) for i in range(4))
            rgt = rg_type(a, b)
            if rgt is not None and rgt not in rg_map:
                rg_map[rgt] = (a, b)
    return rg_map


if __name__ == '__main__':
    rg_map = enumerate_rg_types()
    print(f"R-G types enumerated: {len(rg_map)} (expect 144)")

    # Signs of 17 generators + 9 magnitude comparisons
    pat_to_rg = {}
    collisions = 0
    for rgt, (a, b) in rg_map.items():
        gens = eval_generators(*mean_zero(*a, *b))
        signs = tuple(sign(g) for g in gens)
        comps = (
            sign(gens[0] - gens[2]), sign(gens[0] - gens[4]), sign(gens[2] - gens[4]),
            sign(gens[6] - gens[7]), sign(gens[6] - gens[8]), sign(gens[7] - gens[8]),
            sign(gens[0] - gens[6]), sign(gens[2] - gens[7]), sign(gens[4] - gens[8]),
        )
        pat = signs + comps
        if pat in pat_to_rg and pat_to_rg[pat] != rgt:
            collisions += 1
        else:
            pat_to_rg[pat] = rgt
    print(f"Signs + comparisons: {len(pat_to_rg)} patterns, {collisions} collisions")
    if collisions == 0 and len(pat_to_rg) == 144:
        print("VERIFIED: 17-generator signs + 9 comparisons exactly recover 144 R-G types")
