# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Experimental AI-assisted research code; human review is incomplete.
"""Elimination discriminant for three players with two strategies each.
A game has shape (3,2,2,2): payoff player followed by the three strategy axes.
Indifference equations are bilinear in opponents' mixing probabilities.
The computed discriminant concerns algebraic roots, not their membership in [0,1]."""
import numpy as np
from itertools import product as iproduct


def get_bilinear_coeffs(game):
    coeffs = []
    for p in range(3):
        others = sorted(q for q in range(3) if q != p)
        q, r = others
        D = np.zeros((2, 2))
        for sq in range(2):
            for sr in range(2):
                idx0 = [0]*3; idx1 = [0]*3
                idx0[p], idx1[p] = 0, 1
                idx0[q] = idx1[q] = sq
                idx0[r] = idx1[r] = sr
                D[sq, sr] = game[p][tuple(idx0)] - game[p][tuple(idx1)]
        a = D[1, 1]
        b = D[0, 1] - D[1, 1]
        c = D[1, 0] - D[1, 1]
        d = D[0, 0] - D[0, 1] - D[1, 0] + D[1, 1]
        coeffs.append((a, b, c, d))
    return coeffs


def ne_quadratic_discriminant(game):
    """Eliminate x_0 from f_1, f_2; then x_1 from f_0; collect quadratic in x_2."""
    (a0, b0, c0, d0), (a1, b1, c1, d1), (a2, b2, c2, d2) = get_bilinear_coeffs(game)
    E = a1*b2 - a2*b1
    F = a1*d2 - c2*b1
    G = c1*b2 - a2*d1
    H = c1*d2 - c2*d1
    A_co = G*d0 - H*c0
    B_co = E*d0 - F*c0 + G*b0 - H*a0
    C_co = E*b0 - F*a0
    return B_co**2 - 4 * A_co * C_co


def apply_s2_cubed(game, g):
    out = game.copy()
    for i in range(3):
        if g[i] == 1:
            out = np.flip(out, axis=i + 1)
    return out


if __name__ == '__main__':
    rng = np.random.default_rng(42)
    max_err = 0
    for _ in range(3000):
        game = rng.standard_normal((3, 2, 2, 2))
        d0 = ne_quadratic_discriminant(game)
        for g in iproduct([0, 1], repeat=3):
            d2 = ne_quadratic_discriminant(apply_s2_cubed(game, g))
            max_err = max(max_err, abs(d0 - d2))
    print(f"Max (S_2)^3 invariance error: {max_err:.2e}")

    game = rng.standard_normal((3, 2, 2, 2))
    d1 = ne_quadratic_discriminant(game)
    d2 = ne_quadratic_discriminant(2.0 * game)
    print(f"disc(2*game)/disc(game) = {d2/d1:.1f} (expect 2^6 = 64)")
