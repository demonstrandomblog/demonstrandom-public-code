# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Experimental AI-assisted research code; human review is incomplete.
"""Numerical scaling diagnostics for equilibrium indifference equations.
For two players, A and B are square payoff matrices. For n players with two
strategies, game.shape is (n,)+(2,)*n and x[p] is probability of strategy 0.
ne_disc_n2 returns the Jacobian determinant at one numerically found algebraic
root; it is not a polynomial discriminant and the root need not be a probability
vector. Uniform payoff scaling multiplies each Jacobian row, giving degree n."""
import math
import numpy as np
from itertools import product as iproduct
from scipy.optimize import fsolve


def random_n2_game(n, rng):
    return rng.standard_normal((n,) + (2,) * n)


def indiff_coeffs_n2(game, player):
    n = game.shape[0]
    others = [q for q in range(n) if q != player]
    D = np.zeros((2,) * len(others))
    for s in iproduct([0, 1], repeat=len(others)):
        idx0 = [0] * n; idx1 = [0] * n
        idx0[player], idx1[player] = 0, 1
        for i, q in enumerate(others):
            idx0[q] = idx1[q] = s[i]
        D[s] = game[player][tuple(idx0)] - game[player][tuple(idx1)]
    return D, others


def eval_indiff(D, others, x):
    val = 0.0
    for s in iproduct([0, 1], repeat=len(others)):
        w = 1.0
        for i, q in enumerate(others):
            w *= x[q] if s[i] == 0 else (1 - x[q])
        val += D[s] * w
    return val


def ne_disc_n2(game):
    n = game.shape[0]
    Ds = [indiff_coeffs_n2(game, p) for p in range(n)]

    def system(x):
        return [eval_indiff(D, others, x) for D, others in Ds]

    sol, _, ier, _ = fsolve(system, np.full(n, 0.5), full_output=True)
    if max(abs(v) for v in system(sol)) > 1e-8:
        return None
    eps = 1e-7
    J = np.zeros((n, n))
    f0 = system(sol)
    for j in range(n):
        x2 = sol.copy(); x2[j] += eps
        f1 = system(x2)
        for i in range(n):
            J[i, j] = (f1[i] - f0[i]) / eps
    return np.linalg.det(J)


def ne_disc_2k(A, B):
    k = A.shape[0]
    M_A = np.zeros((k, k))
    for i in range(k - 1):
        M_A[i, :] = A[0, :] - A[i + 1, :]
    M_A[k - 1, :] = 1.0
    M_B = np.zeros((k, k))
    for j in range(k - 1):
        M_B[:, j] = B[:, 0] - B[:, j + 1]
    M_B[:, k - 1] = 1.0
    return np.linalg.det(M_A) * np.linalg.det(M_B)


if __name__ == '__main__':
    rng = np.random.default_rng(42)
    print(f"{'(n,k)':<8} {'predicted':<10} {'measured':<10}")
    for k in [2, 3, 4, 5]:
        predicted = 2 * (k - 1)
        A = rng.standard_normal((k, k)); B = rng.standard_normal((k, k))
        d1 = ne_disc_2k(A, B)
        d2 = ne_disc_2k(2 * A, 2 * B)
        deg = round(math.log2(abs(d2 / d1)))
        print(f"(2,{k})    {predicted:<10} {deg:<10}")
    for n in [2, 3, 4, 5]:
        predicted = n  # Jacobian determinant at a fixed indifference root.
        ratios = []
        for _ in range(200):
            game = random_n2_game(n, rng)
            d1 = ne_disc_n2(game)
            d2 = ne_disc_n2(2 * game)
            if d1 is not None and d2 is not None and abs(d1) > 1e-10:
                ratios.append(d2 / d1)
        deg = round(math.log2(abs(np.median(ratios))))
        print(f"({n},2)    {predicted:<10} {deg:<10}")
