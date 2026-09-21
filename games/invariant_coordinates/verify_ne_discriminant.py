# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Experimental AI-assisted research code; human review is incomplete.
"""Test the two-player square-game equilibrium determinant under strategy relabeling.
A and B are k-by-k payoff matrices with row/column strategies, respectively.
The product of two augmented indifference-system determinants is evaluated
floating point. A nonzero value alone does not establish an interior equilibrium."""
import numpy as np
from itertools import permutations


def ne_discriminant(A, B):
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


def apply_group_element(A, B, sigma1, sigma2):
    k = A.shape[0]
    P1 = np.zeros((k, k)); P2 = np.zeros((k, k))
    for i in range(k):
        P1[i, sigma1[i]] = 1.0
        P2[i, sigma2[i]] = 1.0
    return P1 @ A @ P2.T, P1 @ B @ P2.T


if __name__ == '__main__':
    rng = np.random.default_rng(42)

    # G-invariance at k = 3
    group = list(permutations(range(3)))
    max_err = 0
    for _ in range(1000):
        A = rng.standard_normal((3, 3)); B = rng.standard_normal((3, 3))
        d0 = ne_discriminant(A, B)
        for s1 in group:
            for s2 in group:
                A2, B2 = apply_group_element(A, B, s1, s2)
                max_err = max(max_err, abs(d0 - ne_discriminant(A2, B2)))
    print(f"k=3: max invariance error = {max_err:.2e}")

    # Degree by scaling
    for k in [2, 3, 4, 5]:
        A = rng.standard_normal((k, k)); B = rng.standard_normal((k, k))
        d1 = ne_discriminant(A, B)
        d2 = ne_discriminant(2 * A, 2 * B)
        if abs(d1) > 1e-10:
            ratio = d2 / d1
            print(f"k={k}: disc(2*game)/disc(game) = {ratio:.1f} (expect 2^{2*(k-1)} = {2**(2*(k-1))})")
