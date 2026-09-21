# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Experimental AI-assisted research code; human review is incomplete.
"""Molien coefficients for n players with two strategies and player exchange.
Bipartitions describe conjugacy classes of the signed permutation group.
The payoff representation has n*2**n coordinates; mean_zero removes one
constant-payoff coordinate per player. Character averaging uses exact integer arithmetic."""
from collections import Counter
from math import factorial


def partitions(n, max_val=None):
    if max_val is None:
        max_val = n
    if n == 0:
        yield ()
        return
    for k in range(min(n, max_val), 0, -1):
        for rest in partitions(n - k, k):
            yield (k,) + rest


def bipartitions(n):
    """Conjugacy classes of Z_2 wr S_n are bipartitions (alpha, beta) of n."""
    for a in range(n + 1):
        for alpha in partitions(a):
            for beta in partitions(n - a):
                yield alpha, beta


def centralizer_order(alpha, beta):
    result = 1
    for parts in [alpha, beta]:
        for k, m in Counter(parts).items():
            result *= (2 * k) ** m * factorial(m)
    return result


def class_size(alpha, beta, n):
    return factorial(n) * 2 ** n // centralizer_order(alpha, beta)


def build_representative(alpha, beta, n):
    """Induced permutation on n * 2^n coordinates for bipartition (alpha, beta)."""
    num_profiles = 2 ** n
    dim = n * num_profiles
    sigma = list(range(n))
    epsilon = [0] * n
    pos = 0
    for k in alpha:
        for i in range(k - 1):
            sigma[pos + i] = pos + i + 1
        sigma[pos + k - 1] = pos
        pos += k
    for k in beta:
        for i in range(k - 1):
            sigma[pos + i] = pos + i + 1
        sigma[pos + k - 1] = pos
        epsilon[pos] = 1
        pos += k

    inv_sigma = [0] * n
    for i in range(n):
        inv_sigma[sigma[i]] = i

    perm = [0] * dim
    for p in range(n):
        for prof_int in range(num_profiles):
            prof = [(prof_int >> (n - 1 - q)) & 1 for q in range(n)]
            src_prof = [0] * n
            for q in range(n):
                sq = inv_sigma[q]
                src_prof[sq] = prof[q] ^ epsilon[sq]
            src_int = 0
            for q in range(n):
                src_int = (src_int << 1) | src_prof[q]
            perm[p * num_profiles + prof_int] = inv_sigma[p] * num_profiles + src_int
    return perm


def build_player_permutation(alpha, beta, n):
    sigma = list(range(n))
    pos = 0
    for parts in [alpha, beta]:
        for k in parts:
            for i in range(k - 1):
                sigma[pos + i] = pos + i + 1
            sigma[pos + k - 1] = pos
            pos += k
    return sigma


def fixed_points_of_power(perm, power):
    count = 0
    for i in range(len(perm)):
        j = i
        for _ in range(power):
            j = perm[j]
        if j == i:
            count += 1
    return count


def molien_conj(n, max_degree, mean_zero=False):
    """Molien series via Newton's identity over conjugacy classes."""
    group_order = factorial(n) * 2 ** n
    result = [0] * (max_degree + 1)
    for alpha, beta in bipartitions(n):
        csize = class_size(alpha, beta, n)
        perm = build_representative(alpha, beta, n)
        player_perm = build_player_permutation(alpha, beta, n) if mean_zero else None
        p = []
        for k in range(max_degree + 1):
            pk = fixed_points_of_power(perm, k)
            if mean_zero:
                pk -= fixed_points_of_power(player_perm, k)
            p.append(pk)
        h = [0] * (max_degree + 1)
        h[0] = 1
        for d in range(1, max_degree + 1):
            h[d] = sum(p[k] * h[d - k] for k in range(1, d + 1)) // d
        for d in range(max_degree + 1):
            result[d] += csize * h[d]
    return [r // group_order for r in result]


if __name__ == '__main__':
    print("n  full Molien (deg 0..3)    mean-zero Molien (deg 0..3)")
    for n in range(2, 9):
        full = molien_conj(n, 3, mean_zero=False)
        mz = molien_conj(n, 3, mean_zero=True)
        print(f"{n}  {full}    {mz}")
    print("\nDegree-2 formulas: full = 5n - 3, mean-zero = 5(n - 1)")
    for n in range(2, 9):
        full2 = molien_conj(n, 2)[2]
        mz2 = molien_conj(n, 2, mean_zero=True)[2]
        print(f"  n={n}: full={full2} (5n-3={5*n-3}), mz={mz2} (5(n-1)={5*(n-1)})")
