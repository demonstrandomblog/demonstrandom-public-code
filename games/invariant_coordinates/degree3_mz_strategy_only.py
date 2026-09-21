# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Experimental AI-assisted research code; human review is incomplete.
"""Count cubic invariants for n players with two strategies, keeping players fixed.
Each player's payoff mean is removed. Three equivalent combinatorial formulas
count the invariant cubic monomials under independent strategy swaps."""
from fractions import Fraction


def h3_mz_n2_closed_form(n: int) -> int:
    """Closed form from collapsing the Molien average over (S_2)^n."""
    M = n * (2 ** n - 1)
    g_order = 2 ** n
    id_term = M * (M + 1) * (M + 2)
    non_id_term = (g_order - 1) * (-n) * (n * n + 3 * M + 2)
    total = Fraction(id_term + non_id_term, 6 * g_order)
    assert total.denominator == 1
    return total.numerator


def h3_mz_n2_triple_count(n: int) -> int:
    """Count zero-XOR triples of nonzero binary contrast labels:
    h_3(n,2) = n^3 * (2^n - 1)(2^n - 2) / 6."""
    return n ** 3 * (2 ** n - 1) * (2 ** n - 2) // 6


def h3_mz_n2_explicit(n: int) -> int:
    """Explicit average over all 2^n group elements."""
    N = n * (2 ** n)
    g_order = 2 ** n
    total = Fraction(0)
    for mask in range(g_order):
        n_swaps = bin(mask).count("1")
        chi = N if n_swaps == 0 else 0
        chi_sq = N        # g^2 = identity for any g in (S_2)^n
        chi_cu = N if n_swaps == 0 else 0
        chi_mz = chi - n
        chi_sq_mz = chi_sq - n
        chi_cu_mz = chi_cu - n
        total += Fraction(chi_mz ** 3 + 3 * chi_mz * chi_sq_mz + 2 * chi_cu_mz, 6)
    total /= g_order
    assert total.denominator == 1
    return total.numerator


if __name__ == '__main__':
    print(f"{'n':>3}  {'dim_mz':>8}  {'|G|':>6}  {'h_3':>10}")
    for n in range(2, 7):
        M = n * (2 ** n - 1)
        h_closed = h3_mz_n2_closed_form(n)
        h_triple = h3_mz_n2_triple_count(n)
        h_explicit = h3_mz_n2_explicit(n)
        assert h_closed == h_triple == h_explicit
        print(f"{n:>3}  {M:>8}  {2**n:>6}  {h_closed:>10}")
