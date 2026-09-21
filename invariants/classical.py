# AI-assisted experimental code; full human and mathematical review is not established.
"""
Classical group actions via First Fundamental Theorems.

Instead of computing invariants from scratch, we hard-code the known
generators (inner products, brackets, symplectic pairings) and build
higher-degree invariants as products.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations, combinations_with_replacement

from . import poly
from .poly import Poly
from .action import Action


def orthogonal_action(n: int, k: int) -> Action:
    """O(n) acting diagonally on k copies of C^n.

    Generators (FFT): inner products <v_i, v_j> for 0 <= i <= j < k.
    Each generator has degree 2 in the n*k original variables.
    """
    n_vars = n * k
    gens = _inner_products(n, k)

    def invariants_of_degree(d: int) -> list[Poly]:
        if d == 0:
            return [poly.const(1, n_vars)]
        if d % 2 != 0 or d < 0:
            return []
        return _products_of_degree(gens, d // 2)

    def is_invariant(f: Poly) -> bool:
        # Check by reducing: f is invariant iff it's a polynomial in the generators
        from . import groebner
        from .action import _is_polynomial_in
        return _is_polynomial_in(f, gens)

    return Action(
        is_invariant=is_invariant,
        invariants_of_degree=invariants_of_degree,
        n_vars=n_vars,
    )


def sl_action(n: int, k: int) -> Action:
    """SL(n) acting diagonally on k copies of C^n.

    Generators (FFT): n x n minors [i_1, ..., i_n] for 0 <= i_1 < ... < i_n < k.
    Each generator has degree n in the n*k original variables.
    """
    n_vars = n * k
    gens = _brackets(n, k)

    def invariants_of_degree(d: int) -> list[Poly]:
        if d == 0:
            return [poly.const(1, n_vars)]
        if d % n != 0 or d < 0:
            return []
        return _products_of_degree(gens, d // n)

    def is_invariant(f: Poly) -> bool:
        from .action import _is_polynomial_in
        return _is_polynomial_in(f, gens)

    return Action(
        is_invariant=is_invariant,
        invariants_of_degree=invariants_of_degree,
        n_vars=n_vars,
    )


def symplectic_action(n: int, k: int) -> Action:
    """Sp(2n) acting diagonally on k copies of C^{2n}.

    Generators (FFT): symplectic pairings omega(v_i, v_j) for 0 <= i < j < k.
    Each generator has degree 2 in the 2n*k original variables.
    """
    n_vars = 2 * n * k
    gens = _symplectic_pairings(n, k)

    def invariants_of_degree(d: int) -> list[Poly]:
        if d == 0:
            return [poly.const(1, n_vars)]
        if d % 2 != 0 or d < 0:
            return []
        return _products_of_degree(gens, d // 2)

    def is_invariant(f: Poly) -> bool:
        from .action import _is_polynomial_in
        return _is_polynomial_in(f, gens)

    return Action(
        is_invariant=is_invariant,
        invariants_of_degree=invariants_of_degree,
        n_vars=n_vars,
    )


# ============================================================================
# Generator construction
# ============================================================================

def _inner_products(n: int, k: int) -> list[Poly]:
    """Inner products <v_i, v_j> for O(n) on k copies of C^n."""
    n_vars = n * k
    gens = []
    for i, j in combinations_with_replacement(range(k), 2):
        # <v_i, v_j> = sum_a x_{i*n+a} * x_{j*n+a}
        f: Poly = {}
        for a in range(n):
            xi = (0,) * n_vars
            xj = (0,) * n_vars
            xi = tuple(1 if idx == i * n + a else 0 for idx in range(n_vars))
            xj = tuple(1 if idx == j * n + a else 0 for idx in range(n_vars))
            term = poly.mono(tuple(xi[idx] + xj[idx] for idx in range(n_vars)))
            f = poly.add(f, term)
        gens.append(f)
    return gens


def _brackets(n: int, k: int) -> list[Poly]:
    """n x n minors (bracket invariants) for SL(n) on k copies of C^n."""
    n_vars = n * k
    gens = []
    for indices in combinations(range(k), n):
        # det of the n x n matrix with columns v_{i_1}, ..., v_{i_n}
        f = _determinant(n, k, indices)
        if f:
            gens.append(f)
    return gens


def _symplectic_pairings(n: int, k: int) -> list[Poly]:
    """Symplectic pairings omega(v_i, v_j) for Sp(2n) on k copies of C^{2n}."""
    dim = 2 * n
    n_vars = dim * k
    gens = []
    for i, j in combinations(range(k), 2):
        # omega(v_i, v_j) = sum_a (v_i^a * v_j^{n+a} - v_i^{n+a} * v_j^a)
        f: Poly = {}
        for a in range(n):
            # v_i^a * v_j^{n+a}
            alpha_pos = tuple(
                (1 if idx == i * dim + a else 0) +
                (1 if idx == j * dim + n + a else 0)
                for idx in range(n_vars)
            )
            # v_i^{n+a} * v_j^a
            alpha_neg = tuple(
                (1 if idx == i * dim + n + a else 0) +
                (1 if idx == j * dim + a else 0)
                for idx in range(n_vars)
            )
            f = poly.add(f, poly.mono(alpha_pos, Fraction(1)))
            f = poly.sub(f, poly.mono(alpha_neg, Fraction(1)))
        gens.append(f)
    return gens


def _determinant(n: int, k: int, indices: tuple) -> Poly:
    """Compute the n x n determinant of columns v_{i_1}, ..., v_{i_n}."""
    n_vars = n * k

    # Expand determinant via Leibniz formula
    from itertools import permutations as perms
    f: Poly = {}
    for perm in perms(range(n)):
        # sign of permutation
        sign = _perm_sign(perm)
        # product of entries: row a, column indices[perm[a]]
        alpha = [0] * n_vars
        for a in range(n):
            col = indices[perm[a]]
            alpha[col * n + a] += 1
        term = poly.mono(tuple(alpha), Fraction(sign))
        f = poly.add(f, term)
    return f


def _perm_sign(perm: tuple) -> int:
    """Sign of a permutation: +1 for even, -1 for odd."""
    n = len(perm)
    visited = [False] * n
    sign = 1
    for i in range(n):
        if visited[i]:
            continue
        j = i
        cycle_len = 0
        while not visited[j]:
            visited[j] = True
            j = perm[j]
            cycle_len += 1
        if cycle_len % 2 == 0:
            sign *= -1
    return sign


# ============================================================================
# Product enumeration
# ============================================================================

def _products_of_degree(generators: list[Poly], num_factors: int) -> list[Poly]:
    """Return a linearly independent subset of products of num_factors generators.

    Polynomial relations can make distinct products dependent. Exact coefficient
    elimination removes these dependencies before a graded dimension is counted."""
    if num_factors == 0:
        if not generators:
            return []
        n_vars = len(next(iter(generators[0])))
        return [poly.const(1, n_vars)]

    from .action import _in_linear_span
    products = []
    for combo in combinations_with_replacement(range(len(generators)), num_factors):
        product = generators[combo[0]]
        for idx in combo[1:]:
            product = poly.mul(product, generators[idx])
        if not _in_linear_span(product, products):
            products.append(product)
    return products
