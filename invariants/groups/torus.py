# AI-assisted experimental code; full human and mathematical review is not established.
"""
Torus actions via integer linear algebra.

A torus T = (C*)^r acts on C^m via an integer weight matrix W (r x m).
The monomial x^alpha is invariant iff W @ alpha = 0.
Invariant monomials = ker(W) ∩ N^m.

No Reynolds operator, no polynomial substitution — pure combinatorics.
"""

from __future__ import annotations

import numpy as np
from operator import index

from .. import poly
from ..poly import Poly
from ..action import Action


class Torus:
    """Torus T = (C*)^r acting on C^m via weight matrix W."""

    def __init__(self, W: np.ndarray):
        self.W = _integer_matrix(W)
        self.rank = self.W.shape[0]
        self.n_vars = self.W.shape[1]

    def monomial_weight(self, alpha: tuple[int, ...]) -> np.ndarray:
        """Total weight W @ alpha of monomial x^alpha."""
        return self.W @ np.array(alpha, dtype=object)

    def is_invariant_monomial(self, alpha: tuple[int, ...]) -> bool:
        return np.all(self.monomial_weight(alpha) == 0)

    def is_invariant(self, f: Poly) -> bool:
        """A polynomial is torus-invariant iff every monomial has weight zero."""
        return all(self.is_invariant_monomial(alpha) for alpha in f)

    def invariants_of_degree(self, d: int) -> list[Poly]:
        """Basis of invariant monomials of degree d."""
        return [
            poly.mono(alpha)
            for alpha in poly.monomials_of_degree(self.n_vars, d)
            if self.is_invariant_monomial(alpha)
        ]

    def hilbert_coeffs(self, max_d: int) -> list[int]:
        """Number of invariant monomials in each degree."""
        return [len(self.invariants_of_degree(d)) for d in range(max_d + 1)]

    def kernel_basis(self) -> np.ndarray:
        """Return a full integer-lattice basis of ker(W), stored as columns.

        The result has shape (n_vars, nullity) and dtype object with Python
        integers. Unimodular column operations preserve the whole lattice,
        including primitive solutions; intermediate arithmetic cannot overflow.
        This signed lattice basis is not a nonnegative monomial Hilbert basis.
        """
        return _integer_kernel(self.W)

    def hilbert_basis(self, max_degree: int = 20) -> list[tuple[int, ...]]:
        """Find indecomposable nonnegative kernel vectors through max_degree.

        This finite enumeration may omit higher-degree semigroup generators. A full
        Hilbert-basis claim requires an independently justified degree bound."""
        all_inv = []
        for d in range(1, max_degree + 1):
            for alpha in poly.monomials_of_degree(self.n_vars, d):
                if self.is_invariant_monomial(alpha):
                    all_inv.append(alpha)

        inv_set = set(all_inv)
        generators = []
        for alpha in all_inv:
            a = np.array(alpha)
            is_reducible = any(
                np.all(a - np.array(beta) >= 0)
                and np.any(a - np.array(beta) > 0)
                and tuple(a - np.array(beta)) in inv_set
                for beta in generators
            )
            if not is_reducible:
                generators.append(alpha)
        return generators

    def action(self, space) -> Action:
        """Connect this torus to a space, producing an Action."""
        assert space.n_vars == self.n_vars
        return Action(
            is_invariant=self.is_invariant,
            invariants_of_degree=self.invariants_of_degree,
            hilbert_coeffs=self.hilbert_coeffs,
            n_vars=self.n_vars,
        )


def _integer_matrix(W) -> np.ndarray:
    """Validate a rectangular integer matrix without truncation or overflow."""
    values = np.asarray(W, dtype=object)
    if values.ndim != 2:
        raise ValueError("Weights must be a two-dimensional integer matrix")
    try:
        return np.array([index(v) for v in values.flat], dtype=object).reshape(values.shape)
    except TypeError as exc:
        raise ValueError("Every weight must be an integer") from exc


def _bezout(a: int, b: int) -> tuple[int, int, int]:
    """Return nonnegative gcd g and coefficients s,t with s*a+t*b=g."""
    old_r, r = abs(a), abs(b)
    old_s, s, old_t, t = 1, 0, 0, 1
    while r:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
        old_t, t = t, old_t - q * t
    return old_r, old_s if a >= 0 else -old_s, old_t if b >= 0 else -old_t


def _integer_kernel(W: np.ndarray) -> np.ndarray:
    """Intersect integer hyperplanes using unimodular changes of basis.

    Start with a basis of Z^m. For each weight row, reduce its values on the
    current basis to (gcd,0,...,0). Each two-column Bezout transformation has
    determinant one, so it preserves all integer combinations. Dropping the
    first column then gives exactly the intersection with that row's kernel.
    Induction over rows proves both membership and lattice completeness.
    """
    W = _integer_matrix(W)
    basis = np.eye(W.shape[1], dtype=object)
    for row in W:
        coefficients = row @ basis
        if not any(coefficients):
            continue
        for j in range(1, basis.shape[1]):
            a, b = coefficients[0], coefficients[j]
            if b == 0:
                continue
            g, s, t = _bezout(a, b)
            first, other = basis[:, 0].copy(), basis[:, j].copy()
            basis[:, 0] = s * first + t * other
            basis[:, j] = (-b // g) * first + (a // g) * other
            coefficients[0], coefficients[j] = g, 0
        basis = basis[:, 1:]
    return basis
