# AI-assisted experimental code; full human and mathematical review is not established.
"""Finite group operations on explicit square matrix lists.

Exact polynomial substitution supports signed permutation matrices, including
integer entries represented as floats. Other representations raise for polynomial
operations. All group elements must be supplied; group closure is not checked.
Molien coefficients use floating eigenvalues followed by rounding, and same_orbit
uses a numerical tolerance. Those numerical diagnostics are not exact certificates."""

from __future__ import annotations

import numpy as np
from fractions import Fraction

from .. import poly
from ..poly import Poly
from ..action import Action


class FiniteGroup:
    """A finite group given by its matrix representation."""

    def __init__(self, matrices: list[np.ndarray]):
        self.matrices = matrices
        self.order = len(matrices)
        self.n_vars = matrices[0].shape[0]

    def apply_to_poly(self, g: np.ndarray, f: Poly) -> Poly:
        """(g · f)(x) = f(g^{-1} x) via variable substitution."""
        from ..spaces import _substitute_vars
        return _substitute_vars(g, f, self.n_vars)

    def reynolds(self, f: Poly) -> Poly:
        """Reynolds operator: R(f) = (1/|G|) sum_{g in G} g · f."""
        total: Poly = {}
        for g in self.matrices:
            total = poly.add(total, self.apply_to_poly(g, f))
        return poly.scale(Fraction(1, self.order), total)

    def is_invariant(self, f: Poly) -> bool:
        for g in self.matrices:
            if poly.sub(f, self.apply_to_poly(g, f)):
                return False
        return True

    def orbit_sum(self, f: Poly) -> Poly:
        """Sum over distinct images of f under G."""
        seen = set()
        total: Poly = {}
        for g in self.matrices:
            gf = self.apply_to_poly(g, f)
            key = frozenset(gf.items())
            if key not in seen:
                seen.add(key)
                total = poly.add(total, gf)
        return total

    def invariants_of_degree(self, d: int) -> list[Poly]:
        """Basis of degree-d invariants via orbit sums of monomials."""
        seen_orbits: set[frozenset] = set()
        basis = []
        for alpha in poly.monomials_of_degree(self.n_vars, d):
            f = poly.mono(alpha)
            orbit_key = frozenset(
                frozenset(self.apply_to_poly(g, f).items())
                for g in self.matrices
            )
            if orbit_key not in seen_orbits:
                seen_orbits.add(orbit_key)
                os = self.orbit_sum(f)
                if os:
                    basis.append(os)
        return basis

    def molien_coeffs(self, max_d: int) -> list[int]:
        """Molien series: H(t) = (1/|G|) sum_{g in G} 1/det(I - tg)."""
        coeffs = np.zeros(max_d + 1, dtype=complex)
        for g in self.matrices:
            eigenvalues = np.linalg.eigvals(g)
            series = np.zeros(max_d + 1, dtype=complex)
            series[0] = 1.0
            for lam in eigenvalues:
                powers = np.array([lam**d for d in range(max_d + 1)])
                new_series = np.zeros(max_d + 1, dtype=complex)
                for d in range(max_d + 1):
                    new_series[d] = sum(series[k] * powers[d - k] for k in range(d + 1))
                series = new_series
            coeffs += series
        return [round((coeffs[d] / self.order).real) for d in range(max_d + 1)]

    def same_orbit(self, v: np.ndarray, u: np.ndarray) -> bool:
        """Test whether v and u lie in the same orbit."""
        for g in self.matrices:
            if np.allclose(g @ v, u):
                return True
        return False

    def action(self, space) -> Action:
        """Connect this finite group to a space, producing an Action."""
        assert space.n_vars == self.n_vars
        return Action(
            is_invariant=self.is_invariant,
            invariants_of_degree=self.invariants_of_degree,
            hilbert_coeffs=self.molien_coeffs,
            orbit_test=self.same_orbit,
            apply_element=lambda g, v: g @ v,
            elements=lambda: iter(self.matrices),
            n_vars=self.n_vars,
        )
