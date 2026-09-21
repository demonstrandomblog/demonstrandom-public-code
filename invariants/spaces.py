# AI-assisted experimental code; full human and mathematical review is not established.
"""Polynomial space descriptors and signed-permutation substitutions.

The Space record contains coordinate count and arithmetic callbacks. The finite
group backend operates on sparse polynomials directly. Its exact substitution
path accepts signed permutation matrices only."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Callable

from . import poly
from .poly import Poly


@dataclass(frozen=True)
class Space:
    """A vector space with a linear group action.

    n_vars:         dimension of the underlying coordinate space
    apply_matrix:   (g, element) -> element — how a matrix g acts on a space element
    add:            (a, b) -> a + b
    scale:          (c, a) -> c * a
    zero:           () -> zero element
    """
    n_vars: int
    apply_matrix: Callable
    add: Callable
    scale: Callable
    zero: Callable


def polynomial_ring(n_vars: int) -> Space:
    """The polynomial ring k[x_0, ..., x_{n-1}].

    Group elements (matrices) act by variable substitution:
      (g · f)(x) = f(g^{-1} x)
    """
    return Space(
        n_vars=n_vars,
        apply_matrix=lambda g, f: _substitute_vars(g, f, n_vars),
        add=poly.add,
        scale=poly.scale,
        zero=lambda: {},
    )


def _substitute_vars(g: 'np.ndarray', f: Poly, n_vars: int) -> Poly:
    """Compute f(g^{-1} x) by substituting variables.

    Each x_i is replaced by the linear combination sum_j (g^{-1})_{i,j} x_j.
    """
    import numpy as np

    values = np.asarray(g, dtype=float)
    rounded = np.rint(values)
    if (values.shape != (n_vars, n_vars) or not np.isfinite(values).all()
            or not np.allclose(values, rounded, rtol=0, atol=1e-12)
            or not np.isin(rounded, [-1, 0, 1]).all()
            or not np.all(np.count_nonzero(rounded, axis=0) == 1)
            or not np.all(np.count_nonzero(rounded, axis=1) == 1)):
        raise NotImplementedError("Exact substitution supports signed permutation matrices only")
    g_inv = rounded.astype(int).T

    sub_polys = []
    for i in range(n_vars):
        row_poly: Poly = {}
        for j in range(n_vars):
            c = Fraction(g_inv[i, j]).limit_denominator(10**9)
            if c != 0:
                exp = tuple(1 if k == j else 0 for k in range(n_vars))
                row_poly[exp] = c
        sub_polys.append(row_poly)

    result: Poly = {}
    for alpha, coeff in f.items():
        term = poly.const(coeff, n_vars)
        for i, e in enumerate(alpha):
            for _ in range(e):
                term = poly.mul(term, sub_polys[i])
        result = poly.add(result, term)
    return result
