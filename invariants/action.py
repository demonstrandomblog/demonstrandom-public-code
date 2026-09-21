# AI-assisted experimental code; full human and mathematical review is not established.
"""
The Action: the assembled (group, space) pair that the API operates on.

An Action is a frozen bundle of closures produced by invariant_theory(group, space).
Group-specific modules provide primitives; this module provides derived operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, Optional

import numpy as np

from . import poly
from .poly import Poly


@dataclass(frozen=True)
class Action:
    """Bundle of closures encoding a specific (group, space) invariant theory problem.

    Primitives (group-specific, always provided):
        is_invariant:       f -> bool
        invariants_of_degree: d -> list of basis elements for degree-d invariants

    Optional fast paths (None if not available):
        hilbert_coeffs:     max_d -> [dim_0, dim_1, ..., dim_{max_d}]
        orbit_test:         (v, u) -> bool (same orbit?)
        apply_element:      (g, v) -> v (act on a point in V, for orbit computations)
        elements:           () -> iterable of group elements
    """
    is_invariant: Callable[[Poly], bool]
    invariants_of_degree: Callable[[int], list[Poly]]
    hilbert_coeffs: Optional[Callable[[int], list[int]]] = None
    orbit_test: Optional[Callable[[np.ndarray, np.ndarray], bool]] = None
    apply_element: Optional[Callable] = None
    elements: Optional[Callable] = None
    n_vars: int = 0
    backend: str = ""


def invariant_theory(group, space) -> Action:
    """Assemble an Action from a group and a space descriptor.

    group: a Group object (Torus, FiniteGroup, etc.) with an .action(space) method.
    space: a space descriptor returned by polynomial_ring(n), etc.
    """
    return group.action(space)


# ============================================================================
# Derived API — group-agnostic, built from Action primitives
# ============================================================================

def compute_generators(action: Action, max_degree: int) -> list[Poly]:
    """Compute invariant generators up to max_degree.

    Collects invariants degree by degree and filters out those that
    are polynomial combinations of lower-degree generators.
    """
    generators = []
    for d in range(1, max_degree + 1):
        basis_d = action.invariants_of_degree(d)
        for f in basis_d:
            if not _is_polynomial_in(f, generators):
                generators.append(f)
    return generators


def compute_hilbert_series(action: Action, max_degree: int) -> list[int]:
    """Dimension of invariant subspace in each degree.

    Uses the group-specific fast path (Molien, lattice counting) if available,
    otherwise counts basis elements from invariants_of_degree.
    """
    if action.hilbert_coeffs is not None:
        return action.hilbert_coeffs(max_degree)
    return [len(action.invariants_of_degree(d)) for d in range(max_degree + 1)]


def in_null_cone(action: Action, v: np.ndarray, max_degree: int = 6) -> bool:
    """Test vanishing of positive-degree invariants through max_degree at v.

    False proves that v is outside the null cone. True certifies membership only
    when the tested invariants define the full null cone. Float coordinates are
    approximated by rationals with denominator at most 10**9."""
    v_frac = tuple(Fraction(x).limit_denominator(10**9) for x in v)
    for d in range(1, max_degree + 1):
        for f in action.invariants_of_degree(d):
            if poly.evaluate(f, v_frac) != 0:
                return False
    return True


def same_orbit(action: Action, v: np.ndarray, u: np.ndarray) -> bool:
    """Test whether v and u lie in the same orbit."""
    if action.orbit_test is not None:
        return action.orbit_test(v, u)
    raise NotImplementedError("Orbit test not available for this action")


def find_separator(action: Action, v: np.ndarray, u: np.ndarray,
                   max_degree: int = 6) -> Optional[Poly]:
    """Find an invariant f with f(v) != f(u), or None."""
    v_frac = tuple(Fraction(x).limit_denominator(10**9) for x in v)
    u_frac = tuple(Fraction(x).limit_denominator(10**9) for x in u)
    for d in range(1, max_degree + 1):
        for f in action.invariants_of_degree(d):
            if poly.evaluate(f, v_frac) != poly.evaluate(f, u_frac):
                return f
    return None


def compute_separating_invariants(action: Action, max_degree: int) -> list[Poly]:
    """Return invariant bases through ``max_degree`` as separating candidates.

    Every polynomial invariant of degree at most ``max_degree`` is a linear
    combination of these homogeneous basis elements and a constant, provided
    ``action.invariants_of_degree`` supplies a complete basis in each degree.
    Thus any pair distinguishable by an invariant within the degree cutoff
    is distinguished by at least one returned polynomial.

    Global separation requires a sufficient degree bound. For a finite linear
    group in characteristic zero, the group order is one sufficient bound.
    For other actions, the caller must establish a bound separately. This
    function does not certify global separation or minimize the set's size.
    Degree zero returns an empty list; constants cannot distinguish points.

    A homogeneous system of parameters alone is insufficient: for the action
    (x, y) -> (-x, -y), x**2 and y**2 miss the distinct orbits of (1, 1) and
    (1, -1), while the degree-two invariant x*y distinguishes them.
    """
    if max_degree < 0:
        raise ValueError("max_degree must be nonnegative")
    return [f for degree in range(1, max_degree + 1)
            for f in action.invariants_of_degree(degree)]


def find_hsop(action: Action, max_degree: int) -> list[Poly]:
    """Search for n_vars homogeneous parameters with a zero-dimensional common locus.

    This criterion applies to finite groups on the full coordinate space. The
    search checks algebraic independence by elimination and zero dimensionality
    by pure powers in the leading ideal. It raises ValueError if the greedy
    degree-bounded search cannot complete a set. It does not certify separation.
    Elimination can be expensive even at moderate dimension or degree."""
    from . import groebner

    n = action.n_vars
    hsop: list[Poly] = []

    for d in range(1, max_degree + 1):
        for f in action.invariants_of_degree(d):
            # Must be algebraically independent of what we have
            candidate = hsop + [f]
            if groebner.compute_relations(candidate, n):
                continue
            # Must cut out a finite set of points (Hilbert function -> 0)
            gb = groebner.buchberger(candidate)
            leading = [poly.leading_monomial(g) for g in gb if g]
            zero_dimensional = all(any(
                alpha[i] > 0 and sum(e != 0 for e in alpha) == 1
                for alpha in leading) for i in range(n))
            if len(candidate) == n and not zero_dimensional:
                continue
            hsop.append(f)
            if len(hsop) == n:
                return hsop
    raise ValueError("No full HSOP found within the degree bound")


def find_secondaries(action: Action, primaries: list[Poly],
                     max_degree: int) -> list[Poly]:
    """Collect module generators through max_degree over the supplied primaries.

    The list begins with 1. A complete module decomposition requires a genuine
    HSOP and a sufficient degree bound; this routine does not certify either.
    Within the cutoff it retains invariants outside the span of existing products."""
    n = action.n_vars
    secondaries: list[Poly] = [poly.const(1, n)]
    primary_degs = [poly.degree(p) for p in primaries]

    for d in range(1, max_degree + 1):
        basis_d = action.invariants_of_degree(d)
        span = _module_span_in_degree(secondaries, primaries, primary_degs, d)
        for f in basis_d:
            if not _in_linear_span(f, span):
                secondaries.append(f)
                span.append(f)
    return secondaries


def primary_secondary(action: Action, max_degree: int
                      ) -> tuple[list[Poly], list[Poly]]:
    """Find a full parameter set and collect secondary candidates through max_degree.

    The HSOP search can raise ValueError. The secondary list is degree bounded
    and does not by itself certify a complete free-module decomposition."""
    primaries = find_hsop(action, max_degree)
    secondaries = find_secondaries(action, primaries, max_degree)
    return primaries, secondaries


# ============================================================================
# Internal
# ============================================================================

def _module_span_in_degree(secondaries: list[Poly], primaries: list[Poly],
                           primary_degs: list[int], target_deg: int
                           ) -> list[Poly]:
    """All products eta_i * theta^a with total degree = target_deg."""
    products = []
    for eta in secondaries:
        eta_deg = poly.degree(eta) if eta else 0
        remaining = target_deg - eta_deg
        if remaining < 0:
            continue
        for exps in _weighted_compositions(remaining, primary_degs):
            product = dict(eta)
            for k, e in enumerate(exps):
                for _ in range(e):
                    product = poly.mul(product, primaries[k])
            if product:
                products.append(product)
    return products


def _weighted_compositions(target: int, weights: list[int]):
    """Generate tuples (a_0,...,a_{d-1}) with sum(a_i * w_i) = target."""
    if not weights:
        if target == 0:
            yield ()
        return
    w = weights[0]
    rest = weights[1:]
    for a in range(target // w + 1 if w > 0 else 1):
        for sub in _weighted_compositions(target - a * w, rest):
            yield (a,) + sub


def _in_linear_span(target: Poly, spanning: list[Poly]) -> bool:
    """Test if target is a Fraction-linear combination of spanning polynomials."""
    if not spanning:
        return not target

    # Collect all monomials
    all_monos: set = set()
    for p in spanning:
        all_monos.update(p.keys())
    all_monos.update(target.keys())
    if not all_monos:
        return True
    monos = sorted(all_monos)

    def to_vec(p: Poly) -> list[Fraction]:
        return [p.get(m, Fraction(0)) for m in monos]

    # Row-reduce spanning vectors
    rows = [to_vec(p) for p in spanning]
    n_cols = len(monos)
    pivots: list[tuple[int, int]] = []
    pivot_row = 0
    for col in range(n_cols):
        found = None
        for r in range(pivot_row, len(rows)):
            if rows[r][col] != 0:
                found = r
                break
        if found is None:
            continue
        rows[pivot_row], rows[found] = rows[found], rows[pivot_row]
        s = rows[pivot_row][col]
        rows[pivot_row] = [x / s for x in rows[pivot_row]]
        for r in range(len(rows)):
            if r != pivot_row and rows[r][col] != 0:
                fac = rows[r][col]
                rows[r] = [rows[r][c] - fac * rows[pivot_row][c]
                           for c in range(n_cols)]
        pivots.append((pivot_row, col))
        pivot_row += 1

    # Reduce target against the pivots
    t = to_vec(target)
    for pr, col in pivots:
        if t[col] != 0:
            fac = t[col]
            t = [t[c] - fac * rows[pr][c] for c in range(n_cols)]
    return all(x == 0 for x in t)

def _is_polynomial_in(f: Poly, generators: list[Poly]) -> bool:
    """Test whether f is a polynomial expression in the generators.

    Uses Gröbner elimination: form <y_i - g_i> in k[x,y], reduce f.
    If the normal form lives in k[y] only, then f is expressible.
    """
    if not f or all(not any(alpha) for alpha in f):
        return True
    if not generators:
        return False

    from . import groebner

    n_vars = len(next(iter(f)))
    s = len(generators)
    total = n_vars + s

    # build the substitution ideal <y_i - g_i> in k[x_0,...,x_{n-1}, y_0,...,y_{s-1}]
    sub_gens = []
    for i, g in enumerate(generators):
        y_alpha = (0,) * n_vars + tuple(1 if j == i else 0 for j in range(s))
        yi = poly.mono(y_alpha)
        g_emb: Poly = {}
        for alpha, c in g.items():
            g_emb[alpha + (0,) * s] = c
        sub_gens.append(poly.sub(yi, g_emb))

    # compute Gröbner basis with elimination ordering
    elim_order = poly.elimination_order(n_vars)
    gb = groebner.buchberger(sub_gens, elim_order)

    # embed f into the larger ring and reduce
    f_emb: Poly = {}
    for alpha, c in f.items():
        f_emb[alpha + (0,) * s] = c

    nf = groebner.reduce(f_emb, gb, elim_order)

    # check if the normal form involves only y-variables
    for alpha in nf:
        if any(alpha[i] != 0 for i in range(n_vars)):
            return False
    return True
