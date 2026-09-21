# AI-assisted experimental code; full human and mathematical review is not established.
"""
Orbit separation, null cone, and separating invariants.

These functions operate on lists of invariant polynomials (Poly objects)
and use poly.evaluate to test separation. They are group-agnostic: given
invariants, they answer geometric questions about orbits and quotients.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Optional

from . import poly
from .poly import Poly


def evaluate_invariants(invariants: list[Poly], point: tuple) -> list[Fraction]:
    """Evaluate all invariants at a point. Returns the image in quotient space."""
    point_frac = tuple(Fraction(x) for x in point)
    return [poly.evaluate(f, point_frac) for f in invariants]


def same_image(invariants: list[Poly], v: tuple, w: tuple) -> bool:
    """Test whether v and w have the same image under the invariant map.

    If invariants generate the full ring, this tests orbit-closure equivalence.
    If invariants are merely separating, this tests orbit separation.
    """
    return evaluate_invariants(invariants, v) == evaluate_invariants(invariants, w)


def find_separator(invariants: list[Poly], v: tuple, w: tuple) -> Optional[Poly]:
    """Find an invariant that separates v from w, or None if none does."""
    v_frac = tuple(Fraction(x) for x in v)
    w_frac = tuple(Fraction(x) for x in w)
    for f in invariants:
        if poly.evaluate(f, v_frac) != poly.evaluate(f, w_frac):
            return f
    return None


def in_null_cone(invariants: list[Poly], point: tuple) -> bool:
    """Test whether a point lies in the null cone.

    The null cone is {v : f(v) = 0 for all homogeneous invariants of positive degree}.
    Equivalently, v is in the null cone iff its orbit closure contains the origin.
    """
    point_frac = tuple(Fraction(x) for x in point)
    for f in invariants:
        if poly.degree(f) > 0 and poly.evaluate(f, point_frac) != 0:
            return False
    return True


def quotient_map(invariants: list[Poly], point: tuple) -> list[Fraction]:
    """Evaluate a supplied list of invariant polynomials at a point.

    These coordinates represent the full quotient only when the list generates
    the invariant ring. A degree-bounded candidate list may identify distinct orbits."""
    return evaluate_invariants(invariants, point)


def is_separating(invariants: list[Poly],
                  test_pairs: list[tuple[tuple, tuple]]) -> bool:
    """Test whether invariants separate all given pairs of non-equivalent points.

    A separating set S distinguishes orbits: for every pair (v, w) in different
    orbits, some f in S has f(v) != f(w).
    """
    for v, w in test_pairs:
        if same_image(invariants, v, w):
            return False  # failed to separate this pair
    return True


def minimal_separating_subset(invariants: list[Poly],
                              test_pairs: list[tuple[tuple, tuple]]) -> list[Poly]:
    """Greedily cover the supplied point pairs with distinguishing invariants.

    The set-cover heuristic does not guarantee minimum cardinality or global
    separation. Raise ValueError if the candidates cannot distinguish every pair."""
    remaining = list(range(len(test_pairs)))
    selected: list[Poly] = []

    while remaining:
        best_f = None
        best_separated: list[int] = []
        for f in invariants:
            if f in selected:
                continue
            separated = []
            for idx in remaining:
                v, w = test_pairs[idx]
                v_frac = tuple(Fraction(x) for x in v)
                w_frac = tuple(Fraction(x) for x in w)
                if poly.evaluate(f, v_frac) != poly.evaluate(f, w_frac):
                    separated.append(idx)
            if len(separated) > len(best_separated):
                best_f = f
                best_separated = separated
        if best_f is None:
            raise ValueError("Candidates do not separate all supplied point pairs")
        selected.append(best_f)
        remaining = [i for i in remaining if i not in best_separated]

    return selected
