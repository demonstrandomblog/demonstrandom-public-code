# AI-assisted experimental code; full human and mathematical review is not established.
"""Check algebraic identities and failure contracts beyond the worked examples."""
from fractions import Fraction
import numpy as np
import pytest
from invariants import poly
from invariants.action import invariant_theory, find_hsop, compute_hilbert_series
from invariants.classical import orthogonal_action, sl_action
from invariants.groups.constructors import cyclic
from invariants.groups.finite import FiniteGroup
from invariants.spaces import polynomial_ring
from invariants.groebner import compute_relations, compute_syzygies
from invariants.orbits import minimal_separating_subset


def test_relations_use_generator_coordinates_and_vanish():
    generators = [poly.mono((2, 0)), poly.mono((1, 1)), poly.mono((0, 2))]
    relations = compute_relations(generators, 2)
    assert len(relations) == 1
    assert all(len(alpha) == 3 for relation in relations for alpha in relation)
    for point in [(1, 2), (-3, 4), (0, 7)]:
        values = tuple(poly.evaluate(g, point) for g in generators)
        assert all(poly.evaluate(r, values) == 0 for r in relations)


def test_syzygies_vanish_as_polynomial_identities():
    x, y = poly.var(0, 2), poly.var(1, 2)
    generators = [poly.add(x, y), poly.sub(x, y), poly.mul(x, y)]
    for relation in compute_syzygies(generators, 2):
        total = {}
        for alpha, coefficient in relation.items():
            assert len(alpha) == 5 and sum(alpha[2:]) == 1
            index = alpha[2:].index(1)
            total = poly.add(total, poly.mul(poly.mono(alpha[:2], coefficient), generators[index]))
        assert total == {}


def test_classical_hilbert_count_removes_relations():
    # O(1) acts by simultaneous sign reversal on two scalar copies.
    action = orthogonal_action(1, 2)
    assert compute_hilbert_series(action, 4) == [1, 0, 3, 0, 5]


def test_constant_ring_handles_degree_zero_and_zero_polynomial():
    action = sl_action(3, 1)
    assert action.invariants_of_degree(0) == [poly.const(1, 3)]
    assert action.is_invariant({})
    assert action.is_invariant(poly.const(Fraction(2, 3), 3))
    assert list(poly.monomials_of_degree(0, 0)) == [()]


def test_incomplete_hsop_search_raises():
    action = invariant_theory(FiniteGroup([np.eye(2), -np.eye(2)]), polynomial_ring(2))
    with pytest.raises(ValueError, match='HSOP'):
        find_hsop(action, 1)


def test_unsupported_rotation_cannot_silently_produce_rational_invariants():
    with pytest.raises(NotImplementedError, match='signed permutation'):
        cyclic(3).invariants_of_degree(2)


def test_unseparable_test_pairs_are_reported():
    with pytest.raises(ValueError, match='do not separate'):
        minimal_separating_subset([poly.mono((2, 0)), poly.mono((0, 2))], [((1, 1), (1, -1))])
