# AI-assisted experimental code; full human and mathematical review is not established.
"""Exact arithmetic and worked-example regressions for the invariant core."""

import pytest


from fractions import Fraction


import numpy as np


from invariants import poly


from invariants.action import (invariant_theory, compute_generators,
                               compute_hilbert_series, find_hsop,
                               find_secondaries, primary_secondary,
                               in_null_cone, find_separator)


from invariants.groebner import (buchberger, normal_form, in_ideal, eliminate,
                                  compute_relations, hilbert_function,
                                  has_common_root, compute_syzygies,
                                  ideal_intersection)


from invariants.orbits import (evaluate_invariants, same_image, find_separator as orb_find_separator,
                               in_null_cone as orb_in_null_cone, quotient_map, is_separating,
                               minimal_separating_subset)


from invariants.spaces import polynomial_ring


from invariants.groups.torus import Torus


from invariants.groups.finite import FiniteGroup


from invariants.groups.constructors import symmetric, cyclic, dihedral, diagonal_torus


class TestPoly:
    def test_add(self):
        f = poly.mono((1, 0))
        g = poly.mono((0, 1))
        h = poly.add(f, g)
        assert h == {(1, 0): Fraction(1), (0, 1): Fraction(1)}

    def test_add_cancellation(self):
        f = poly.mono((1, 0))
        g = poly.mono((1, 0), -1)
        assert poly.add(f, g) == {}

    def test_mul(self):
        x = poly.var(0, 2)
        y = poly.var(1, 2)
        xy = poly.mul(x, y)
        assert xy == {(1, 1): Fraction(1)}

    def test_evaluate(self):
        f = poly.add(poly.mono((2, 0)), poly.mono((0, 1)))  # x^2 + y
        assert poly.evaluate(f, (Fraction(3), Fraction(2))) == Fraction(11)

    def test_degree(self):
        f = poly.add(poly.mono((3, 1)), poly.mono((0, 2)))
        assert poly.degree(f) == 4
        assert poly.degree({}) == -1

    def test_monomials_of_degree(self):
        monos = list(poly.monomials_of_degree(3, 2))
        assert len(monos) == 6  # C(3+2-1, 2) = 6

    def test_show(self):
        f = poly.add(poly.var(0, 2), poly.var(1, 2))
        s = poly.show(f)
        assert "x0" in s and "x1" in s


class TestTorus:
    def test_c_star_on_c2_invariance(self):
        T = diagonal_torus([[1, -1]])
        ring = polynomial_ring(2)
        action = invariant_theory(T, ring)

        assert action.is_invariant(poly.mono((1, 1)))
        assert action.is_invariant(poly.mono((3, 3)))
        assert not action.is_invariant(poly.mono((1, 0)))
        assert not action.is_invariant(poly.mono((2, 1)))

    def test_c_star_on_c2_hilbert_series(self):
        T = diagonal_torus([[1, -1]])
        action = invariant_theory(T, polynomial_ring(2))
        hs = compute_hilbert_series(action, 8)
        assert hs == [1, 0, 1, 0, 1, 0, 1, 0, 1]

    def test_c_star_on_c2_hilbert_basis(self):
        T = Torus(np.array([[1, -1]]))
        hb = T.hilbert_basis(10)
        assert hb == [(1, 1)]

    def test_two_dimensional_torus(self):
        T = Torus(np.array([[1, -1, 0, 0], [0, 0, 1, -1]]))
        assert T.is_invariant_monomial((1, 1, 0, 0))
        assert T.is_invariant_monomial((0, 0, 1, 1))
        assert not T.is_invariant_monomial((1, 0, 0, 0))
        hb = T.hilbert_basis(6)
        assert set(hb) == {(1, 1, 0, 0), (0, 0, 1, 1)}

    def test_kernel_basis(self):
        T = Torus(np.array([[1, -1, 0, 0], [0, 0, 1, -1]]))
        kb = T.kernel_basis()
        assert kb.shape[1] == 2
        assert np.all(T.W @ kb == 0)


class TestFiniteGroups:
    def test_s3_hilbert_series(self):
        action = invariant_theory(symmetric(3), polynomial_ring(3))
        hs = compute_hilbert_series(action, 10)
        assert hs == [1, 1, 2, 3, 4, 5, 7, 8, 10, 12, 14]

    def test_s3_invariant_dimensions(self):
        action = invariant_theory(symmetric(3), polynomial_ring(3))
        assert len(action.invariants_of_degree(1)) == 1
        assert len(action.invariants_of_degree(2)) == 2
        assert len(action.invariants_of_degree(3)) == 3

    def test_s3_e1_is_invariant(self):
        action = invariant_theory(symmetric(3), polynomial_ring(3))
        e1 = poly.add(poly.add(poly.var(0, 3), poly.var(1, 3)), poly.var(2, 3))
        assert action.is_invariant(e1)
        assert not action.is_invariant(poly.var(0, 3))

    def test_s3_reynolds(self):
        G = symmetric(3)
        r = G.reynolds(poly.mono((2, 0, 0)))
        expected = {
            (2, 0, 0): Fraction(1, 3),
            (0, 2, 0): Fraction(1, 3),
            (0, 0, 2): Fraction(1, 3),
        }
        assert r == expected
        assert G.is_invariant(r)

    def test_s3_orbit_sum(self):
        G = symmetric(3)
        os = G.orbit_sum(poly.var(0, 3))
        expected = poly.add(poly.add(poly.var(0, 3), poly.var(1, 3)), poly.var(2, 3))
        assert os == expected
        assert G.is_invariant(os)

    def test_z3_molien(self):
        G = cyclic(3, dim=2)
        hs = G.molien_coeffs(6)
        assert hs == [1, 0, 1, 2, 1, 2, 3]

    def test_d4_hilbert_series(self):
        G = dihedral(4)
        assert G.order == 8
        action = invariant_theory(G, polynomial_ring(2))
        hs = compute_hilbert_series(action, 8)
        assert hs == [1, 0, 1, 0, 2, 0, 2, 0, 3]

    def test_s3_same_orbit(self):
        G = symmetric(3)
        v = np.array([1.0, 2.0, 3.0])
        assert G.same_orbit(v, np.array([2.0, 1.0, 3.0]))
        assert G.same_orbit(v, np.array([3.0, 1.0, 2.0]))
        assert not G.same_orbit(v, np.array([1.0, 1.0, 2.0]))


class TestGroebner:
    def test_buchberger(self):
        f1 = {(2, 0): Fraction(1), (0, 1): Fraction(1), (0, 0): Fraction(-1)}
        f2 = {(1, 1): Fraction(1), (0, 0): Fraction(1)}
        gb = buchberger([f1, f2])
        assert len(gb) == 3
        assert in_ideal(f1, gb)
        assert in_ideal(f2, gb)
        assert not in_ideal(poly.var(0, 2), gb)

    def test_elimination(self):
        f1 = {(2, 0): Fraction(1), (0, 1): Fraction(1), (0, 0): Fraction(-1)}
        f2 = {(1, 1): Fraction(1), (0, 0): Fraction(1)}
        elim = eliminate([f1, f2], k=1, n_vars=2)
        assert len(elim) == 1
        for alpha in elim[0]:
            assert alpha[0] == 0  # univariate in y

    def test_torus_relations(self):
        T = Torus(np.array([[1, 1, -2]]))
        hb = T.hilbert_basis(6)
        gens = [poly.mono(alpha) for alpha in hb]
        rels = compute_relations(gens, n_vars=3)
        assert len(rels) == 1  # y0*y2 - y1^2

    def test_s3_no_relations(self):
        action = invariant_theory(symmetric(3), polynomial_ring(3))
        gens = compute_generators(action, 3)
        assert len(gens) == 3
        rels = compute_relations(gens, n_vars=3)
        assert len(rels) == 0

    def test_ideal_intersection(self):
        x2 = poly.mono((2, 0))
        y = poly.var(1, 2)
        x = poly.var(0, 2)
        y2 = poly.mono((0, 2))
        result = ideal_intersection([x2, y], [x, y2], n_vars=2)
        lms = {poly.leading_monomial(g) for g in result}
        assert lms == {(2, 0), (1, 1), (0, 2)}

    def test_hilbert_function(self):
        f1 = {(2, 0): Fraction(1), (0, 1): Fraction(1), (0, 0): Fraction(-1)}
        f2 = {(1, 1): Fraction(1), (0, 0): Fraction(1)}
        gb = buchberger([f1, f2])
        hf = hilbert_function(gb, n_vars=2, max_d=6)
        assert hf == [1, 2, 0, 0, 0, 0, 0]
        assert sum(hf) == 3

    def test_nullstellensatz_has_root(self):
        f1 = {(2, 0): Fraction(1), (0, 1): Fraction(1), (0, 0): Fraction(-1)}
        f2 = {(1, 1): Fraction(1), (0, 0): Fraction(1)}
        assert has_common_root([f1, f2]) == True

    def test_nullstellensatz_no_root(self):
        g1 = {(1,): Fraction(1)}
        g2 = {(0,): Fraction(1), (1,): Fraction(-1)}
        assert has_common_root([g1, g2]) == False

    def test_syzygies(self):
        f1 = poly.var(0, 2)
        f2 = poly.var(1, 2)
        syz = compute_syzygies([f1, f2], n_vars=2)
        assert len(syz) == 1
        s = syz[0]
        assert (0, 1, 1, 0) in s  # y*e_0
        assert (1, 0, 0, 1) in s  # x*e_1


class TestPrimarySecondary:
    def test_s3_polynomial_ring(self):
        action = invariant_theory(symmetric(3), polynomial_ring(3))
        primaries, secondaries = primary_secondary(action, 4)
        assert len(primaries) == 3
        assert len(secondaries) == 1  # ring is polynomial

    def test_z2_non_polynomial_ring(self):
        G = FiniteGroup([np.eye(2), -np.eye(2)])
        action = invariant_theory(G, polynomial_ring(2))
        primaries, secondaries = primary_secondary(action, 4)
        assert len(primaries) == 2
        assert len(secondaries) == 2  # ring is NOT polynomial


class TestClassical:
    def test_orthogonal_o3_on_2_copies(self):
        from invariants.classical import orthogonal_action
        action = orthogonal_action(n=3, k=2)
        gens = action.invariants_of_degree(2)
        assert len(gens) == 3  # <v1,v1>, <v1,v2>, <v2,v2>
        assert action.invariants_of_degree(3) == []
        deg4 = action.invariants_of_degree(4)
        assert len(deg4) == 6  # C(3+1, 2) = 6

    def test_sl2_on_3_vectors(self):
        from invariants.classical import sl_action
        action = sl_action(n=2, k=3)
        gens = action.invariants_of_degree(2)
        assert len(gens) == 3  # [12], [13], [23]
        rels = compute_relations(gens, n_vars=6)
        assert len(rels) == 0  # freely generated

    def test_sl2_on_4_vectors_plucker(self):
        from invariants.classical import sl_action
        action = sl_action(n=2, k=4)
        gens = action.invariants_of_degree(2)
        assert len(gens) == 6  # C(4,2) = 6
        rels = compute_relations(gens, n_vars=8)
        assert len(rels) == 1  # Plucker relation

    def test_symplectic_sp2_on_3_copies(self):
        from invariants.classical import symplectic_action
        action = symplectic_action(n=1, k=3)  # Sp(2) on 3 copies of C^2
        gens = action.invariants_of_degree(2)
        assert len(gens) == 3  # omega(v_i, v_j) for i<j


class TestOrbits:
    def test_null_cone(self):
        gen = poly.mono((1, 1))
        invariants = [gen]
        assert orb_in_null_cone(invariants, (0, 0))
        assert orb_in_null_cone(invariants, (1, 0))
        assert orb_in_null_cone(invariants, (0, 5))
        assert not orb_in_null_cone(invariants, (1, 1))
        assert not orb_in_null_cone(invariants, (2, 3))

    def test_orbit_separation(self):
        action = invariant_theory(symmetric(3), polynomial_ring(3))
        gens = compute_generators(action, 3)
        # Same orbit
        assert same_image(gens, (1, 2, 3), (3, 1, 2))
        # Different orbit
        assert not same_image(gens, (1, 2, 3), (1, 1, 4))

    def test_find_separator(self):
        action = invariant_theory(symmetric(3), polynomial_ring(3))
        gens = compute_generators(action, 3)
        sep = orb_find_separator(gens, (1, 2, 3), (1, 1, 4))
        assert sep is not None

    def test_separating_invariants(self):
        G = symmetric(3)
        action = invariant_theory(G, polynomial_ring(3))
        gens = compute_generators(action, 3)
        pairs = [
            ((1, 2, 3), (1, 1, 4)),
            ((1, 2, 3), (2, 2, 2)),
            ((1, 0, 0), (0, 0, 2)),
        ]
        assert is_separating(gens, pairs)

    def test_minimal_separating_subset(self):
        G = symmetric(3)
        action = invariant_theory(G, polynomial_ring(3))
        gens = compute_generators(action, 3)
        pairs = [
            ((1, 2, 3), (1, 1, 4)),
            ((1, 2, 3), (2, 2, 2)),
            ((1, 0, 0), (0, 0, 2)),
        ]
        minimal = minimal_separating_subset(gens, pairs)
        assert is_separating(minimal, pairs)
        assert len(minimal) <= len(gens)

    def test_quotient_map(self):
        action = invariant_theory(symmetric(3), polynomial_ring(3))
        gens = compute_generators(action, 3)
        img = quotient_map(gens, (1, 2, 3))
        assert len(img) == len(gens)
        assert all(isinstance(v, Fraction) for v in img)


class TestComputeSeparatingInvariants:
    """Degree-bounded candidates must retain orbit-distinguishing invariants."""

    def test_simultaneous_sign_reversal_retains_mixed_term(self):
        from invariants.action import compute_separating_invariants

        matrices = [np.eye(2, dtype=int), -np.eye(2, dtype=int)]
        action = invariant_theory(FiniteGroup(matrices), polynomial_ring(2))
        candidates = compute_separating_invariants(action, 2)
        v, w = (1, 1), (1, -1)
        orbit = lambda point: {tuple(g @ np.array(point)) for g in matrices}

        assert orbit(v).isdisjoint(orbit(w))
        assert same_image([poly.mono((2, 0)), poly.mono((0, 2))], v, w)
        assert all(action.is_invariant(f) for f in candidates)
        assert poly.mono((1, 1)) in candidates
        assert not same_image(candidates, v, w)
        assert same_image(candidates, v, (-1, -1))

    def test_cubic_invariant_needed_after_quadratic_cutoff(self):
        from invariants.action import compute_separating_invariants

        action = invariant_theory(symmetric(3), polynomial_ring(3))
        v, w = (0, 3, 3), (1, 1, 4)
        # These unordered triples share their sum and sum of squares,
        # but their sums of cubes are 54 and 66, respectively.
        assert sorted(v) != sorted(w)
        assert same_image(compute_separating_invariants(action, 2), v, w)
        assert not same_image(compute_separating_invariants(action, 3), v, w)

    def test_zero_and_negative_degree(self):
        from invariants.action import compute_separating_invariants

        action = invariant_theory(symmetric(2), polynomial_ring(2))
        assert compute_separating_invariants(action, 0) == []
        with pytest.raises(ValueError, match="nonnegative"):
            compute_separating_invariants(action, -1)
