"""Independent identities and relabeling checks for game-structure companions."""
from fractions import Fraction
from itertools import permutations
import numpy as np
import pytest

from .game_classes import build_group_perms, monomial_orbits, eval_invariant, class_diagnostics
from .potential_subvariety import rectangle_residuals, potential_obstruction, recover_potential
from .hodge import candogan_projectors, decompose
from .hodge_invariants import build_subspace_bases, component_energies
from .br_type_invariants import br_type, best_response_maps, br_type_properties
from .cycles import (
    best_response_graph, alternating_dynamics, find_cycles_in_dynamics,
    analyze_br_structure, cycle_witness,
)
from .selection import eval_all, eval_degree2, eval_degree3, d4_orbit, compare_pair


def relabelings(A,B):
    k=len(A)
    for rows in permutations(range(k)):
        for cols in permutations(range(k)):
            yield A[np.ix_(rows,cols)],B[np.ix_(rows,cols)]
            yield B.T[np.ix_(rows,cols)],A.T[np.ix_(rows,cols)]


@pytest.mark.parametrize("k",[2,3,4])
def test_hodge_recovers_independently_constructed_components(k):
    rng=np.random.default_rng(k)
    phi=rng.normal(size=(k,k))
    pot=np.array([phi-phi.mean(axis=0),phi-phi.mean(axis=1,keepdims=True)])
    a=rng.normal(size=(k,k))
    a=a-a.mean(axis=0)-a.mean(axis=1,keepdims=True)+a.mean()
    harm=np.array([a,-a])
    ns=np.array([np.tile(rng.normal(size=k),(k,1)),
                 np.tile(rng.normal(size=(k,1)),(1,k))])
    recovered=decompose(*(pot+harm+ns))
    for expected,actual in zip((pot,harm,ns),recovered):
        np.testing.assert_allclose(actual,expected,atol=2e-13)
    projectors=candogan_projectors(k)
    dimensions=(k*k-1,(k-1)**2,2*k)
    for P,dim in zip(projectors,dimensions):
        np.testing.assert_allclose(P@P,P,atol=2e-13)
        assert np.trace(P) == pytest.approx(dim)
    np.testing.assert_allclose(sum(projectors),np.eye(2*k*k),atol=2e-13)
    assert tuple(V.shape[1] for V in build_subspace_bases(k)) == dimensions


def test_hodge_equivariance_and_energy_partition():
    rng=np.random.default_rng(52)
    A,B=rng.normal(size=(2,3,3))
    initial=decompose(A,B)
    energies=component_energies(A,B)
    assert sum(energies.values()) == pytest.approx(np.sum(A*A+B*B))
    for Ar,Br in relabelings(A,B):
        assert component_energies(Ar,Br) == pytest.approx(energies,abs=1e-12)
        assert potential_obstruction(Ar,Br) == pytest.approx(potential_obstruction(A,B))
    v=np.r_[A.ravel(),B.ravel()]
    for perm in build_group_perms(3):
        inverse=np.argsort(perm)
        for P in candogan_projectors(3):
            np.testing.assert_allclose(P@v[inverse],(P@v)[inverse],atol=3e-13)


def test_potential_condition_is_additive_not_determinantal():
    # Additive D has nonzero determinant; a generic rank-one D fails the curl test.
    A=np.array([[1,2],[3,4]])
    B=np.zeros((2,2))
    assert np.linalg.det(A) != 0
    assert potential_obstruction(A,B) == 0
    phi,f,g=recover_potential(A,B)
    np.testing.assert_allclose(phi+f[None,:],A)
    np.testing.assert_allclose(phi+g[:,None],B)
    rank_one=np.array([[1,2],[2,4]])
    assert np.linalg.det(rank_one) == 0
    assert potential_obstruction(rank_one,B) == 1
    with pytest.raises(ValueError):
        recover_potential(rank_one,B)


@pytest.mark.parametrize("k",[2,3])
def test_exact_reynolds_average_matches_full_group_sum(k):
    group=build_group_perms(k)
    v=list(map(Fraction,range(2*k*k)))
    for degree in (1,2):
        for rep,orbit in monomial_orbits(k,degree).items():
            expected=sum((np.prod([v[g[i]] for i in rep]) for g in group),Fraction(0))/len(group)
            assert eval_invariant(v,orbit,len(group)) == expected
    linear_orbit=next(iter(monomial_orbits(k,1).values()))
    assert eval_invariant(v,linear_orbit) == sum(v)/len(v)


def test_br_canonicalization_handles_nonbijective_maps_and_player_swap():
    A=np.array([[9,8,0],[1,2,9],[0,0,1]])
    B=np.array([[9,1,0],[8,0,1],[0,9,1]])
    sigma,tau=best_response_maps(A,B)
    assert len(set(sigma)) < 3 and len(set(tau)) < 3
    expected=br_type(A,B)
    for Ar,Br in relabelings(A,B):
        assert br_type(Ar,Br) == expected


def test_ties_are_preserved_in_graph_and_rejected_by_unique_map_api():
    A=B=np.zeros((2,2))
    assert best_response_graph(A,B)[(0,0)] == {(0,0),(1,0),(0,1)}
    with pytest.raises(ValueError,match="tied"):
        br_type(A,B)
    with pytest.raises(ValueError,match="tied"):
        alternating_dynamics(A,B)


def test_functional_cycles_do_not_include_transient_tails():
    assert find_cycles_in_dynamics({0:1,1:2,2:1,3:0,4:4}) == [[1,2],[4]]
    assert br_type_properties((0,1,2),(1,2,1))["cycle_structure"] == (2,)
    with pytest.raises(ValueError):
        find_cycles_in_dynamics({0:1})


def test_matching_pennies_cycle_and_witness():
    A=np.array([[1,-1],[-1,1]])
    B=-A
    result=analyze_br_structure(A,B)
    assert result["pure_ne"] == []
    assert sorted(map(len,result["alternating_cycles"])) == [2]
    vertices=[(0,0),(0,1),(1,1),(1,0)]
    assert cycle_witness(A,B,vertices) == 16
    assert cycle_witness(3*A,3*B,vertices) == 3**4*16
    with pytest.raises(ValueError):
        cycle_witness(A,B,[(0,0),(1,1)])


def test_low_degree_coordinates_are_invariant():
    x=(1,2,3,4,5,6)
    for y in d4_orbit(*x):
        np.testing.assert_array_equal(eval_all(*x),eval_all(*y))


def test_potential_games_can_need_cubic_statistics():
    x=np.array([1,1,1,0,0,1])
    result=compare_pair(x,-x)
    assert result == {"same_d4_orbit":False,"quadratics_equal":True,"cubics_equal":False}


def test_nine_coordinates_do_not_separate_all_orbits():
    x=[1,0,0,0,1,0]
    y=[np.sqrt(2),0,0,0,0,0]
    assert compare_pair(x,y) == {
        "same_d4_orbit":False,"quadratics_equal":True,"cubics_equal":True,
    }


def test_literal_game_classes_and_dominance():
    A=np.array([[3,3],[0,0]])
    B=np.array([[2,0],[2,0]])
    d=class_diagnostics(A,B)
    assert d["potential"]
    assert d["strictly_dominant_rows"] == [0]
    assert d["strictly_dominant_columns"] == [0]
    assert not d["zero_sum"]
    assert not d["common_interest"]


@pytest.mark.parametrize("bad",[np.zeros((2,3)),np.array([[0,np.nan],[1,2]])])
def test_invalid_game(bad):
    with pytest.raises(ValueError):
        decompose(bad,bad)
