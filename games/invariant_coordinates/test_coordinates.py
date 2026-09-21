# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Exact identities, independent enumeration, and bounded numerical regressions."""
from fractions import Fraction
from itertools import permutations, product, combinations_with_replacement
import numpy as np
import pytest
import sympy as sp
from . import rg_from_invariants as ordinal
from . import syzygies_strategy_only as relations
from . import contrast_blocks_3x3 as contrast
from . import orbit_separation_3x3 as fingerprint
from . import candidate_atlases as stored
from .molien_3x3_strategy_only import molien_33_strategy_only
from .ne_disc_degree import ne_disc_n2
from .ne_disc_3_2_solve import ne_quadratic_discriminant, get_bilinear_coeffs
from .verify_ne_discriminant import ne_discriminant, apply_group_element

PERMS = list(permutations(range(3)))


def test_all_strict_ordinal_games_and_invariant_values():
    counts = {}
    for a in permutations(range(4)):
        for b in permutations(range(4)):
            key = ordinal.rg_type(a,b)
            counts[key] = counts.get(key,0)+1
            u = np.array([a,b]).reshape(2,2,2)
            values = ordinal.eval_generators(*ordinal.mean_zero(*a,*b))
            orbit = []
            for row, col in product(((0,1),(1,0)),repeat=2):
                moved = u[:,row][:,:,col].reshape(2,4)
                assert ordinal.eval_generators(*ordinal.mean_zero(*moved.flat)) == values
                orbit.append(tuple(map(tuple,moved)))
            assert key == min(orbit)
    assert len(counts)==144 and set(counts.values())=={4}
    assert len(ordinal.enumerate_rg_types())==144
    assert ordinal.rg_type([0,0,1,2],[0,1,2,3]) is None


def test_seventeen_generators_span_every_invariant_monomial_through_degree_six():
    exponents=[sp.Poly(g,relations.VARS).monoms()[0] for g in relations.I+relations.J]
    reached={(0,)*6}
    for degree in range(1,7):
        for alpha in list(reached):
            for beta in exponents:
                gamma=tuple(a+b for a,b in zip(alpha,beta))
                if sum(gamma)<=degree:
                    reached.add(gamma)
        invariant=set()
        for indices in combinations_with_replacement(range(6),degree):
            alpha=tuple(indices.count(i) for i in range(6))
            if (alpha[0]+alpha[2]+alpha[3]+alpha[5])%2==0 and (alpha[1]+alpha[2]+alpha[4]+alpha[5])%2==0:
                invariant.add(alpha)
        assert {alpha for alpha in reached if sum(alpha)==degree}==invariant


def test_exact_quartic_relations():
    names,vectors,rank=relations.find_syzygies_at_degree(4)
    lookup=dict(zip(relations.I_NAMES+relations.J_NAMES,relations.I+relations.J))
    products=[sp.prod(lookup[factor] for factor in name.split('*')) for name in names]
    assert rank==42 and len(vectors)==3
    for vector in vectors:
        assert sp.expand(sum(c*p for c,p in zip(vector,products)))==0


def test_molien_against_explicit_group_character_average():
    # Independently count fixed strategy profiles under each power of each element.
    totals=[Fraction(0)]*5
    for sigmas in product(PERMS,repeat=3):
        characters=[]
        for power in range(1,5):
            fixed=1
            for sigma in sigmas:
                current=tuple(range(3))
                for _ in range(power):
                    current=tuple(sigma[x] for x in current)
                fixed*=sum(current[i]==i for i in range(3))
            characters.append(3*fixed-3)
        h=[Fraction(1)]
        for degree in range(1,5):
            h.append(sum(characters[k-1]*h[degree-k] for k in range(1,degree+1))/degree)
        totals=[a+b for a,b in zip(totals,h)]
    expected=[int(v/216) for v in totals]
    assert expected==[1,0,42,556,9057]
    assert molien_33_strategy_only(4)==expected


def test_contrast_reconstruction_and_orthogonality():
    game=np.random.default_rng(50).normal(size=(3,3,3,3))
    centered=contrast.mean_zero_payoff(game)
    for p in range(3):
        blocks=contrast.all_contrast_blocks(centered[p])
        np.testing.assert_allclose(sum(blocks.values()),centered[p],atol=1e-14)
        for a,b in combinations_with_replacement(blocks,2):
            if a!=b:
                assert abs(np.sum(blocks[a]*blocks[b]))<1e-13


def test_fingerprint_all_group_elements_and_actual_degrees():
    game=np.random.default_rng(51).normal(size=(3,3,3,3))
    values=fingerprint.fingerprint(game)
    assert values.shape==(93,)
    for sigmas in product(PERMS,repeat=3):
        moved=fingerprint.apply_group_element(game,*sigmas)
        np.testing.assert_allclose(fingerprint.fingerprint(moved),values,atol=1e-10,rtol=1e-10)
    expected=values*np.array([4]*42+[8]*49+[64]*2)
    np.testing.assert_allclose(fingerprint.fingerprint(2*game),expected,atol=1e-9)
    np.testing.assert_allclose(values[72:81],0,atol=1e-13)
    np.testing.assert_allclose(fingerprint.fingerprint(game+np.arange(3)[:,None,None,None]),values,atol=1e-10)


@pytest.mark.parametrize('n',[2,3,4])
def test_jacobian_at_known_root_scales_with_number_of_players(n):
    game=np.zeros((n,)+(2,)*n)
    for p in range(n):
        for profile in product((0,1),repeat=n):
            game[(p,)+profile]=(1 if profile[p]==0 else -1)*(1 if profile[(p+1)%n]==0 else -1)/2
    assert ne_disc_n2(game)==pytest.approx((-1)**(n-1)*2**n,rel=1e-7)
    assert ne_disc_n2(2*game)/ne_disc_n2(game)==pytest.approx(2**n,rel=1e-7)


def test_bilinear_elimination_discriminant_against_symbolic_resultants():
    game=np.random.default_rng(2).integers(-3,4,size=(3,2,2,2))
    x=sp.symbols('x0:3')
    equations=[]
    for p,(a,b,c,d) in enumerate(get_bilinear_coeffs(game)):
        q,r=[i for i in range(3) if i!=p]
        equations.append(int(a)+int(b)*x[q]+int(c)*x[r]+int(d)*x[q]*x[r])
    eliminated=sp.resultant(equations[1],equations[2],x[0])
    quadratic=sp.resultant(equations[0],eliminated,x[1])
    expected=sp.discriminant(quadratic,x[2])
    assert ne_quadratic_discriminant(game)==float(expected)


def test_square_game_determinant_all_relabelings():
    rng=np.random.default_rng(55)
    a,b=rng.normal(size=(2,3,3))
    value=ne_discriminant(a,b)
    for row,col in product(PERMS,repeat=2):
        moved=apply_group_element(a,b,row,col)
        assert ne_discriminant(*moved)==pytest.approx(value,abs=1e-12)
    assert ne_discriminant(2*a,2*b)==pytest.approx(16*value)


def test_sparse_evaluation_constants_zeros_signs_and_chunking():
    atlas={'indices':np.array([[0,0,78,78],[1,2,78,78],[78,78,78,78]]),
           'numerator':np.array([3,-2,7]),'denominator':np.array([2,1,1]),
           'offsets':np.array([0,2,3])}
    points=np.zeros((3,78));points[:,:3]=[[0,3,4],[-2,-3,4],[2,0,-7]]
    expected=np.column_stack((1.5*points[:,0]**2-2*points[:,1]*points[:,2],np.full(3,7)))
    for chunk in (1,2,4096):
        np.testing.assert_array_equal(stored.evaluate(atlas,points,chunk),expected)


def test_game_coordinates_remove_means_and_reconstruct():
    game=np.random.default_rng(52).normal(size=(3,3,3,3))
    point=stored.game_to_point(game).reshape(3,26)
    full=np.column_stack((point,-point.sum(axis=1)))
    np.testing.assert_allclose(full,contrast.mean_zero_payoff(game).reshape(3,27),atol=1e-14)
    np.testing.assert_allclose(stored.game_to_point(game+np.arange(3)[:,None,None,None]),point.ravel(),atol=1e-14)


@pytest.mark.parametrize('name,count',[('atlas598',598),('polarized302',299),('polarized1456',1453)])
def test_stored_atlas_schema_hash_counts_and_degrees(name,count):
    atlas=stored.load_atlas(name)
    assert len(atlas['offsets'])==count+1
    degrees=np.sum(atlas['indices']<78,axis=1)
    observed={}
    for a,b in zip(atlas['offsets'][:-1],atlas['offsets'][1:]):
        degree=int(degrees[a])
        assert np.all(degrees[a:b]==degree)
        observed[str(degree)]=observed.get(str(degree),0)+1
    record=next(r for r in stored.manifest()['atlases'] if r['name']==name)
    assert observed==record['degree_counts']

def test_player_exchange_degree_six_uses_quartic_times_quadratic_products():
    from .generators_2x2_mz import build_d4, find_new_generators
    generators, product_rank, rank = find_new_generators(6,build_d4(),n_pts=120)
    assert generators == []
    assert product_rank == rank == 71
