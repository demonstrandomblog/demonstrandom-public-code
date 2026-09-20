# AI-assisted research code; see README.md and the repository AI_NOTICE.md.
from itertools import product
import math
import numpy as np
import pytest
from .hamming_radius_plot import hamming_ball_volume, min_radius_for_covering, fractional_overlap
from .weighted_hamming_radius_plot import integer_weights, weighted_cumulative_counts, weighted_ball_volume, min_weighted_radius_for_covering


def test_hamming_counts_against_all_binary_strings():
    for k in range(1,9):
        counts=[sum(bits) for bits in product((0,1),repeat=k)]
        for r in range(k+1):
            assert hamming_ball_volume(k,r) == sum(c<=r for c in counts)
    assert hamming_ball_volume(4,-1) == 0


def test_published_example_and_integer_boundaries():
    assert min_radius_for_covering(50,5000000) == 8
    assert fractional_overlap(50,5000000) == .84
    assert min_radius_for_covering(np.int64(100),1) == 100
    assert min_radius_for_covering(100,1<<100) == 0
    assert min_radius_for_covering(np.int64(100),np.int64(1_000_000)) == min_radius_for_covering(100,1_000_000)
    for k in (5,20,50):
        for N in (1,100,1000000):
            r=min_radius_for_covering(k,N)
            assert hamming_ball_volume(k,r)*N >= 2**k
            assert r == 0 or hamming_ball_volume(k,r-1)*N < 2**k


@pytest.mark.parametrize('beta',[0,.5,1,2])
def test_weighted_counts_by_exhaustive_enumeration(beta):
    k,scale=7,50
    weights=integer_weights(k,beta,scale)
    distances=[sum(w*b for w,b in zip(weights,bits)) for bits in product((0,1),repeat=k)]
    cumulative=weighted_cumulative_counts(k,beta,scale)
    assert cumulative == [sum(d<=r for d in distances) for r in range(sum(weights)+1)]
    assert weighted_ball_volume(k,beta,scale,-1) == 0
    for N in (1,3,128):
        r=min_weighted_radius_for_covering(k,N,beta,scale)
        assert cumulative[r]*N >= 2**k
        assert r==0 or cumulative[r-1]*N < 2**k
