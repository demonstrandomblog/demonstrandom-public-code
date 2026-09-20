# AI-assisted research code; see README.md and the repository AI_NOTICE.md.
import math
import numpy as np
import pytest
from .persistence import compute_I, compute_log2_N, log2_sum_exp2, MODELS
from .fit import fit_all


def test_direct_enumeration_and_constant_weights():
    # Independently enumerate the first two complexity weights.
    weights=[math.comb(3,1)*4, math.comb(3,2)*4*3*.64]
    assert compute_log2_N(2,3,4) == pytest.approx(math.log2(sum(weights)))
    expected=-math.log2((weights[0]*.5+weights[1]*.25)/sum(weights))
    assert compute_I(2,3,4,lambda k:-k) == pytest.approx(expected)
    for bits in [0,3,1000]:
        assert compute_I(3,5,6,lambda k:-bits) == pytest.approx(bits)


def test_logsum_zero_and_large_values():
    assert log2_sum_exp2([]) == -np.inf
    assert log2_sum_exp2([-np.inf,-np.inf]) == -np.inf
    assert log2_sum_exp2([1000,1000]) == pytest.approx(1001)
    assert log2_sum_exp2(np.array([0.,0.])) == pytest.approx(1)


def test_recorded_fits_and_direct_residuals():
    results=fit_all()
    for name,expected in [('power_law',.72),('exponential',.90),('gaussian',1.35),('factorial',2.02)]:
        r=results[name]
        assert r['rmse'] == pytest.approx(expected,abs=.006)
        assert r['rmse'] == pytest.approx(np.sqrt(np.mean([(a-b)**2 for a,b in r['predictions']])))
    assert results['power_law']['params'][1] == pytest.approx(34.8,abs=.05)
