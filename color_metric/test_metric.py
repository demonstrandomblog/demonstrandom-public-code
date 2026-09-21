# AI-assisted experimental code; full human and mathematical review is not established.
import numpy as np
import pytest
from .metric import ellipse_metric,MetricField,gamut_grid,killing_matrix,poly_basis,evaluate


def test_ellipse_unit_axes():
    a,b,theta=.02,.01,37
    g=ellipse_metric(a,b,theta)
    t=np.radians(theta)
    for v in [a*np.array([np.cos(t),np.sin(t)]),b*np.array([-np.sin(t),np.cos(t)])]:
        assert v@g@v == pytest.approx(1)
    assert np.linalg.eigvalsh(g).min()>0


def test_article_table_reproduced_and_defect_detected():
    result=evaluate('article-component')
    assert result['shape']==[996,20]
    assert result['singular_values'][0]==pytest.approx(1849601.014443969,rel=1e-9)
    assert result['singular_values'][-1]==pytest.approx(1753.2663676078628,rel=1e-9)
    assert result['nonpositive_metric_points']==160


def test_log_metric_positive_on_grid_and_difference_stencils():
    metric=MetricField()
    for x,y in gamut_grid():
        for dx,dy in [(0,0),(.005,0),(-.005,0),(0,.005),(0,-.005)]:
            assert np.linalg.eigvalsh(metric(x+dx,y+dy)).min()>0
    result=evaluate()
    assert result['nonpositive_metric_points']==0
    assert result['rank']==20


def test_euclidean_metric_has_two_translations_and_rotation():
    # Known solutions independent of the empirical color fit.
    points=np.array([(x,y) for x in (-1,0,1) for y in (-1,0,1)])
    A=killing_matrix(lambda x,y:np.eye(2),points,degree=1)
    assert np.linalg.matrix_rank(A,tol=1e-9)==3  # six coefficients, three symmetries
    for c in ([1,0,0,0,0,0],[0,0,0,1,0,0],[0,-1,0,0,0,1]):
        np.testing.assert_allclose(A@c,0,atol=1e-12)
