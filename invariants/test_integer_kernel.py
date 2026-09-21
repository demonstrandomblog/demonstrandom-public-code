# AI-assisted experimental code; full human and mathematical review is not established.
from itertools import combinations, product
from math import gcd
import numpy as np
import pytest
from sympy import Matrix
from invariants.groups.torus import Torus
from invariants.groups.constructors import diagonal_torus


def certify(W):
    W = np.asarray(W, dtype=object)
    K = Torus(W).kernel_basis()
    rows, cols = W.shape
    A = Matrix(rows, cols, list(W.flat))
    B = Matrix(cols, K.shape[1], list(K.flat))
    nullity = cols - A.rank()
    assert K.shape == (cols, nullity)
    assert A * B == Matrix.zeros(rows, nullity)
    assert B.rank() == nullity
    # A full-rank integer sublattice is primitive iff the gcd of its maximal
    # minors is one. Together with rank and W*K=0 this certifies the full kernel.
    minors_gcd = 0
    for indices in combinations(range(cols), nullity):
        minors_gcd = gcd(minors_gcd, abs(int(B.extract(indices, range(nullity)).det())))
    assert minors_gcd == 1
    return K


@pytest.mark.parametrize('W', [
    [[-2, 1]], [[2, 1, 1]], [[2, 0, 1], [0, 3, 1]],
    [[0, -4, 6], [0, 8, -12]], [[0, 0, 0]], [[1, 0], [0, -1]],
    np.empty((0, 3), dtype=object), np.empty((2, 0), dtype=object),
    [[2**80 + 1, -(2**80), 1]],
])
def test_signed_dependent_empty_and_large_weights(W):
    certify(W)


def test_exhaustive_small_integer_rows():
    for weights in product(range(-2, 3), repeat=3):
        certify([weights])


def test_random_rectangular_lattices():
    rng = np.random.default_rng(2718)
    for rows in range(1, 5):
        for cols in range(1, 6):
            for _ in range(6):
                certify(rng.integers(-8, 9, (rows, cols)))


def test_no_truncation_of_fractional_weights():
    with pytest.raises(ValueError, match='integer'):
        diagonal_torus([[1.5, -1]])
