# AI-assisted experimental code; full human and mathematical review is not established.
"""
Convenience constructors for common groups.
"""

from __future__ import annotations

import numpy as np
from itertools import permutations

from .finite import FiniteGroup
from .torus import Torus


def cyclic(n: int, dim: int = 2) -> FiniteGroup:
    """Z/nZ acting on C^dim by rotation in the first two coordinates."""
    matrices = []
    for k in range(n):
        theta = 2 * np.pi * k / n
        g = np.eye(dim)
        g[0, 0] = np.cos(theta)
        g[0, 1] = -np.sin(theta)
        g[1, 0] = np.sin(theta)
        g[1, 1] = np.cos(theta)
        matrices.append(g)
    return FiniteGroup(matrices)


def symmetric(n: int) -> FiniteGroup:
    """S_n acting on C^n by permutation matrices."""
    matrices = []
    for perm in permutations(range(n)):
        P = np.zeros((n, n))
        for i, j in enumerate(perm):
            P[i, j] = 1.0
        matrices.append(P)
    return FiniteGroup(matrices)


def dihedral(n: int) -> FiniteGroup:
    """D_n acting on C^2 by rotations and reflections."""
    matrices = []
    for k in range(n):
        theta = 2 * np.pi * k / n
        matrices.append(np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ]))
        matrices.append(np.array([
            [np.cos(theta), np.sin(theta)],
            [np.sin(theta), -np.cos(theta)]
        ]))
    return FiniteGroup(matrices)


def diagonal_torus(weights: list[list[int]]) -> Torus:
    """Torus from a weight matrix. weights[i][j] = exponent of t_i on x_j."""
    return Torus(np.array(weights, dtype=object))
