# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# AI-assisted experimental code; full human and mathematical review is not established.

"""Potential-game constraints in payoff coordinates.

For square bimatrix games, A-B must be additive in row and column indices:
D[i,j]+D[i',j']-D[i,j']-D[i',j]=0. These are additive rectangle differences,
not determinant minors. Their sum of squares is a degree-two invariant.
"""

from itertools import combinations
import numpy as np
from .game_classes import validate_game, validate_k


def rectangle_residuals(A, B):
    A, B = validate_game(A, B)
    D = A-B
    return np.array([D[i,j]+D[ip,jp]-D[i,jp]-D[ip,j]
                     for i,ip in combinations(range(len(A)),2)
                     for j,jp in combinations(range(len(A)),2)])


def potential_obstruction(A, B):
    """Squared norm of all rectangle residuals; zero iff potential over the reals."""
    residual = rectangle_residuals(A, B)
    return float(residual @ residual)


def is_potential_game(game, k, tol=1e-8):
    k = validate_k(k)
    game = np.asarray(game, dtype=float)
    if game.shape != (2*k*k,) or not np.isfinite(tol) or tol < 0:
        raise ValueError("Expected a flat length-2k^2 game and finite nonnegative tol")
    A, B = game.reshape(2,k,k)
    return bool(np.all(np.abs(rectangle_residuals(A,B)) <= tol))


def recover_potential(A, B, tol=1e-8):
    """Return Phi, f, g with A=Phi+f[None,:], B=Phi+g[:,None].

    Raises when the rectangle constraints exceed tol. The chosen gauge
    sets g[0]=0. For floating inputs, reconstruction is within tolerance.
    """
    A, B = validate_game(A, B)
    if not is_potential_game(np.r_[A.ravel(), B.ravel()], len(A), tol):
        raise ValueError("The game is not potential at the requested tolerance")
    D = A-B
    f = D[0].copy()
    g = D[0,0]-D[:,0]
    return A-f[None,:], f, g
