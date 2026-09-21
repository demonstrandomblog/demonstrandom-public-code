# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# AI-assisted experimental code; full human and mathematical review is not established.

"""Orthogonal Hodge component coordinates and quadratic invariant energies.

For square k-by-k games with the standard Euclidean payoff inner product,
potential, harmonic and nonstrategic dimensions are k^2-1, (k-1)^2 and 2k.
Nonstrategic payoffs may depend on the opponent's action, not just constants.
"""

import numpy as np
from .game_classes import validate_game
from .hodge import candogan_projectors


def build_subspace_bases(k):
    """Orthonormal bases for strategic potential, harmonic, and nonstrategic parts."""
    bases = []
    for projector in candogan_projectors(k):
        values, vectors = np.linalg.eigh(projector)
        bases.append(vectors[:, values > 0.5])
    return tuple(bases)


def component_energies(A, B):
    """Squared component norms, invariant under strategy relabeling/player exchange."""
    A, B = validate_game(A, B)
    v = np.r_[A.ravel(), B.ravel()]
    return {name: float(np.linalg.norm(P @ v)**2)
            for name, P in zip(("potential", "harmonic", "nonstrategic"),
                               candogan_projectors(len(A)))}
