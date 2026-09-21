# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# AI-assisted experimental code; full human and mathematical review is not established.

"""Best-response graphs and cycles of alternating response dynamics.

An alternating step is two moves: row player responds, then column player.
Its cycle length counts these pairs, not individual deviations. Best-response
graphs include every maximizer, including ties; alternating dynamics requires
unique maxima. Neither cycle lengths nor sampled polynomial fingerprints
establish a universal separating degree.
"""

import numpy as np
from .game_classes import validate_game
from .br_type_invariants import best_response_maps


def best_response_graph(A, B):
    """All unilateral best-response edges, including self-loops and tied maxima."""
    A, B = validate_game(A,B)
    return {(i,j): {(int(r),j) for r in np.flatnonzero(A[:,j] == A[:,j].max())}
                   | {(i,int(c)) for c in np.flatnonzero(B[i] == B[i].max())}
            for i in range(len(A)) for j in range(len(A))}


def alternating_dynamics(A, B):
    sigma, tau = best_response_maps(A,B)
    return {(i,j): (sigma[j],tau[sigma[j]])
            for i in range(len(sigma)) for j in range(len(sigma))}


def find_cycles_in_dynamics(dynamics):
    """Each directed cycle of a finite total map once, without its incoming tails."""
    if any(target not in dynamics for target in dynamics.values()):
        raise ValueError("The dynamics must map every state into its own domain")
    visited, cycles = set(), []
    for start in dynamics:
        path, positions = [], {}
        current = start
        while current not in visited and current not in positions:
            positions[current] = len(path)
            path.append(current)
            current = dynamics[current]
        if current in positions:
            cycles.append(path[positions[current]:])
        visited.update(path)
    return cycles


def analyze_br_structure(A, B):
    A,B = validate_game(A,B)
    graph = best_response_graph(A,B)
    pure_ne = [(i,j) for i in range(len(A)) for j in range(len(A))
               if A[i,j] == A[:,j].max() and B[i,j] == B[i].max()]
    cycles = find_cycles_in_dynamics(alternating_dynamics(A,B))
    return {"pure_ne": pure_ne, "alternating_cycles": cycles, "graph": graph}


def cycle_witness(A, B, profiles):
    """Product of unilateral payoff gains along a closed directed profile cycle.

    profiles lists the vertices once; the final edge returns to the first.
    This homogeneous polynomial has degree len(profiles). It is tied to
    specified labels; a single witness is not a relabeling invariant.
    """
    A,B = validate_game(A,B)
    if len(profiles) < 2:
        raise ValueError("A cycle needs at least two profiles")
    for profile in profiles:
        if len(profile) != 2 or any(not isinstance(i,(int,np.integer)) or
                                    not 0 <= i < len(A) for i in profile):
            raise ValueError("Invalid strategy profile")
    gains = []
    for (i,j),(ip,jp) in zip(profiles, list(profiles[1:])+[profiles[0]]):
        if (i == ip) == (j == jp):
            raise ValueError("Each edge must change exactly one player's action")
        gains.append(A[ip,j]-A[i,j] if j == jp else B[i,jp]-B[i,j])
    return float(np.prod(gains))
