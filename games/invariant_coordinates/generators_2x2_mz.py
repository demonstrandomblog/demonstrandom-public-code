# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Experimental AI-assisted research code; human review is incomplete.
"""Reynolds enumeration for mean-zero two-player, two-strategy games with player exchange.
The six coordinates (rA,cA,dA,rB,cB,dB) are row, column, and interaction
contrasts. The eight-element group includes strategy relabeling and exchanging
players with transposition of their payoff matrices. Discovery uses numerical ranks."""
import numpy as np
from itertools import combinations_with_replacement
from sympy import symbols, expand


def build_d4():
    """All 8 elements of D_4 = (S_2 x S_2) >| S_2 as 6x6 matrices."""
    s1 = np.diag([-1, 1, -1, -1, 1, -1.0])
    s2 = np.diag([1, -1, -1, 1, -1, -1.0])
    sw = np.array([[0,0,0,0,1,0],[0,0,0,1,0,0],[0,0,0,0,0,1],
                   [0,1,0,0,0,0],[1,0,0,0,0,0],[0,0,1,0,0,0]], dtype=float)
    group, seen, queue = [], set(), [np.eye(6)]
    while queue:
        g = queue.pop()
        key = tuple(g.flatten().round(10))
        if key in seen:
            continue
        seen.add(key)
        group.append(g)
        queue.extend([g @ s1, g @ s2, g @ sw])
    return group


def reynolds_symbolic(exp_vec, group):
    rA, cA, dA, rB, cB, dB = symbols('rA cA dA rB cB dB')
    sym_vars = [rA, cA, dA, rB, cB, dB]
    total = 0
    for g in group:
        gv = [sum(int(g[i, j]) * sym_vars[j] for j in range(6)) for i in range(6)]
        mono = 1
        for k, e in enumerate(exp_vec):
            mono *= gv[k] ** e
        total += mono
    return expand(total / len(group))


def reynolds_numerical(exp_vec, group, points):
    vals = np.zeros(len(points))
    for g in group:
        for j, pt in enumerate(points):
            gpt = g @ pt
            v = 1.0
            for k, e in enumerate(exp_vec):
                v *= gpt[k] ** e
            vals[j] += v
    return vals / len(group)


def eval_known_generators(pt):
    """I0..I4 (deg 2) and J0..J3 (deg 3)."""
    rA, cA, dA, rB, cB, dB = pt
    I0 = dA**2 + dB**2
    I1 = rA**2 + cB**2
    I2 = cA**2 + rB**2
    I3 = dA * dB
    I4 = rA * rB + cA * cB
    J0 = rA * cA * dA + rB * cB * dB
    J1 = rA * cA * dB + rB * cB * dA
    J2 = cA * rB * (dA + dB)
    J3 = rA * cB * (dA + dB)
    return np.array([I0, I1, I2, I3, I4, J0, J1, J2, J3])


def find_new_generators(target_degree, group, n_pts=300, seed=42):
    """Numerically find degree-target generators after all lower-degree products.

    Rebuild the lower-degree generators on the same sample. Ranks are numerical
    and cannot exceed n_pts; small samples can miss independent polynomials.
    """
    if target_degree < 2:
        raise ValueError("This mean-zero action starts in degree two")
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n_pts, 6))
    known_values, known_degrees = [], []
    for degree in range(2, target_degree+1):
        products = []
        for length in range(1, degree//2+1):
            for indices in combinations_with_replacement(range(len(known_values)), length):
                if sum(known_degrees[i] for i in indices) == degree:
                    products.append(np.prod([known_values[i] for i in indices], axis=0))
        current = np.array(products) if products else np.zeros((0,n_pts))
        base_rank = np.linalg.matrix_rank(current, tol=1e-8) if products else 0
        current_rank = base_rank
        new_gens = []
        for combo in combinations_with_replacement(range(6), degree):
            exponent = tuple(combo.count(i) for i in range(6))
            values = reynolds_numerical(exponent, group, pts)
            trial = np.vstack([current, values])
            rank = np.linalg.matrix_rank(trial, tol=1e-8)
            if rank > current_rank:
                new_gens.append((exponent, reynolds_symbolic(exponent, group)))
                known_values.append(values)
                known_degrees.append(degree)
                current, current_rank = trial, rank
    return new_gens, base_rank, current_rank


if __name__ == '__main__':
    G = build_d4()
    print(f"D_4 order: {len(G)}")
    expected = {2: 5, 3: 4, 4: 8, 5: 5, 6: 0}
    for deg in [2, 3, 4, 5, 6]:
        new_gens, prod_rank, total_rank = find_new_generators(deg, G)
        print(f"Degree {deg}: products rank {prod_rank}, total rank {total_rank}, "
              f"new generators {len(new_gens)} (expect {expected[deg]})")
