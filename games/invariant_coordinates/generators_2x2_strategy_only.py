# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Experimental AI-assisted research code; human review is incomplete.
"""Reynolds enumeration for two players with two strategies, keeping players fixed.
The six coordinates are (rA,cA,dA,rB,cB,dB): row, column, and interaction
contrasts of the two payoff matrices. Independent row/column swaps form S2 x S2.
Symbolic averages are exact; discovery ranks use sampled floating evaluations."""
import numpy as np
from itertools import combinations_with_replacement
from sympy import symbols, expand

rA, cA, dA, rB, cB, dB = symbols('rA cA dA rB cB dB')
VARS = [rA, cA, dA, rB, cB, dB]


def build_s2xs2():
    """S_2 x S_2 acting on (rA, cA, dA, rB, cB, dB) by row/col sign flips."""
    I = np.eye(6)
    s1 = np.diag([-1, 1, -1, -1, 1, -1.0])  # row swap
    s2 = np.diag([1, -1, -1, 1, -1, -1.0])  # col swap
    return [I, s1, s2, s1 @ s2]


def reynolds_symbolic(exp_vec, group):
    total = 0
    for g in group:
        gv = [sum(int(round(g[i, j])) * VARS[j] for j in range(6)) for i in range(6)]
        mono = 1
        for k, e in enumerate(exp_vec):
            mono *= gv[k] ** e
        total += mono
    return expand(total / len(group))


def reynolds_numerical(exp_vec, group, points):
    n = len(points)
    vals = np.zeros(n)
    for g in group:
        for j in range(n):
            gpt = g @ points[j]
            v = 1.0
            for k, e in enumerate(exp_vec):
                v *= gpt[k] ** e
            vals[j] += v
    return vals / len(group)


def find_all_generators(group, n_pts=400, seed=42, max_deg=4):
    """Numerically discover positive-degree generators through max_deg.

    Four is a sufficient bound for this group; sampled ranks remain numerical.
    """
    rng = np.random.default_rng(seed)
    pts = 2 * rng.standard_normal((n_pts, 6))
    all_num, all_sym, all_deg = [], [], []
    molien = {0: 1}

    for deg in range(1, max_deg + 1):
        mono_list = []
        for combo in combinations_with_replacement(range(6), deg):
            exp = [0] * 6
            for c in combo:
                exp[c] += 1
            if tuple(exp) not in mono_list:
                mono_list.append(tuple(exp))
        if not mono_list:
            mono_list = [(0,) * 6]

        prods = []
        for i in range(len(all_num)):
            for j in range(i, len(all_num)):
                if all_deg[i] + all_deg[j] == deg:
                    prods.append(all_num[i] * all_num[j])
            for j in range(i, len(all_num)):
                for k in range(j, len(all_num)):
                    if all_deg[i] + all_deg[j] + all_deg[k] == deg:
                        prods.append(all_num[i] * all_num[j] * all_num[k])

        prod_mat = np.array(prods) if prods else np.zeros((0, n_pts))
        current = prod_mat.copy() if len(prods) > 0 else np.zeros((0, n_pts))
        current_rank = np.linalg.matrix_rank(current, tol=1e-8) if current.shape[0] else 0

        for m in mono_list:
            rv = reynolds_numerical(m, group, pts)
            test = rv.reshape(1, -1) if current.shape[0] == 0 else np.vstack([current, rv.reshape(1, -1)])
            r = np.linalg.matrix_rank(test, tol=1e-8)
            if r > current_rank:
                all_num.append(rv)
                all_sym.append(reynolds_symbolic(m, group))
                all_deg.append(deg)
                current, current_rank = test, r

        molien[deg] = current_rank
    return all_sym, all_deg, molien


def eval_generators(rA, cA, dA, rB, cB, dB):
    """The 17 generators: 9 degree-2 (squares and cross-products), 8 degree-3 triples."""
    deg2 = [rA**2, rA*rB, cA**2, cA*cB, dA**2, dA*dB, rB**2, cB**2, dB**2]
    deg3 = [cA*dA*rA, cA*dB*rA, cB*dA*rA, cB*dB*rA,
            cA*dA*rB, cA*dB*rB, cB*dA*rB, cB*dB*rB]
    return deg2 + deg3


if __name__ == '__main__':
    G = build_s2xs2()
    gens_sym, gens_deg, molien = find_all_generators(G)
    print(f"Total generators: {len(gens_sym)}; Molien: {[molien[d] for d in range(5)]}")
    for i, (expr, deg) in enumerate(zip(gens_sym, gens_deg)):
        print(f"  g{i} (deg {deg}) = {expr}")

    rng = np.random.default_rng(42)
    pts = 2 * rng.standard_normal((400, 6))
    gen_vals = np.array([eval_generators(*pt) for pt in pts])
    for check_deg in [4, 5, 6]:
        prods = []
        for i in range(17):
            di = 2 if i < 9 else 3
            for j in range(i, 17):
                dj = 2 if j < 9 else 3
                if di + dj == check_deg:
                    prods.append(gen_vals[:, i] * gen_vals[:, j])
                for k in range(j, 17):
                    dk = 2 if k < 9 else 3
                    if di + dj + dk == check_deg:
                        prods.append(gen_vals[:, i] * gen_vals[:, j] * gen_vals[:, k])
        rank = np.linalg.matrix_rank(np.array(prods), tol=1e-8)
        expected = {4: 42, 5: 48, 6: 138}[check_deg]
        print(f"Closure deg {check_deg}: rank={rank}, Molien={expected}")
