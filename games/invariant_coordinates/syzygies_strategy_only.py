# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Experimental AI-assisted research code; human review is incomplete.
"""Exact degree-bounded relations among the 17 strategy-only 2x2 generators.
The variables (rA,cA,dA,rB,cB,dB) are row, column, and interaction contrasts.
Products of the nine quadratics and eight cubics are expanded symbolically;
rational null vectors give polynomial identities among generator values."""
from sympy import symbols, expand, Poly, Matrix

rA, cA, dA, rB, cB, dB = symbols('rA cA dA rB cB dB')
VARS = [rA, cA, dA, rB, cB, dB]

I = [rA**2, rA*rB, cA**2, cA*cB, dA**2, dA*dB, rB**2, cB**2, dB**2]
J = [cA*dA*rA, cA*dB*rA, cB*dA*rA, cB*dB*rA,
     cA*dA*rB, cA*dB*rB, cB*dA*rB, cB*dB*rB]
I_NAMES = ['rA2','rArB','cA2','cAcB','dA2','dAdB','rB2','cB2','dB2']
J_NAMES = ['cAdArA','cAdBrA','cBdArA','cBdBrA',
           'cAdArB','cAdBrB','cBdArB','cBdBrB']


def find_syzygies_at_degree(target_deg):
    gens = I + J
    names = I_NAMES + J_NAMES
    degs = [2]*9 + [3]*8

    products, prod_names = [], []
    for i in range(17):
        for j in range(i, 17):
            if degs[i] + degs[j] == target_deg:
                products.append(expand(gens[i] * gens[j]))
                prod_names.append(f'{names[i]}*{names[j]}')
        for j in range(i, 17):
            for k in range(j, 17):
                if degs[i] + degs[j] + degs[k] == target_deg:
                    products.append(expand(gens[i] * gens[j] * gens[k]))
                    prod_names.append(f'{names[i]}*{names[j]}*{names[k]}')

    if not products:
        return [], [], 0
    polys = [Poly(p, VARS) for p in products]
    monos = sorted({m for p in polys for m in p.as_dict()})
    M = Matrix([[p.as_dict().get(m, 0) for p in polys] for m in monos])
    return prod_names, M.nullspace(), M.rank()


if __name__ == '__main__':
    for deg in [4, 5, 6]:
        names, null_vecs, rank = find_syzygies_at_degree(deg)
        print(f'Degree {deg}: {len(names)} products, rank {rank}, {len(null_vecs)} syzygies')
