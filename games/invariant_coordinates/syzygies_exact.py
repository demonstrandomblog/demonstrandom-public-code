# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Experimental AI-assisted research code; human review is incomplete.
"""Exact polynomial relations among nine quadratic/cubic player-exchange invariant coordinates.
SymPy expands products in (rA,cA,dA,rB,cB,dB), the two players' row, column,
and interaction contrasts, and takes a rational nullspace at each tested degree.
The returned relations are degree-bounded, not a complete ideal certificate."""
from sympy import symbols, expand, Poly, Matrix, factor

rA, cA, dA, rB, cB, dB = symbols('rA cA dA rB cB dB')
VARS = [rA, cA, dA, rB, cB, dB]

I = [rA**2 + cB**2, rA*rB + cA*cB, cA**2 + rB**2, dA**2 + dB**2, dA*dB]
J = [cA*dA*rA + cB*dB*rB, cA*dB*rA + cB*dA*rB,
     cB*rA*(dA + dB), cA*rB*(dA + dB)]
I_NAMES = ['I0', 'I1', 'I2', 'I3', 'I4']
J_NAMES = ['J0', 'J1', 'J2', 'J3']


def find_syzygies_at_degree(target_deg):
    gens = I + J
    names = I_NAMES + J_NAMES
    degs = [2]*5 + [3]*4
    n = len(gens)

    products, prod_names = [], []
    for i in range(n):
        for j in range(i, n):
            if degs[i] + degs[j] == target_deg:
                products.append(expand(gens[i] * gens[j]))
                prod_names.append(f'{names[i]}*{names[j]}')
        for j in range(i, n):
            for k in range(j, n):
                if degs[i] + degs[j] + degs[k] == target_deg:
                    products.append(expand(gens[i] * gens[j] * gens[k]))
                    prod_names.append(f'{names[i]}*{names[j]}*{names[k]}')

    if not products:
        return [], [], 0

    polys = [Poly(p, VARS) for p in products]
    monos = sorted({m for p in polys for m in p.as_dict()})
    M = Matrix([[p.as_dict().get(m, 0) for p in polys] for m in monos])
    return prod_names, M.nullspace(), M.rank()


def format_syzygy(prod_names, vec):
    pos = [(vec[i], prod_names[i]) for i in range(len(prod_names)) if vec[i] > 0]
    neg = [(-vec[i], prod_names[i]) for i in range(len(prod_names)) if vec[i] < 0]
    lhs = ' + '.join(f'{c}*{n}' if c != 1 else n for c, n in pos)
    rhs = ' + '.join(f'{c}*{n}' if c != 1 else n for c, n in neg)
    return f'{lhs} = {rhs}'


if __name__ == '__main__':
    for deg in [4, 5, 6]:
        names, null_vecs, rank = find_syzygies_at_degree(deg)
        print(f'Degree {deg}: {len(names)} products, rank {rank}, {len(null_vecs)} syzygies')
        for idx, vec in enumerate(null_vecs):
            print(f'  S{idx+1}: {format_syzygy(names, vec)}')
    D = expand(I[0]*I[2] - I[1]**2)
    print(f'D = I0*I2 - I1^2 = {factor(D)}')
