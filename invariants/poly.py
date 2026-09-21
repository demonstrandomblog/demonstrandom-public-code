# AI-assisted experimental code; full human and mathematical review is not established.
"""
Polynomial arithmetic over exact coefficients.

A polynomial is dict[tuple[int,...], Fraction]:
  - Keys are exponent tuples (alpha_0, ..., alpha_{n-1})
  - Values are nonzero coefficients
  - The zero polynomial is {}
"""

from __future__ import annotations
from fractions import Fraction


Poly = dict[tuple[int, ...], Fraction]


# -- Orderings: alpha -> comparable key --

def grlex(alpha: tuple[int, ...]) -> tuple:
    return (sum(alpha), alpha)

def lex(alpha: tuple[int, ...]) -> tuple:
    return alpha

def elimination_order(k: int):
    def order(alpha: tuple[int, ...]) -> tuple:
        return (alpha[:k], sum(alpha[k:]), alpha[k:])
    return order


# -- Arithmetic --

def add(f: Poly, g: Poly) -> Poly:
    result = dict(f)
    for alpha, c in g.items():
        result[alpha] = result.get(alpha, Fraction(0)) + c
        if result[alpha] == 0:
            del result[alpha]
    return result

def neg(f: Poly) -> Poly:
    return {alpha: -c for alpha, c in f.items()}

def sub(f: Poly, g: Poly) -> Poly:
    return add(f, neg(g))

def mul(f: Poly, g: Poly) -> Poly:
    result: Poly = {}
    for a1, c1 in f.items():
        for a2, c2 in g.items():
            alpha = tuple(e1 + e2 for e1, e2 in zip(a1, a2))
            result[alpha] = result.get(alpha, Fraction(0)) + c1 * c2
            if result[alpha] == 0:
                del result[alpha]
    return result

def scale(c, f: Poly) -> Poly:
    c = Fraction(c)
    if c == 0:
        return {}
    return {alpha: coeff * c for alpha, coeff in f.items()}

def evaluate(f: Poly, point: tuple) -> Fraction:
    result = Fraction(0)
    for alpha, c in f.items():
        term = c
        for xi, ei in zip(point, alpha):
            term *= Fraction(xi) ** ei
        result += term
    return result

def degree(f: Poly) -> int:
    if not f:
        return -1
    return max(sum(alpha) for alpha in f)


# -- Leading term operations --

def leading_monomial(f: Poly, order=grlex) -> tuple[int, ...]:
    return max(f.keys(), key=order)

def leading_coefficient(f: Poly, order=grlex) -> Fraction:
    return f[leading_monomial(f, order)]


# -- Constructors --

def mono(alpha: tuple[int, ...], c=1) -> Poly:
    c = Fraction(c)
    if c == 0:
        return {}
    return {alpha: c}

def const(value, n_vars: int) -> Poly:
    c = Fraction(value)
    if c == 0:
        return {}
    return {tuple(0 for _ in range(n_vars)): c}

def var(i: int, n_vars: int) -> Poly:
    return mono(tuple(1 if j == i else 0 for j in range(n_vars)))


# -- Monomial operations --

def mono_divides(a: tuple[int, ...], b: tuple[int, ...]) -> bool:
    return all(ai <= bi for ai, bi in zip(a, b))

def mono_mul(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(ai + bi for ai, bi in zip(a, b))

def mono_div(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(ai - bi for ai, bi in zip(a, b))

def mono_lcm(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(max(ai, bi) for ai, bi in zip(a, b))


# -- Enumeration --

def monomials_of_degree(n_vars: int, degree: int):
    if n_vars < 0 or degree < 0:
        return
    if n_vars == 0:
        if degree == 0:
            yield ()
        return
    if n_vars == 1:
        yield (degree,)
        return
    for i in range(degree + 1):
        for rest in monomials_of_degree(n_vars - 1, degree - i):
            yield (i,) + rest


# -- Display --

def show(f: Poly, order=grlex) -> str:
    if not f:
        return "0"
    parts = []
    for alpha in sorted(f.keys(), key=order, reverse=True):
        c = f[alpha]
        d = sum(alpha)
        if d == 0:
            parts.append(str(c))
        else:
            vars_str = "*".join(
                f"x{i}" if e == 1 else f"x{i}^{e}"
                for i, e in enumerate(alpha) if e > 0
            )
            if c == 1:
                parts.append(vars_str)
            elif c == -1:
                parts.append(f"-{vars_str}")
            else:
                parts.append(f"{c}*{vars_str}")
    return " + ".join(parts).replace(" + -", " - ")
