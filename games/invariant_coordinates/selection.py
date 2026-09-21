# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# AI-assisted experimental code; full human and mathematical review is not established.

"""Low-degree D4 invariants and finite pair comparisons for 2x2 games.

The five quadratics and four cubics are nine coordinates of a larger
generating set. They do not form a complete generating/separating set.
Coordinates are (rA,cA,dA,rB,cB,dB), the unnormalized row, column and
interaction contrasts of each mean-zero payoff matrix. Comparisons retain
payoff scale and coordinate order.
"""

import numpy as np


def eval_degree2(rA,cA,dA,rB,cB,dB):
    return np.array([dA*dA+dB*dB, rA*rA+cB*cB, cA*cA+rB*rB,
                     dA*dB, rA*rB+cA*cB])


def eval_degree3(rA,cA,dA,rB,cB,dB):
    return np.array([rA*cA*dA+rB*cB*dB, rA*cA*dB+rB*cB*dA,
                     cA*rB*(dA+dB), rA*cB*(dA+dB)])


def eval_all(*coordinates):
    """The nine listed low-degree coordinates, in fixed order."""
    return np.r_[eval_degree2(*coordinates),eval_degree3(*coordinates)]


def d4_orbit(rA,cA,dA,rB,cB,dB):
    """Exact sign/swap images of supplied coordinates; no decimal rounding."""
    images = set()
    for sr in (1,-1):
        for sc in (1,-1):
            sd = sr*sc
            images.add((sr*rA,sc*cA,sd*dA,sr*rB,sc*cB,sd*dB))
            images.add((sr*cB,sc*rB,sd*dB,sr*cA,sc*rA,sd*dA))
    return sorted(images)


def compare_pair(x, y, atol=1e-10):
    """Report a finite-pair diagnostic; no global separation claim is made."""
    x,y = np.asarray(x,dtype=float),np.asarray(y,dtype=float)
    if x.shape != (6,) or y.shape != (6,) or not np.isfinite([x,y]).all():
        raise ValueError("Expected two finite six-coordinate games")
    if not np.isfinite(atol) or atol < 0:
        raise ValueError("atol must be finite and nonnegative")
    return {
        "same_d4_orbit": any(np.allclose(g,y,atol=atol,rtol=0) for g in d4_orbit(*x)),
        "quadratics_equal": bool(np.allclose(eval_degree2(*x),eval_degree2(*y),atol=atol,rtol=0)),
        "cubics_equal": bool(np.allclose(eval_degree3(*x),eval_degree3(*y),atol=atol,rtol=0)),
    }
