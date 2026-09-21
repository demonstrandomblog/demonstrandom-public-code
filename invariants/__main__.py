# AI-assisted experimental code; full human and mathematical review is not established.
"""Run bounded symmetric-group, sign-reversal, and integer-kernel examples."""
import json
import numpy as np
from . import poly
from .action import invariant_theory, compute_generators, compute_hilbert_series, compute_separating_invariants
from .groups.constructors import symmetric, diagonal_torus
from .groups.finite import FiniteGroup
from .orbits import evaluate_invariants
from .spaces import polynomial_ring


def main():
    action = invariant_theory(symmetric(3), polynomial_ring(3))
    sign = invariant_theory(FiniteGroup([np.eye(2), -np.eye(2)]), polynomial_ring(2))
    separators = compute_separating_invariants(sign, 2)
    report = {
        's3_hilbert_coefficients_through_degree_6': compute_hilbert_series(action, 6),
        's3_generators_through_degree_3': [poly.show(f) for f in compute_generators(action, 3)],
        'sign_reversal_candidates': [poly.show(f) for f in separators],
        'values_at_1_1': list(map(str, evaluate_invariants(separators, (1, 1)))),
        'values_at_1_minus1': list(map(str, evaluate_invariants(separators, (1, -1)))),
        'integer_kernel_of_minus2_1': diagonal_torus([[-2, 1]]).kernel_basis().tolist(),
    }
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
