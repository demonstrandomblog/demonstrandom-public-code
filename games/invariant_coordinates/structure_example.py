# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# AI-assisted experimental code; full human and mathematical review is not established.

"""Run small game-class, Hodge, best-response, and invariant diagnostics."""
import json
import numpy as np
from .game_classes import class_diagnostics, monomial_orbits, eval_invariant
from .potential_subvariety import potential_obstruction, recover_potential
from .hodge_invariants import build_subspace_bases, component_energies
from .br_type_invariants import br_type
from .cycles import analyze_br_structure, cycle_witness
from .selection import compare_pair


def diagnostics():
    A=np.array([[1.,-1.],[-1.,1.]])
    B=-A
    response=analyze_br_structure(A,B)
    C=np.array([[3.,0.],[0.,2.]])
    phi,f,g=recover_potential(C,C)
    x=np.array([1,1,1,0,0,1])
    return {
        "matching_pennies": {
            "classes": class_diagnostics(A,B),
            "potential_obstruction": potential_obstruction(A,B),
            "hodge_component_squared_norms": component_energies(A,B),
            "best_response_type": br_type(A,B),
            "pure_nash": response["pure_ne"],
            "alternating_cycle_lengths": sorted(map(len,response["alternating_cycles"])),
            "four_deviation_witness": cycle_witness(A,B,[(0,0),(0,1),(1,1),(1,0)]),
        },
        "coordination_potential_reconstruction_error": float(max(
            np.abs(phi+f[None,:]-C).max(),np.abs(phi+g[:,None]-C).max())),
        "hodge_dimensions_k3": [V.shape[1] for V in build_subspace_bases(3)],
        "exact_quadratic_orbit_averages_k2": [
            str(eval_invariant(list(range(8)),orbit))
            for orbit in monomial_orbits(2,2).values()
        ],
        "potential_pair_quadratic_collision": compare_pair(x,-x),
        "nine_coordinate_collision": compare_pair([1,0,0,0,1,0],[np.sqrt(2),0,0,0,0,0]),
    }


def main():
    print(json.dumps(diagnostics(),indent=2))


if __name__ == "__main__":
    main()
