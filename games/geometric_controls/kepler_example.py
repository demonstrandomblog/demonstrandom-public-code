# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# AI-assisted experimental code; full human and mathematical review is not established.

"""Kepler trajectory and similarity diagnostics; no figures are required.

Run python -m games.geometric_controls.kepler_example --steps 500.
The initial second position uses the articles' first-order q1=q0+h*v0.
J=p.r-(3/2)Et scales as sqrt(lambda); it is not constant along an orbit.
"""

import argparse
import json
import math
import torch
from .discrete_control_lagrange_t import Kepler, Symmetry, StepRecorder, VariationalIntegrator


def simulate(*, steps=500, step_size=0.01, scale=1.0):
    if not isinstance(steps, int) or steps < 3:
        raise ValueError("steps must be an integer at least three")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be positive and finite")
    model = Kepler({"mass": 1.0, "mu": 1.0})
    h = step_size * scale**1.5
    recorder = StepRecorder()
    solver = VariationalIntegrator(model, h, on_step=recorder.on_step)
    solver.register_noether_charge("time_translation", Symmetry(
        lambda e, t, q: q, lambda e, t, q: t + e))
    solver.register_noether_charge("angular_momentum", Symmetry(
        lambda e, t, q: torch.stack((
            math.cos(e)*q[0] - math.sin(e)*q[1],
            math.sin(e)*q[0] + math.cos(e)*q[1]))))
    solver.register_noether_charge("dynamical_similarity", Symmetry(
        lambda e, t, q: math.exp(e)*q, lambda e, t, q: math.exp(1.5*e)*t))
    q0 = torch.tensor([1.0, 0.0], dtype=torch.float64) * scale
    v0 = torch.tensor([0.0, 0.8], dtype=torch.float64) / math.sqrt(scale)
    states = [q0, q0 + h*v0]
    for _ in range(steps - 2):
        q, success = solver.step(states[-2], states[-1])
        if not success:
            raise RuntimeError("The Kepler Newton solve did not converge")
        states.append(q)
    return torch.stack(states), recorder.records


def diagnostics(steps=500, step_size=0.01, scale=2.0):
    base, records = simulate(steps=steps, step_size=step_size)
    scaled, scaled_records = simulate(steps=steps, step_size=step_size, scale=scale)
    energy = [-float(r["noether_charges"]["time_translation"]) for r in records]
    angular = [float(r["noether_charges"]["angular_momentum"]) for r in records]
    js = [float(r["noether_charges"]["dynamical_similarity"]) for r in records]
    scaled_js = [float(r["noether_charges"]["dynamical_similarity"]) for r in scaled_records]
    return {
        "steps": steps, "step_size": step_size, "scale": scale,
        "energy_range": max(energy) - min(energy),
        "angular_momentum_range": max(angular) - min(angular),
        "similarity_charge_range": max(js) - min(js),
        "position_scaling_max_error": float((scaled/scale - base).abs().max()),
        "charge_scaling_max_error": max(abs(b/math.sqrt(scale)-a) for a,b in zip(js,scaled_js)),
        "time_scaling_max_error": max(abs(s["t_next"]/scale**1.5-r["t_next"])
                                     for r,s in zip(records, scaled_records)),
        "max_solver_residual": max(r["residual_norm"] for r in records + scaled_records),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--step-size", type=float, default=0.01)
    parser.add_argument("--scale", type=float, default=2.0)
    args = parser.parse_args()
    print(json.dumps(diagnostics(args.steps, args.step_size, args.scale), indent=2))


if __name__ == "__main__":
    main()
