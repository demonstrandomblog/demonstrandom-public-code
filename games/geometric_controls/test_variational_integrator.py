# Copyright (c) 2024-2026 Kevin T. Procopio and contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# AI-assisted experimental code; full human and mathematical review is not established.

import math
import numpy as np
import pytest
import torch
from scipy.integrate import solve_ivp
from .discrete_control_lagrange import FreeRotor, Pendulum, Symmetry, StepRecorder, VariationalIntegrator

def vector(x):
    return torch.tensor([x], dtype=torch.float64)

def test_free_rotor_analytic_trajectory_and_momentum():
    rotor = FreeRotor({'mass':1.,'length':1.})
    recorder = StepRecorder()
    vi = VariationalIntegrator(rotor, .01, on_step=recorder.on_step)
    vi.register_noether_charge('J', Symmetry(rotor.model.layout['theta'][0], vector(1)))
    a,b = vector(.2),vector(.21)
    for k in range(100):
        c,ok = vi.step(a,b)
        assert ok
        assert c.item() == pytest.approx(.22 + k*.01, abs=1e-10)
        a,b = b,c
    assert max(abs(float(r['noether_charges']['J'])-1) for r in recorder.records) < 1e-5

def test_pendulum_second_order_refinement():
    reference = solve_ivp(lambda t,y:[y[1],-9.81*np.sin(y[0])], (0,1), [.2,0],
                          rtol=1e-12, atol=1e-14, dense_output=True)
    assert reference.success
    errors = []
    for h in [.02,.01]:
        system = Pendulum({'mass':1.,'length':1.,'gravity':9.81})
        vi = VariationalIntegrator(system,h)
        a,b = vector(.2),vector(reference.sol(h)[0])
        for _ in range(round(1/h)-1):
            c,ok = vi.step(a,b)
            assert ok
            a,b = b,c
        errors.append(abs(b.item()-reference.sol(1)[0]))
    assert errors[0] < 4e-6
    assert errors[1] < errors[0]/3.5

def test_zero_iterations_reports_actual_residual():
    pendulum = Pendulum({'mass':1.,'length':1.,'gravity':9.81})
    vi = VariationalIntegrator(pendulum,.1,max_iters=0)
    c,ok = vi.step(vector(.2),vector(.2))
    # Independent midpoint pendulum DEL equation at equal previous positions.
    residual = -c.item()/.1 + .2/.1 - .1*9.81/2*(math.sin(.2)+math.sin((.2+c.item())/2))
    assert abs(residual) > .1
    assert not ok

def test_small_newton_step_is_not_a_success_certificate():
    class SteepNonlinear:
        # Mixed second derivative is large, so a tiny Newton step can leave
        # a large residual. This isolates the stopping condition.
        def discrete_lagrangian(self,a,b,h):
            return (a * torch.exp(1e12*b)).sum()
    vi = VariationalIntegrator(SteepNonlinear(),.1,max_iters=1,tol=1e-10)
    c,ok = vi.step(vector(0),vector(0))
    assert abs(c.item()) < vi.tol
    assert not ok

@pytest.mark.parametrize('h',[0,-.1,float('nan'),float('inf')])
def test_invalid_step_rejected(h):
    with pytest.raises(ValueError):
        VariationalIntegrator(FreeRotor({'mass':1.,'length':1.}),h)
