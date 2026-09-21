"""Analytic and independent-ODE checks for time-aware midpoint integration."""
import math
import numpy as np
import pytest
import torch
from scipy.integrate import solve_ivp

from .discrete_control_lagrange_t import (
    VariationalSystem, VariationalIntegrator, Kepler, Symmetry, StepRecorder,
)
from .kepler_example import diagnostics
from .lie_groups import Rn


class DrivenParticle(VariationalSystem):
    def control_plane(self):
        return {"x": Rn(1)}

    def params(self):
        return []

    def lagrangian(self, ctrl, dctrl):
        return dctrl.x.square().sum()/2 + ctrl.t*ctrl.x.sum()


def tensor(values):
    return torch.tensor(values, dtype=torch.float64)


def test_explicit_time_and_nonzero_start():
    # x''=t, so x(t)=t^3/6 for these data. Midpoint DEL is exact at the grid points.
    start, h = 2.0, 0.02
    records = StepRecorder()
    solver = VariationalIntegrator(DrivenParticle({}), h, start_time=start,
                                   on_step=records.on_step)
    q0,q1 = tensor([start**3/6]),tensor([(start+h)**3/6])
    for n in range(2,22):
        q2,ok = solver.step(q0,q1)
        assert ok
        assert abs(q2.item()-(start+n*h)**3/6) < 2e-12
        q0,q1 = q1,q2
    assert records.records[0]["t_curr"] == pytest.approx(start+h)
    assert records.records[0]["t_next"] == pytest.approx(start+2*h)


def test_endpoint_energy_is_step_derivative():
    # L = v^2/2+t*x; endpoint energy = v^2/2-t_mid*x_mid-h*x_mid/2.
    model = DrivenParticle({})
    solver = VariationalIntegrator(model,0.2)
    q0,q1 = tensor([1]),tensor([1.4])
    assert solver.discrete_energy(q0,q1,3.0).item() == pytest.approx(
        2.0 - 3.1*1.2 - 0.1*1.2, abs=1e-13)


def test_kepler_refinement_against_independent_ode():
    def rhs(t,z):
        r=z[:2]
        return np.r_[z[2:],-r/np.linalg.norm(r)**3]
    reference = solve_ivp(rhs,[0,0.4],[1,0,0,0.8],rtol=2.3e-14,atol=1e-14,
                          method="DOP853",dense_output=True)
    errors=[]
    for h in (0.04,0.02):
        solver=VariationalIntegrator(Kepler({"mass":1.0,"mu":1.0}),h,tol=1e-12)
        q0,q1=tensor([1,0]),tensor(reference.sol(h)[:2])
        for _ in range(round(0.4/h)-1):
            q2,ok=solver.step(q0,q1)
            assert ok
            q0,q1=q1,q2
        errors.append(np.linalg.norm(q1.numpy()-reference.sol(0.4)[:2]))
    assert errors[1] < 1e-4
    assert errors[0]/errors[1] > 3.5


def test_kepler_scaling_and_rotational_charge():
    result=diagnostics(steps=120,step_size=0.01,scale=2.0)
    assert result["position_scaling_max_error"] < 2e-8
    assert result["charge_scaling_max_error"] < 2e-8
    assert result["time_scaling_max_error"] < 1e-12
    assert result["angular_momentum_range"] < 1e-8
    assert 0 < result["energy_range"] < 0.003
    assert result["similarity_charge_range"] > 0.1


def test_failed_solve_preserves_clock_and_records_failure():
    records=StepRecorder()
    solver=VariationalIntegrator(DrivenParticle({}),0.1,max_iters=0,on_step=records.on_step)
    _,ok=solver.step(tensor([0]),tensor([0.01]))
    assert not ok
    assert solver.t_curr == 0.1
    assert not records.records[-1]["success"]


def test_time_translation_charge_sign():
    solver=VariationalIntegrator(Kepler({"mass":1.0,"mu":1.0}),0.01)
    sym=Symmetry(lambda e,t,q:q,lambda e,t,q:t+e)
    q0,q1=tensor([1,0]),tensor([1,0.008])
    assert solver.charge(sym,q0,q1,1.0).item() == pytest.approx(
        -solver.discrete_energy(q0,q1,1.0).item(),abs=1e-10)


@pytest.mark.parametrize("bad",[0,-1,float("nan"),float("inf")])
def test_invalid_step(bad):
    with pytest.raises(ValueError):
        VariationalIntegrator(DrivenParticle({}),bad)


def test_kepler_softening_and_collision():
    model=Kepler({"mass":1.0,"mu":1.0})
    with pytest.raises(ValueError,match="singular"):
        model.discrete_lagrangian(tensor([0,0]),tensor([0,0]),0.1)
    softened=Kepler({"mass":1.0,"mu":1.0},softening=1e-5)
    assert torch.isfinite(softened.discrete_lagrangian(tensor([0,0]),tensor([0,0]),0.1))
