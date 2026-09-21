# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# AI-assisted experimental code; full human and mathematical review is not established.

"""Fixed-step midpoint integration with explicit physical time.

A state contains spatial coordinates only. Time is passed separately; it is
not varied as an extra unknown. Consequently, time-translation diagnostics
need not be exactly conserved. Rotational momentum is conserved up to the
nonlinear solve and finite-difference errors for an invariant discrete action.
"""

import math
from typing import Callable
import torch

from .discrete_control_lagrange import (
    AttrObject, StepRecorder, VariationalSystem as SpatialSystem,
)
from .lie_groups import Rn


class VariationalSystem(SpatialSystem):
    """A system whose lagrangian(ctrl, dctrl) may depend on ctrl.t."""

    def discrete_lagrangian(self, q0, q1, h, t0=0.0):
        if "t" in self.model.layout:
            raise ValueError("Time is passed separately, not packed into the state")
        mids, velocities = {}, {}
        for name, (group, sl) in self.model.layout.items():
            g0, g1 = group.exp(q0[sl]), group.exp(q1[sl])
            velocity = group.log(g0.inverse() * g1) / h
            mids[name] = group.log(g0 * group.exp(0.5 * h * velocity))
            velocities[name] = velocity
        mids["t"] = t0 + h / 2
        velocities["t"] = 1.0
        return (h * self.lagrangian(AttrObject(mids), AttrObject(velocities))).squeeze()


class Kepler(VariationalSystem):
    """Planar L = m|v|^2/2 + mu/sqrt(|r|^2 + softening^2).

    mu is the potential-energy coefficient (m times the gravitational
    parameter). The default unsoftened model obeys exact Kepler scaling.
    """

    def __init__(self, param_values, *, softening=0.0):
        if not math.isfinite(softening) or softening < 0:
            raise ValueError("softening must be finite and nonnegative")
        if any(not math.isfinite(param_values[k]) or param_values[k] <= 0
               for k in ("mass", "mu")):
            raise ValueError("mass and mu must be positive and finite")
        self.softening = float(softening)
        super().__init__(param_values)

    def control_plane(self):
        return {"r": Rn(2)}

    def params(self):
        return ["mass", "mu"]

    def lagrangian(self, ctrl, dctrl):
        radius = torch.sqrt(ctrl.r.square().sum() + self.softening**2)
        if radius.detach().item() == 0:
            raise ValueError("The unsoftened Kepler potential is singular at r=0")
        return self.params.mass * dctrl.r.square().sum() / 2 + self.params.mu / radius


class Symmetry:
    """Space/time transformations with the explicit signature (epsilon, t, q)."""

    def __init__(self, space_transform: Callable, time_transform: Callable | None = None):
        self.space_transform = space_transform
        self.time_transform = time_transform

    def generators(self, t, q, epsilon=1e-5):
        """Centered differences; epsilon is independent of the solver tolerance."""
        omega = (self.space_transform(epsilon, t, q)
                 - self.space_transform(-epsilon, t, q)) / (2 * epsilon)
        tau = 0.0 if self.time_transform is None else (
            self.time_transform(epsilon, t, q)
            - self.time_transform(-epsilon, t, q)
        ) / (2 * epsilon)
        return omega, tau


class VariationalIntegrator:
    """Newton solve of discrete Euler-Lagrange equations at fixed physical h.

    step(q_prev, q_curr) first advances from times start_time and start_time+h.
    A failed solve returns success=False without advancing the internal clock.
    Linear solve errors propagate. Callbacks receive successes and failures.
    """

    def __init__(self, system, step_size, max_iters=25, tol=1e-10,
                 on_step=None, *, start_time=0.0, symmetry_epsilon=1e-5):
        if not math.isfinite(step_size) or step_size <= 0:
            raise ValueError("step_size must be positive and finite")
        if not math.isfinite(tol) or tol <= 0:
            raise ValueError("tol must be positive and finite")
        if not isinstance(max_iters, int) or max_iters < 0:
            raise ValueError("max_iters must be a nonnegative integer")
        if not math.isfinite(start_time):
            raise ValueError("start_time must be finite")
        if not math.isfinite(symmetry_epsilon) or symmetry_epsilon <= 0:
            raise ValueError("symmetry_epsilon must be positive and finite")
        self.system, self.h = system, float(step_size)
        self.max_iters, self.tol = max_iters, tol
        self.on_step, self.epsilon = on_step, symmetry_epsilon
        self.t_curr = start_time + self.h
        self.noether_charges = {}

    def D1_Ld(self, q0, q1, t0):
        x = q0.detach().clone().requires_grad_(True)
        value = self.system.discrete_lagrangian(x, q1, self.h, t0)
        return torch.autograd.grad(value, x, create_graph=True)[0]

    def D2_Ld(self, q0, q1, t0):
        y = q1.detach().clone().requires_grad_(True)
        value = self.system.discrete_lagrangian(q0.detach(), y, self.h, t0)
        return torch.autograd.grad(value, y)[0].detach()

    def discrete_energy(self, q0, q1, t0):
        """Endpoint energy -dLd/dh with q0, q1 and initial time held fixed."""
        h = q0.new_tensor(self.h, requires_grad=True)
        value = self.system.discrete_lagrangian(q0.detach(), q1.detach(), h, t0)
        return -torch.autograd.grad(value, h)[0].detach()

    def register_noether_charge(self, name, symmetry):
        """Register J=p.omega-E*tau; a diagnostic is not a conservation assertion."""
        self.noether_charges[name] = symmetry

    def charge(self, symmetry, q0, q1, t0):
        omega, tau = symmetry.generators(t0 + self.h, q1, self.epsilon)
        return (self.D2_Ld(q0, q1, t0) * omega).sum() - (
            self.discrete_energy(q0, q1, t0) * tau
        )

    def step(self, q_prev, q_curr):
        if q_prev.shape != (self.system.model.dim,) or q_curr.shape != q_prev.shape:
            raise ValueError("States must be flat spatial tensors matching the model")
        const = self.D2_Ld(q_prev, q_curr, self.t_curr - self.h)
        q_next = (2 * q_curr - q_prev).detach().clone().requires_grad_(True)

        def residual(x):
            return const + self.D1_Ld(q_curr, x, self.t_curr)

        for _ in range(self.max_iters):
            value = residual(q_next)
            if torch.isfinite(value).all() and value.norm().item() < self.tol:
                break
            jacobian = torch.autograd.functional.jacobian(residual, q_next)
            delta = torch.linalg.solve(jacobian, value)
            q_next = (q_next - delta).detach().requires_grad_(True)
            if delta.norm().item() < self.tol:
                break
        value = residual(q_next)
        success = bool(torch.isfinite(value).all() and value.norm().item() < self.tol)
        result = q_next.detach()
        record = {
            "q_prev": q_prev.detach().clone(), "q_curr": q_curr.detach().clone(),
            "q_next": result.clone(), "t_curr": self.t_curr,
            "t_next": self.t_curr + self.h, "success": success,
            "residual_norm": float(value.norm().detach()),
            "noether_charges": {
                name: self.charge(sym, q_curr, result, self.t_curr)
                for name, sym in self.noether_charges.items()
            },
        }
        if self.on_step is not None:
            self.on_step(record)
        if success:
            self.t_curr += self.h
        return result, success
