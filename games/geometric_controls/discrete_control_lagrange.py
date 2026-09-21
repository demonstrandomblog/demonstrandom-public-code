# Copyright (c) 2024-2026 Kevin T. Procopio and contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# AI-assisted experimental code; full human and mathematical review is not established.

"""Midpoint variational integration for pendulum and free-rotor models.

Flat configuration tensors follow the order of each model's control plane.
The discrete Lagrangian is h times the Lagrangian at the group midpoint and
relative velocity. Given two positions, Newton's method solves
D2 Ld(q_prev,q,h) + D1 Ld(q,q_next,h) = 0. step returns a detached position
and a final-residual success flag. Use float64 at the default tolerance.
"""
from dataclasses import dataclass
from typing import Callable, Optional
import math
import torch

import matplotlib.pyplot as plt

from .lie_groups import LieGroup, SOn, Rn


@dataclass(frozen=True)
class StateHandle:
    name: str
    group: LieGroup

class AttrObject:
    def __init__(self, mapping):
        for k, v in mapping.items():
            setattr(self, k, v)

class DiscreteModel:
    def __init__(self, control_plane):
        self.vars = control_plane

        offset = 0
        layout = {}
        for name, group in self.vars.items():
            d = group.dim
            layout[name] = (group, slice(offset, offset+d))
            offset += d

        self.layout = layout
        self.dim = offset

    def unpack(self, q):
        out = {}
        for name, (_, sl) in self.layout.items():
            out[name] = q[sl]
        return AttrObject(out)

    def pack(self, ctrl):
        parts = []
        for name, (_, _) in self.layout.items():
            parts.append(getattr(ctrl, name).reshape(-1))
        return torch.cat(parts)

class VariationalSystem:
    def __init__(self, param_values):

        # Build params object
        names = self.params()
        self.params = AttrObject({name: param_values[name] for name in names})

        # Build model
        cp = self.control_plane()
        self.model = DiscreteModel(cp)

    def discrete_lagrangian(self, qk: torch.Tensor, qk1: torch.Tensor, h: float) -> torch.Tensor:

        mid_coords = []
        vel_coords = []

        for name, (group, sl) in self.model.layout.items():
            qk_i  = qk[sl]
            qk1_i = qk1[sl]

            gk  = group.exp(qk_i)
            gk1 = group.exp(qk1_i)

            omega = group.log(gk.inverse() * gk1) / h
            g_mid = gk * group.exp(0.5 * h * omega) # equivalent of midpoint quadrature

            mid_i = group.log(g_mid)
            vel_i = omega

            mid_coords.append(mid_i)
            vel_coords.append(vel_i)

        mid = torch.cat(mid_coords, dim=-1)
        vel = torch.cat(vel_coords, dim=-1)

        ctrl  = self.model.unpack(mid)
        dctrl = self.model.unpack(vel)

        return h * self.lagrangian(ctrl, dctrl)

    def control_plane(self): raise NotImplementedError
    def params(self): raise NotImplementedError
    def lagrangian(self, ctrl, dctrl): raise NotImplementedError

class Pendulum(VariationalSystem):
    def control_plane(self):
        return {
            "theta": SOn(2)
        }

    def params(self):
        return ["mass", "length", "gravity"]

    def lagrangian(self, ctrl, dctrl):
        th  = ctrl.theta
        thd = dctrl.theta

        m = self.params.mass
        l = self.params.length
        g = self.params.gravity

        T = 0.5 * m * l*l * thd*thd
        V = m * g * l * (1 - torch.cos(th))
        return T - V

class FreeRotor(VariationalSystem):
    def control_plane(self):
        return {
            "theta": Rn(1)
        }

    def params(self):
        return ["mass", "length"]

    def lagrangian(self, ctrl, dctrl):
        th  = ctrl.theta
        thd = dctrl.theta

        m = self.params.mass
        l = self.params.length

        T = 0.5 * m * l*l * thd*thd
        return T

@dataclass
class Symmetry:

    def __init__(self, group: LieGroup, generator: torch.Tensor):
        self.group = group
        self.generator = generator

    def exp(self, v: torch.Tensor):
        return self.group.exp(v)

    def log(self, g):
        return self.group.log(g)

    def shift(self, q: torch.Tensor, eps: float) -> torch.Tensor:
        g_q   = self.group.exp(q)
        g_eps = self.group.exp(eps * self.generator.to(q.device))
        g_new = g_eps * g_q
        return self.group.log(g_new)

class VariationalIntegrator:

    def __init__(self, system: VariationalSystem, step_size: float,
                 max_iters: int = 25, tol: float = 1e-10, on_step: Optional[Callable] = None):
        self.system = system
        self.h = float(step_size)
        if not math.isfinite(self.h) or self.h <= 0:
            raise ValueError("step_size must be positive and finite")
        if not isinstance(max_iters, int) or max_iters < 0:
            raise ValueError("max_iters must be a nonnegative integer")
        if not math.isfinite(tol) or tol <= 0:
            raise ValueError("tol must be positive and finite")
        self.max_iters = max_iters
        self.tol = tol
        self.on_step = on_step
        self.noether_charges = {}

    def D1_Ld(self, qk: torch.Tensor, qk1: torch.Tensor) -> torch.Tensor:
        qk_var = qk.clone().detach().requires_grad_(True)
        Ld = self.system.discrete_lagrangian(qk_var, qk1, self.h)
        grad_qk, = torch.autograd.grad(Ld, qk_var, create_graph=True)
        return grad_qk

    def D2_Ld(self, qk: torch.Tensor, qk1: torch.Tensor) -> torch.Tensor:
        qk_var = qk.clone().detach().requires_grad_(True)
        qk1_var = qk1.clone().detach().requires_grad_(True)
        Ld = self.system.discrete_lagrangian(qk_var, qk1_var, self.h)
        grad_qk1, = torch.autograd.grad(Ld, qk1_var, create_graph=False)
        return grad_qk1.detach()

    def register_noether_charge(self, name:str, symmetry: Symmetry):
        def _new_charge(qk, qk1):
            p = self.D2_Ld(qk, qk1)

            qk1_eps = symmetry.log(symmetry.exp(self.tol * symmetry.generator) * symmetry.exp(qk1))
            omega_qk1 = (qk1_eps - qk1)/self.tol

            return (p*omega_qk1).sum()
        self.noether_charges[name] = _new_charge

    def step(self, q_prev: torch.Tensor, q_curr: torch.Tensor):

        q_next = q_curr + (q_curr - q_prev)
        q_next = q_next.clone().detach().requires_grad_(True)
        const_term = self.D2_Ld(q_prev, q_curr).detach()

        success = False

        for _ in range(self.max_iters):
            F = const_term + self.D1_Ld(q_curr, q_next)

            if F.norm().item() < self.tol:
                success = True
                break

            def F_of(x: torch.Tensor) -> torch.Tensor:
                return const_term + self.D1_Ld(q_curr, x)

            J = torch.autograd.functional.jacobian(F_of, q_next)
            delta = torch.linalg.solve(J, F)

            with torch.no_grad():
                q_next -= delta
            q_next.requires_grad_(True)

            if delta.norm().item() < self.tol:
                break

        # A small Newton step alone does not establish equation convergence.
        residual = const_term + self.D1_Ld(q_curr, q_next)
        success = bool(torch.isfinite(residual).all() and residual.norm().item() < self.tol)
        q_prev_det = q_prev.detach()
        q_curr_det = q_curr.detach()
        q_next_det = q_next.detach()

        noether_charge_outputs = {
            name: fn(q_curr_det, q_next_det)
            for name, fn in self.noether_charges.items()
        }

        if self.on_step is not None:
            self.on_step({
                "q_prev": q_prev_det,
                "q_curr": q_curr_det,
                "q_next": q_next_det,
                "noether_charges": noether_charge_outputs,
                "success": success
            })

        return q_next_det, success

class StepRecorder:
    def __init__(self):
        self.records = []

    def on_step(self, record):
        self.records.append(record)

def pendulum_observables_from_records(pendulum: Pendulum,
                                      recorder: StepRecorder,
                                      h: float):
    m = pendulum.params.mass
    l = pendulum.params.length
    g = pendulum.params.gravity

    ts = []
    thetas = []
    theta_dots = []
    energies = []

    for k, rec in enumerate(recorder.records):
        t = (k + 1) * h
        q_prev = rec["q_prev"]
        q_curr = rec["q_curr"]

        theta = q_curr[0]
        theta_prev = q_prev[0]

        theta_dot = (theta - theta_prev) / h

        T_k = 0.5 * m * l * l * theta_dot * theta_dot
        V_k = m * g * l * (1.0 - torch.cos(theta))
        E_k = T_k + V_k

        ts.append(t)
        thetas.append(float(theta))
        theta_dots.append(float(theta_dot))
        energies.append(float(E_k))

    return ts, thetas, theta_dots, energies


def plot_theta(ts, thetas):
    plt.figure()
    plt.plot(ts, thetas)
    plt.xlabel("t")
    plt.ylabel("theta")
    plt.title("Pendulum angle over time")
    plt.grid(True)


def plot_energy(ts, energies):
    plt.figure()
    plt.plot(ts, energies)
    plt.xlabel("t")
    plt.ylabel("H = T + V")
    plt.title("Energy over time")
    plt.grid(True)


def plot_phase(thetas, theta_dots):
    plt.figure()
    plt.plot(thetas, theta_dots)
    plt.xlabel("theta")
    plt.ylabel("theta_dot")
    plt.title("Phase portrait")
    plt.grid(True)
