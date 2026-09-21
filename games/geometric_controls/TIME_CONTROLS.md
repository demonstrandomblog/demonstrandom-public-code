# Time-dependent controls and Kepler scaling

Companion to [Noether's Theorem with Time](https://demonstrandom.com/game_theory/posts/noether_time/)
and [Dynamical Similarity and Equivariant Symmetry](https://demonstrandom.com/game_theory/posts/dynamical_similarity/).

> **AI warning:** Experimental code developed with AI assistance. Automated
> checks do not establish full human or mathematical review.

## Run

From the repository root:

```sh
python -m pip install -r games/geometric_controls/requirements.txt
python -m games.geometric_controls.kepler_example
python -m pytest -q games/geometric_controls/test_time_controls.py
```

The command runs the two 500-position article examples with h=0.01 and scale
lambda=2, printing JSON diagnostics. It requires no external data or images.
Use --steps, --step-size and --scale to change those parameters.

## State, time, and equations

The time-aware module reuses the shared spatial layouts and Lie primitives.
A model subclasses VariationalSystem, provides control_plane(), params() and
lagrangian(ctrl, dctrl), and reads physical time through ctrl.t. Coordinates
contain spatial entries only. dctrl.t is one; spatial velocities are measured
per unit physical time.

VariationalIntegrator solves the discrete Euler-Lagrange equation
D2 Ld(q_prev,q_curr,h,t_prev) + D1 Ld(q_curr,q_next,h,t_curr) = 0.
The discrete action is h times the Lagrangian at the group midpoint, relative
velocity, and midpoint time. Physical time advances by a prescribed positive h;
this is not an adaptive extended-state time integrator.

Initialize q_prev at start_time and q_curr at start_time+h. The first step
returns the position at start_time+2h. A failed residual check returns False
and leaves the clock unchanged. Callbacks receive states, endpoint times,
residual norm, success and registered charges. Use float64 for the default
1e-10 nonlinear tolerance. Linear-solve failures raise.

```python
import torch
from games.geometric_controls.discrete_control_lagrange_t import Kepler, VariationalIntegrator

model = Kepler({"mass": 1.0, "mu": 1.0})
solver = VariationalIntegrator(model, step_size=0.01)
q0 = torch.tensor([1.0, 0.0], dtype=torch.float64)
q1 = torch.tensor([1.0, 0.008], dtype=torch.float64)
q2, success = solver.step(q0, q1)
assert success
```

## Diagnostics and scaling

For a Symmetry, space_transform(epsilon,t,q) and the optional
time_transform(epsilon,t,q) define infinitesimal generators omega and tau.
Centered differences use epsilon=1e-5 independently of the Newton tolerance.
Registered charges are J=p.omega-E*tau, with p=D2 Ld and
E=-dLd/dh holding both spatial endpoints and the initial time fixed.
The time-translation charge is **minus** energy.

Fixed-step integration does not generally conserve energy exactly. Rotational
invariance of the discrete action preserves angular momentum within numerical
error. Kepler dynamical similarity rescales r by lambda, time and h by
lambda^(3/2), and velocity by lambda^(-1/2). Its diagnostic
J=p.r-(3/2)Et scales by sqrt(lambda); J is not constant along an orbit.

Kepler uses L=m|v|^2/2+mu/|r|, so mu is an energy coefficient and the acceleration
is -(mu/m)r/|r|^3. By default the potential is unsoftened and rejects r=0.
An explicit softening length uses sqrt(|r|^2+softening^2). To retain the scaling
relation for softened models, scale that length by lambda too.

## Article comparison and checks

The article model packs only r, so its supplied t value was unused. This
implementation passes time explicitly and corrects the first recorded timestamp.
Energy uses the step derivative instead of momentum times a finite-difference
velocity. The article's 1e-10 radius-squared regularizer is optional here:
softening=1e-5 reproduces it. The example retains first-order initialization.

Eleven tests cover an analytic explicitly driven particle at nonzero initial
time, endpoint energy, independent SciPy Kepler refinement, trajectory/charge
scaling, rotational momentum, failure reporting, invalid steps and collision
handling. The 500-step run measures energy drift; it does not assert exact
energy conservation. See [validation](../../VALIDATION.md) for measured results.

Original code uses [PolyForm Noncommercial 1.0.0](../../LICENSE).
Preserve [NOTICE](../../NOTICE); dependencies retain their own licenses.
