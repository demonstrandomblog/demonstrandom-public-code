# Lie groups and variational integration

Code accompanying [Demonstrandom](https://demonstrandom.com/game_theory/posts/discrete_controls_lagrange/index.html).

> **AI warning:** This is experimental research code developed with AI assistance.
> Generation history and human-review coverage are incomplete; see individual
> files for additional disclosures. Passing automated checks does not establish
> a full human or mathematical review. Independently validate results you rely on.

## Coordinates and components

`lie_groups.py` implements Euclidean translations `Rn(n)`, proper rotations
`SOn(n)`, direct products, and rotation/translation semidirect products. Group
elements wrap tensors; multiplication composes elements, `inverse()` reverses
them, and `exp`/`log` convert between algebra coordinates and elements.
For rotations, coordinates fill the upper triangle of a skew-symmetric
matrix in row order. In 2D the convention is `hat(a) = [[0,a],[-a,0]]`.
For rigid motions `(R,t)`, composition is
`(R1,t1)*(R2,t2) = (R1@R2, t1 + R1@t2)`.

`learnable_lie_groups.py` stores trainable algebra coordinates in PyTorch
parameters and exponentiates them to group elements. Rotation and translation
actions are available; the rigid-motion class does not implement a point action.
`discrete_control_lagrange.py` supplies mechanical models and a variational
integrator, with pendulum and free-rotor demonstrations in `variational_example.py`.

The integrator approximates the action over one step by a midpoint discrete
Lagrangian `Ld(q_k,q_next,h)`. Given consecutive configurations, Newton's method
solves `D2 Ld(q_prev,q_k,h) + D1 Ld(q_k,q_next,h) = 0` for the next one.
Coordinates are flat tensors in the order defined by the model's control plane.
The pendulum uses `L = m*l**2*theta_dot**2/2 - m*g*l*(1-cos(theta))`;
the rotor omits the potential term. Principal rotation logarithms restrict
the coordinate chart and are discontinuous at a half-turn.

## Run

From the repository root, install this component's dependencies in your own
Python environment:

```sh
python -m pip install -r games/geometric_controls/requirements.txt
python -m pytest -q games/geometric_controls
python -m games.geometric_controls.variational_example --output-dir outputs/variational
```

## Scope and known limitations

SO(n) sampling and rigid-motion exponentials are included. Principal logarithms are supported for SO(2), SO(3), SE(2), and SE(3); unsupported general semidirect products raise NotImplementedError. Matrix inputs use exp_matrix(). CPU checks do not establish GPU coverage. The trainable layer retains its entirely-GPT-generated, unreviewed notice.

See [validation report](../../VALIDATION.md) for the exact checks and their results.
Dependencies retain their own licenses. The original project code is covered
by [PolyForm Noncommercial 1.0.0](../../LICENSE); preserve [NOTICE](../../NOTICE).
For citation details see [CITATION.cff](../../CITATION.cff).

## Basic variational integrator

`discrete_control_lagrange.py` accompanies the basic integrator in
[Controls from the Geometric Perspective](https://demonstrandom.com/game_theory/posts/discrete_controls_lagrange/index.html)
and the free-rotor charge in
[Noether's Theorem and Geometric Controls](https://demonstrandom.com/game_theory/posts/noether_geometric_controls/).
It uses the bundled Lie primitives, midpoint discrete Lagrangians,
autograd derivatives, and Newton steps for the discrete Euler-Lagrange equation.

```python
import torch
from games.geometric_controls.discrete_control_lagrange import FreeRotor, VariationalIntegrator

rotor = FreeRotor({"mass": 1.0, "length": 1.0})
solver = VariationalIntegrator(rotor, step_size=0.01)
q_next, success = solver.step(torch.tensor([0.20], dtype=torch.float64),
                             torch.tensor([0.21], dtype=torch.float64))
assert success  # q_next is 0.22
```

Use float64 for the default `1e-10` Newton tolerance. `step()` returns a detached
position and a boolean determined by the final equation residual. A small Newton
step alone does not establish convergence. Linear solve failures still raise.
Eight tests cover analytic rotor motion/momentum, second-order pendulum refinement,
zero-iteration failure, small-step false success, and invalid step sizes.

The CLI saves pendulum and rotor angle plots, using 500 positions at `h=0.01`
by default. `--steps 10000 --step-size 0.001` restores the article-scale run.
It retains the article's first-order initialization `q1=q0+h*v0`; the convergence
test uses an independently accurate q1 to isolate integration error. Reported
energy uses the article's backward-difference velocity estimate.
The Noether charge retains the article's finite-difference generator with epsilon
equal to the solver tolerance; the tested rotor charge error is below `1e-5`.
