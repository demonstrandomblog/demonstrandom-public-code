# Differential games

Companion to [Differential Games and Stag Hunt](https://demonstrandom.com/game_theory/posts/differential_stag_hunt/).

> **AI warning:** This is experimental research code developed with AI assistance.
> Generation history and human-review coverage are incomplete; see individual
> files for additional disclosures. Passing automated checks does not establish
> a full human or mathematical review. Independently validate results you rely on.

## Simulation model

A differential game evolves a joint state by `dx/dt = f(x, u)`. Each agent has
named state variables and a policy mapping observations to a control vector.
`Game` bundles these definitions with an initial-state sampler and payoff
rule. `Arena` evaluates the policies, takes Euler or fourth-order Runge-Kutta
steps, and records state and accumulated payoff. Policies, dynamics, and
payoffs must agree on the slices supplied by `StateSpace`; positions in the
bundled example use `(x, y)` coordinates.

The supplied Stag Hunt has two hunters, `c1` and `c2`, and three moving prey
agents: a stag and two hares. Each hunter chooses `ChaseStag` (cooperate) or
`ChaseHare` (defect). Joint stag capture earns both hunters 4; hare capture
earns 3. With the bundled initial conditions, a lone stag pursuer earns 0. The normal-form converter
runs each policy combination from sampled initial states and returns a tensor
of shape `(players, strategies_1, strategies_2, ...)`. These are simulation
payoffs for the supplied policies, not a search for optimal continuous controls.

## Run

From the repository root:

```sh
python -m pip install -r games/differential_games/requirements.txt
python -m games.differential_games --output-dir outputs/differential_games
python -m pytest -q games/differential_games
```

The command saves three trajectory plots and the payoff heatmap. It reproduces
cooperation `(4, 4)`, defection `(3, 3)`, asymmetric play `(0, 3)`, and the full
2-by-2 payoff matrix. `games.differential_games.core` contains the reusable state,
observation, policy, dynamics, payoff, Euler/RK4 integration, arena, and
normal-form conversion classes.

```python
from games.differential_games.core import Arena, NormalFormConverter, build_stag_hunt

game, state_space = build_stag_hunt()
arena = Arena(game, dt=0.02, max_time=15.0)
print(NormalFormConverter.to_payoff_matrix(arena, ["c1", "c2"]))
```

## Article comparison and checks

`BoundaryConstraint` clamps positions to a box and reflects boundary velocities.
Multiple supplied constraints are applied once in sequence, without a joint
feasibility solver. The Stag Hunt example runs without constraints.

Positive finite time steps and finite simulation bounds are required. Simulation
retains whole fixed steps, ending at the first grid time at or beyond `until`.
For example, `dt=0.1, until=0.25` finishes at `0.3`. Payoffs use the same steps.
RK4 holds controls fixed within each integration step; it does not reevaluate
policies at intermediate RK stages. `GameState` is a mutable snapshot; `clone()`
copies its tensor and top-level dictionaries.

Eleven tests reproduce all four payoff profiles, check Euler/RK4 error refinement
against `exp(-t)`, reject invalid time bounds, and pin the fixed-step convention.

Original code and documentation: [PolyForm Noncommercial 1.0.0](../../LICENSE).
Preserve [NOTICE](../../NOTICE). Citation: [CITATION.cff](../../CITATION.cff).
Dependencies retain their own licenses.
