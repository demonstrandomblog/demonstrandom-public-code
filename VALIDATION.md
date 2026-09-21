# Validation

Checks recorded on 20-21 September 2026. These are experimental teaching and
research implementations. AI warnings, source-level disclosures, and component
limitations remain applicable; passing these checks is not a correctness proof.

| Component | Measured result | Limits |
|---|---|---|
| Invariant theory | 63 tests pass, including exact rank and lattice-saturation checks, bounded separation, and symbolic relations | Degree bounds and signed-permutation restriction are explicit; Koszul relations are incomplete |
| Game invariant coordinates | 17 tests and all 21 appendix scripts run; all three candidate atlases pass 216 group images and 630 sample-pair checks | Numerical ranks and finite samples do not certify global generation or separation |
| EGraphs | 6 tests pass, including a real hash-collision regression | Simple teaching implementation and extractor |
| Selectorate | Nested-depth, unequal-level, infeasibility, budget, and gradient checks pass | Continued-fraction model; old product-form results superseded |
| Canonicalizer | All 576 strict games partition into 78 reference orbits without splits/merges; tied examples and returned transformations pass | Exact 2x2 ordinal ID, separate from the soft path |
| Lie groups and variational integration | 85 Lie tests and 8 integrator tests pass | SO/SE scope plus rotor trajectory/charge, pendulum refinement and Newton failure checks; trainable layer remains explicitly unreviewed |
| Gradient learning | 4 tests and seeded 200-step demo pass | Article payoffs, independent three-player enumeration and finite-difference gradients |
| Differential games | 11 tests and three scenario plots/payoff heatmap pass | Stag Hunt matrix, Euler/RK4 refinement and fixed-step timing |
| System identification | 7 analytic tests pass and full demo completes | Arnoldi unimplemented; no broad noisy-data/conditioning claim |
| Game control | Ten vignettes and all three portable verification suites pass | 89 mathematical/PID/polynomial assertions; human review pending |
| Color metric | 4 checks; original spectrum reproduced; corrected field positive at grid/stencil points | Original interpolation is indefinite at 160 points; log-Euclidean mode changes the field |
| Inspection bias | 3 checks; direct combinatorial cases and four fitted RMSE values reproduced | Provisional functional-information model |
| Cultural counting | 6 checks; exact small-space enumerations and 0.84 article example reproduced | Five overlap figures; rounded weights and volume criterion |

The full suite passes **248 pytest tests**, with warnings treated as errors and
pytest plugin autoload disabled. Component examples complete, and thirteen
generated figures were visually inspected.

Game control's report is in [verification.json](game_control/verification.json).
Its wheel installs and runs the full portable verification harness. The source
distribution includes the examples, checks, extraction tool, and legal notices.
The implementation definitions and example bodies reproduce the recorded article
source. Generated headers describe their local purpose; the extraction check
reproduces those headers as well.

## Commands

Run from the repository root in your Python environment. Install the dependencies
for the components you need, as described in each component's README. To check
all thirteen:

```sh
python -m pip install -r games/selectorate/requirements.txt -r games/canonicalization/requirements.txt -r games/geometric_controls/requirements.txt -r systems/requirements.txt -r color_metric/requirements.txt -r inspection_bias/requirements.txt -r art_and_info/requirements.txt -r games/gradient_learning/requirements.txt -r games/differential_games/requirements.txt
python -m pip install './game_control[polynomial]' './invariants[test]' './games/invariant_coordinates[test]'
python -m pytest -q reasoning/egraphs/union_find.py reasoning/egraphs/hashcons.py reasoning/egraphs/e_graphs.py games/selectorate games/canonicalization games/geometric_controls systems color_metric inspection_bias art_and_info games/gradient_learning games/differential_games invariants games/invariant_coordinates
python -m games.canonicalization.test_games
python systems/data_driven_systems.py
python game_control/verify.py
python -m color_metric
python -m inspection_bias.fit
python -m inspection_bias.plot
python -m art_and_info
python -m games.gradient_learning
python -m games.geometric_controls.variational_example
python -m games.differential_games
python -m invariants
python -m games.invariant_coordinates
python -m games.invariant_coordinates.verify_atlases all
```

## Implementation differences

[ARTICLE_DIFFERENCES.md](ARTICLE_DIFFERENCES.md) records the implementation
corrections and their relationship to the articles. Source-file revisions and
hashes are recorded in [SOURCE_PROVENANCE.json](SOURCE_PROVENANCE.json).

## Validated environment

The full 248-test suite used Python 3.13.5 in WSL, NumPy 2.3.3, SciPy 1.16.1,
PyTorch 2.7.1+cu126, torchsort 0.1.10, and SymPy 1.14.0, on CPU. GPU behavior
was not tested.

An isolated virtual environment without system site packages also passed:

- 13 tests for color metric, inspection bias, and cultural counting.
- 23 tests for gradient learning, variational integration, and differential games.
- The complete game-control verification harness.

That environment used Python 3.13.5, NumPy 2.5.3, SciPy 1.18.1,
Matplotlib 3.11.2, pytest 9.1.1, SymPy 1.14.0, PyTorch 2.7.1+cpu, and
z3-solver 5.1.0.0. CPU Torch was installed from the official PyTorch CPU wheel
index before installing `./game_control[polynomial]`. These results cover the
listed CPU configurations. [The game-control report](game_control/verification.json)
records its executed checks and environment.

## Simulation checks

The gradient-learning, variational, and differential-game command-line examples
complete in the isolated environment. Their six plots show the pendulum, rotor,
three Stag Hunt trajectories, and the payoff matrix.

The rotor matches its analytic constant-velocity trajectory over 100 steps;
its finite-difference Noether charge error stays below 1e-5. Pendulum endpoint
errors against independent SciPy integration are below 4e-6 at h=0.02 and fall
by more than a factor of 3.5 at h=0.01. Newton failure checks cover zero iterations
and a small update with a large remaining equation residual.

The differential Stag Hunt matrix is exactly `[[[4,0],[3,3]],[[4,3],[0,3]]]`.
For the independent decay ODE, halving the step reduces Euler error by more than
1.8 and RK4 error by more than 14. Whole-step end-time semantics are tested.
The gradient-learning tests cover unequal action counts across three players
and central finite differences of each player's own payoff gradient.

## Invariant companion packages

Both new companions build as wheels and source distributions. Their wheels
install in a fresh environment without access to either source checkout.
All 80 installed-package tests, both module commands, and both README Python
examples pass. The distributions include required license notices and the
game companion includes all three numeric atlas files.

That fresh environment used Python 3.13.5, NumPy 2.5.3, SciPy 1.18.1,
SymPy 1.14.0, and pytest 9.1.1 on CPU. The full 248-test suite also passes
in the previously recorded scientific environment, with warnings as errors.

The integer-kernel tests include negative weights, dependent and empty
matrices, integers larger than machine-word size, 125 exhaustive signed rows,
and 120 random rectangular matrices. Independent exact rank and maximal-minor
gcd checks establish lattice completeness for the tested cases.

All 21 game-article appendix scripts were executed. Exact checks include
2x2 generator monomials through degree six, quartic relations, and the
three-player Molien coefficients. Numerical checks include all 216 strategy
relabelings, contrast reconstruction and orthogonality, and degree-six products.

The [candidate-atlas report](games/invariant_coordinates/verification.json)
records all 216 images of one game and every one of the 630 pairs in a seeded
36-game sample for each of three atlases. All three show zero false merges;
relative invariance errors are below 2.2e-14. This reproduces the finite
experiment and does not certify global separation.
