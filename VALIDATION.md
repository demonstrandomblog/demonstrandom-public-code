# Validation and release status

Checks recorded on 20-21 September 2026. These are experimental teaching and
research implementations. AI warnings, source-level disclosures, and component
limitations remain applicable; passing these checks is not a correctness proof.

| Component | Measured result | Limits |
|---|---|---|
| EGraphs | 6 tests pass, including a real hash-collision regression | Simple teaching implementation and extractor |
| Selectorate | Nested-depth, unequal-level, infeasibility, budget, and gradient checks pass | Continued-fraction model; old product-form results superseded |
| Canonicalizer | All 576 strict games partition into 78 reference orbits without splits/merges; tied examples and returned transformations pass | Exact 2x2 ordinal ID, separate from the soft path |
| Lie groups and variational integration | 85 Lie tests and 8 new integrator tests pass | SO/SE scope plus rotor trajectory/charge, pendulum refinement and Newton failure checks; trainable layer remains explicitly unreviewed |
| Gradient learning | 4 tests and seeded 200-step demo pass | Article payoffs, independent three-player enumeration and finite-difference gradients |
| Differential games | 11 tests and three scenario plots/payoff heatmap pass | Stag Hunt matrix, Euler/RK4 refinement; unfinished optional collision API excluded |
| System identification | 7 analytic tests pass and full demo completes | Arnoldi unimplemented; no broad noisy-data/conditioning claim |
| Game control | Ten vignettes and all three portable verification suites pass | 89 mathematical/PID/polynomial assertions; human review pending |
| Color metric | 4 checks; original spectrum reproduced; corrected field positive at grid/stencil points | Original interpolation is indefinite at 160 points; log-Euclidean mode changes the field |
| Inspection bias | 3 checks; direct combinatorial cases and four fitted RMSE values reproduced | Fitting companion to a provisional draft |
| Cultural counting | 6 checks; exact small-space enumerations and 0.84 article example reproduced | Five overlap figures; rounded weights and volume criterion |

The current eleven-component release passes **168 pytest tests** against an exact
Git-index snapshot, with warnings treated as errors and pytest plugin autoload
disabled. The prior nine-component batch passed 145 tests. The selectorate demo,
canonicalizer example driver, and seven generated figures also complete.
The new figures were visually inspected. The historical 115-test count is
superseded: those checks missed the hierarchy and canonicalization defects.

Game control's report is in [verification.json](game_control/verification.json).
Its wheel installs and runs the full portable verification harness. The source
distribution includes the examples, checks, extraction tool, and legal notices.
The implementation and examples reproduce the recorded article source exactly.

## Commands

Run from the repository root in your Python environment. Install the dependencies
for the components you need, as described in each component's README. To check
all eleven:

```sh
python -m pip install -r games/selectorate/requirements.txt -r games/canonicalization/requirements.txt -r games/geometric_controls/requirements.txt -r systems/requirements.txt -r color_metric/requirements.txt -r inspection_bias/requirements.txt -r art_and_info/requirements.txt -r games/gradient_learning/requirements.txt -r games/differential_games/requirements.txt
python -m pip install './game_control[polynomial]'
python -m pytest -q reasoning/egraphs/union_find.py reasoning/egraphs/hashcons.py reasoning/egraphs/e_graphs.py games/selectorate games/canonicalization games/geometric_controls systems color_metric inspection_bias art_and_info games/gradient_learning games/differential_games
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
```

## Implementation changes relative to the earlier sources

- EGraphs: hash-consing now uses equality as well as hashes, the tuple-identity
  fixture constructs distinct runtime tuples, and tests do not return objects.
- Canonicalizer: exact action/player orbit enumeration replaces the flawed hard
  sorting path; tied ranks are preserved and hashes have a stable versioned
  serialization. Imports preserve Torch settings.
- Selectorate: the hierarchy uses the published continued fraction; invalid
  parameters or nonpositive attenuation raise explicitly. Over-budget finite
  requirements remain observable.
- Lie groups: SO(n) sampling produces proper rotations; repaired SO(2) signs,
  SO(3) logarithms, rigid-motion exponentials/logarithms, and coordinate versus
  matrix exponential handling are covered by the regression tests. Unsupported
  general semidirect-product exponentials/logarithms fail explicitly.
- System identification: removed the unconditional demo abort, documented the
  Arnoldi stub, and fixed heat diffusion to use its computed step size. Added
  analytic regression tests without claiming general numerical robustness.
- Game control: bundled the existing mathematical, PID, and polynomial assertions
  with portable runners and an article extractor. Superseded editorial snapshot
  checks and private-workspace loaders are excluded; the mathematical checks,
  generated implementation, and vignette examples are preserved.

The published articles were not rewritten to match implementation changes. The
EGraphs collision fix and Lie-group API repairs are implementation updates, not
claims that the original article code already had these corrections. Source
recovery and prepared-file hashes are recorded in
[SOURCE_PROVENANCE.json](SOURCE_PROVENANCE.json). Full article discrepancies are
listed in [ARTICLE_DIFFERENCES.md](ARTICLE_DIFFERENCES.md).

## Validated environment

The full 168-test suite used Python 3.13.5 in WSL, NumPy 2.3.3, SciPy 1.16.1,
PyTorch 2.7.1+cu126, torchsort 0.1.10, and SymPy 1.14.0, on CPU. GPU behavior
was not tested.

A new isolated virtual environment, without system site packages, installed
all dependencies for the three new companions and game control. Its 13 companion
tests and the complete game-control harness passed with NumPy 2.5.3, SciPy 1.18.1,
Matplotlib 3.11.2, pytest 9.1.1, SymPy 1.14.0, PyTorch 2.7.1+cpu, and z3-solver
5.1.0.0. CPU Torch was installed from the official PyTorch CPU wheel index before
installing `./game_control[polynomial]`. This supersedes the earlier check that
shared preinstalled scientific dependencies. An initial attempt on the Windows-
mounted filesystem timed out during extraction; the clean native-Linux-filesystem
installation succeeded.

These runs are two concrete environment checks, not a tested multi-version or
cross-platform support matrix. [The fresh game-control report](game_control/verification.json)
records its executed checks and environment.

## Published simulation companions, 21 September 2026

The new 23 tests also pass in the existing isolated environment described above,
without system site packages or torchsort. All three companion CLI commands
complete there. Six generated figures (pendulum, rotor, three Stag Hunt trajectories,
and the payoff heatmap) were visually inspected.

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

These selected implementations accompany four confirmed published articles,
including the rotor's Noether extension. The corresponding article sources were
unchanged during this release. Existing component checks and legal notices are
retained. No claim is made here about unpublished adjacent research modules.
