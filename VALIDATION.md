# Validation and release status

Checks recorded on 20 September 2026. These are experimental teaching and
research implementations. AI warnings, source-level disclosures, and component
limitations remain applicable; passing these checks is not a correctness proof.

| Component | Measured result | Limits |
|---|---|---|
| EGraphs | 6 tests pass, including a real hash-collision regression | Simple teaching implementation and extractor |
| Selectorate | Nested-depth, unequal-level, infeasibility, budget, and gradient checks pass | Continued-fraction model; old product-form results superseded |
| Canonicalizer | All 576 strict games partition into 78 reference orbits without splits/merges; tied examples and returned transformations pass | Exact 2x2 ordinal ID, separate from the soft path |
| Lie groups | 85 tests pass | Documented SO/SE scope, CPU checks; trainable layer remains explicitly unreviewed |
| System identification | 7 analytic tests pass and full demo completes | Arnoldi unimplemented; no broad noisy-data/conditioning claim |
| Game control | Ten vignettes and all three portable verification suites pass | 89 mathematical/PID/polynomial assertions; human review pending |
| Color metric | 4 checks; original spectrum reproduced; corrected field positive at grid/stencil points | Original interpolation is indefinite at 160 points; log-Euclidean mode changes the field |
| Inspection bias | 3 checks; direct combinatorial cases and four fitted RMSE values reproduced | Fitting companion to a provisional draft |
| Cultural counting | 6 checks; exact small-space enumerations and 0.84 article example reproduced | Five overlap figures; rounded weights and volume criterion |

The expanded nine-component release passes **145 pytest tests**, with warnings
treated as errors and pytest plugin autoload disabled. The selectorate demo,
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
all nine:

```sh
python -m pip install -r games/selectorate/requirements.txt -r games/canonicalization/requirements.txt -r games/geometric_controls/requirements.txt -r systems/requirements.txt -r color_metric/requirements.txt -r inspection_bias/requirements.txt -r art_and_info/requirements.txt
python -m pip install './game_control[polynomial]'
python -m pytest -q reasoning/egraphs/union_find.py reasoning/egraphs/hashcons.py reasoning/egraphs/e_graphs.py games/selectorate games/canonicalization games/geometric_controls systems color_metric inspection_bias art_and_info
python -m games.canonicalization.test_games
python systems/data_driven_systems.py
python game_control/verify.py
python -m color_metric
python -m inspection_bias.fit
python -m inspection_bias.plot
python -m art_and_info
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

The full 145-test suite used Python 3.13.5 in WSL, NumPy 2.3.3, SciPy 1.16.1,
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
