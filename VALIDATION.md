# Validation and release status

Checks recorded on 20 September 2026. These are experimental teaching and
research implementations. AI warnings, source-level disclosures, and component
limitations remain applicable; passing these checks is not a correctness proof.

| Component | Measured result | Limits |
|---|---|---|
| EGraphs | 6 tests pass, including a real hash-collision regression | Simple teaching implementation and extractor |
| Selectorate | 14 tests pass | Model behavior, not empirical validation |
| Canonicalizer | 3 pytest checks pass; example driver and import-state check pass | Five example families with variants; not arbitrary-game invariance |
| Lie groups | 85 tests pass | Documented SO/SE scope, CPU checks; trainable layer remains explicitly unreviewed |
| System identification | 7 analytic tests pass and full demo completes | Arnoldi unimplemented; no broad noisy-data/conditioning claim |
| Game control | Ten vignettes and all three portable verification suites pass | 89 mathematical/PID/polynomial assertions; human review pending |

The six-component release was checked from a clean copy of the staged files:
**115 pytest tests passed**, plus the game-control verification command and the
system-identification demo. Earlier release checks also ran the canonicalizer
example driver and confirmed that imports preserve Torch random state and
display preferences.

Game control's report is in [verification.json](game_control/verification.json).
Its wheel installs and runs the full portable verification harness. The source
distribution includes the examples, checks, extraction tool, and legal notices.
The implementation and examples reproduce the recorded article source exactly.

## Commands

Run from the repository root in your Python environment. Install the dependencies
for the components you need, as described in each component's README. To check
all six:

```sh
python -m pip install -r games/selectorate/requirements.txt -r games/canonicalization/requirements.txt -r games/geometric_controls/requirements.txt -r systems/requirements.txt
python -m pip install './game_control[polynomial]'
python -m pytest -q reasoning/egraphs/union_find.py reasoning/egraphs/hashcons.py reasoning/egraphs/e_graphs.py games/selectorate games/canonicalization games/geometric_controls systems
python -m games.canonicalization.test_games
python systems/data_driven_systems.py
python game_control/verify.py
```

## Implementation changes relative to the earlier sources

- EGraphs: hash-consing now uses equality as well as hashes, the tuple-identity
  fixture constructs distinct runtime tuples, and tests do not return objects.
- Canonicalizer: imports no longer seed Torch or change its display preferences;
  the existing tests have a scoped pytest fixture.
- Selectorate: its test uses a package-relative import.
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
[SOURCE_PROVENANCE.json](SOURCE_PROVENANCE.json).

## Validated environment

Checks used Python 3.13.5 in WSL, NumPy 2.3.3, SciPy 1.16.1, and PyTorch
2.7.1+cu126, torchsort 0.1.10, SymPy 1.14.0, and z3 5.1.0. CPU tests were run;
the CUDA suffix does not imply GPU coverage. The checks used an existing scientific Python
environment. Game control was installed as a wheel in a separate virtual
environment sharing the preinstalled scientific dependencies, with z3-solver
installed separately through pip. This did not exercise a fresh installation
of the entire dependency set. Dependency files do not yet define a tested
multi-version support matrix. Other platforms and GPU behavior need separate
validation.
