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

The release was checked from a clean copy of the staged files, containing only
these four components: **108 tests passed**. The canonicalizer example driver
also completed, and importing its implementation and tests preserved the caller's
Torch random state and display preferences.

## Commands

Run from the repository root in your Python environment. Install the dependencies
for the components you need, as described in each component's README. To check
all four:

```sh
python -m pip install -r games/selectorate/requirements.txt -r games/canonicalization/requirements.txt -r games/geometric_controls/requirements.txt
python -m pytest -q reasoning/egraphs/union_find.py reasoning/egraphs/hashcons.py reasoning/egraphs/e_graphs.py games/selectorate games/canonicalization games/geometric_controls
python -m games.canonicalization.test_games
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

The published articles were not rewritten to match implementation changes. The
EGraphs collision fix and Lie-group API repairs are implementation updates, not
claims that the original article code already had these corrections. Source
recovery and prepared-file hashes are recorded in
[SOURCE_PROVENANCE.json](SOURCE_PROVENANCE.json).

## Validated environment

Checks used Python 3.13.5 in WSL, NumPy 2.3.3, SciPy 1.16.1, and PyTorch
2.7.1+cu126 with torchsort installed. CPU tests were run; the CUDA suffix does
not imply GPU coverage. The checks used an existing scientific Python
environment, not a fresh dependency installation. Dependency files do not yet
define a tested multi-version support matrix. Fresh installation, other
platforms, and GPU behavior need separate validation.
