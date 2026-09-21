# Game canonicalization

Code accompanying [Demonstrandom](https://demonstrandom.com/game_theory/posts/canonical_games/index.html).

> **AI warning:** This is experimental research code developed with AI assistance.
> Generation history and human-review coverage are incomplete; see individual
> files for additional disclosures. Passing automated checks does not establish
> a full human or mathematical review. Independently validate results you rely on.

## Run

From the repository root, install this component's dependencies in your own
Python environment:

```sh
python -m pip install -r games/canonicalization/requirements.txt
python -m pytest -q games/canonicalization
python -m games.canonicalization.test_games
```

## Scope and known limitations

`class_id` now exactly enumerates the eight action/player relabelings for
finite real payoff tensors of shape `(2, 2, 2)`. Player exchange carries its
strategy axis with it. Tied payoffs retain equal dense ranks. The exhaustive
strict-game check partitions all 576 rankings into 78 classes with no split
or merged reference orbits. Named-player equivalence instead has 144 classes;
the article's count conflicts with its player-exchange definition.

```python
import torch
from games.canonicalization.robinson_goforth import GameCanonicalizer
payoffs = torch.tensor([[[3., 0.], [5., 1.]], [[3., 5.], [0., 1.]]])
short_id, full_id, ranks = GameCanonicalizer(2).class_id(payoffs)
```

**Compatibility:** IDs change from the earlier algorithm. Version-2 hashes use
an explicit version prefix and eight rank bytes, independent of input dtype
and machine endianness. Use the full hash for identification; the short form
is a display abbreviation. `hard_canonical` returns one-based float ranks and
permutation metadata; its docstring explains how to apply the accompanying
strategy-axis transpose. `class_id` returns zero-based integer ranks.

`forward` remains the separate article-derived soft representation; its output
is not an exact canonical identifier. The discrete rank/hash path has no
gradient. Ordinal equivalence preserves order, not numerical payoff spacing or
all mixed-strategy equilibria. Inputs outside the exact 2x2 scope and nonfinite
payoffs raise `ValueError`. Importing preserves Torch RNG and display settings.

See [validation report](../../VALIDATION.md) for the exact checks and their results.
Dependencies retain their own licenses. The original project code is covered
by [PolyForm Noncommercial 1.0.0](../../LICENSE); preserve [NOTICE](../../NOTICE).
For citation details see [CITATION.cff](../../CITATION.cff).
