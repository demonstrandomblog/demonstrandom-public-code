# Game canonicalization

Code accompanying [Demonstrandom](https://demonstrandom.com/game_theory/posts/canonical_games/index.html).

> **AI warning:** This is experimental research code. AI assistance has been
> used in the project and in this cleanup; historical generation details are
> not established for every file. Existing file-level disclosures are retained.
> Passing automated checks does not establish a full human or mathematical
> review. Independently validate the behavior and results you rely on.

## Run

From the repository root, install this component's dependencies in your own
Python environment:

```sh
python -m pip install -r games/canonicalization/requirements.txt
python -m pytest -q games/canonicalization
python -m games.canonicalization.test_games
```

## Scope and known limitations

The article targets ordinal 2x2 games. The soft path and discrete canonicalization differ. Numeric sort keys and sampled examples are not a proof of invariance for arbitrary games. Importing the implementation no longer changes Torch random seeds or print settings.

See [validation report](../../VALIDATION.md) for the exact checks and their results.
Dependencies retain their own licenses. The original project code is covered
by [PolyForm Noncommercial 1.0.0](../../LICENSE); preserve [NOTICE](../../NOTICE).
For citation details see [CITATION.cff](../../CITATION.cff).
