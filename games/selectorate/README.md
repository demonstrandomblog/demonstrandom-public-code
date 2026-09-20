# Selectorate model

Code accompanying [Demonstrandom](https://demonstrandom.com/governance/posts/game_theory_dictatorships_selectorate/index.html).

> **AI warning:** This is experimental research code. AI assistance has been
> used in the project and in this cleanup; historical generation details are
> not established for every file. Existing file-level disclosures are retained.
> Passing automated checks does not establish a full human or mathematical
> review. Independently validate the behavior and results you rely on.

## Run

From the repository root, install this component's dependencies in your own
Python environment:

```sh
python -m pip install -r games/selectorate/requirements.txt
python -m pytest -q games/selectorate
```

## Scope and known limitations

Scaling, hierarchy, channel, and gradient checks cover the implemented model. They do not establish empirical political predictions.

See [validation report](../../VALIDATION.md) for the exact checks and their results.
Dependencies retain their own licenses. The original project code is covered
by [PolyForm Noncommercial 1.0.0](../../LICENSE); preserve [NOTICE](../../NOTICE).
For citation details see [CITATION.cff](../../CITATION.cff).
