# Lie groups

Code accompanying [Demonstrandom](https://demonstrandom.com/game_theory/posts/discrete_controls_lagrange/index.html).

> **AI warning:** This is experimental research code. AI assistance has been
> used in the project and in this cleanup; historical generation details are
> not established for every file. Existing file-level disclosures are retained.
> Passing automated checks does not establish a full human or mathematical
> review. Independently validate the behavior and results you rely on.

## Run

From the repository root, install this component's dependencies in your own
Python environment:

```sh
python -m pip install -r games/geometric_controls/requirements.txt
python -m pytest -q games/geometric_controls
```

## Scope and known limitations

Repaired SO(n) sampling and rigid-motion exponentials are included. Principal logarithms are supported for SO(2), SO(3), SE(2), and SE(3); unsupported general semidirect products raise NotImplementedError. Matrix inputs use exp_matrix(). CPU checks do not establish GPU coverage. The trainable layer retains its entirely-GPT-generated, unreviewed notice.

See [validation report](../../VALIDATION.md) for the exact checks and their results.
Dependencies retain their own licenses. The original project code is covered
by [PolyForm Noncommercial 1.0.0](../../LICENSE); preserve [NOTICE](../../NOTICE).
For citation details see [CITATION.cff](../../CITATION.cff).
