# Selectorate model

Code accompanying [Demonstrandom](https://demonstrandom.com/governance/posts/game_theory_dictatorships_selectorate/index.html).

> **AI warning:** This is experimental research code developed with AI assistance.
> Generation history and human-review coverage are incomplete; see individual
> files for additional disclosures. Passing automated checks does not establish
> a full human or mathematical review. Independently validate results you rely on.

## Model and notation

`W` is the size of the coalition needed to retain power, `S` is the size of
the selectorate from which coalition members are chosen, and `B` is the
available budget. A targeted transfer `p` pays each coalition member `p/W`;
the defection benchmark is `B/S`. The minimum loyal transfer is therefore
`p_min = B*W/S`, leaving rents `B-p_min`. `SelectorateModel()` returns these
quantities as a `SelectorateEquilibrium` of scalar tensors.

For a hierarchy, let `r_i = W_i/S_i`, with levels supplied from top to bottom.
Start with `x = r_bottom`, then repeatedly set `x = r_i/(1-x)` moving upward.
`p_min_composed` returns `B_top*x`; subordinate budgets do not enter this
continued-fraction model. `total_attenuation` returns `1-x` for the supplied
sub-hierarchy. A nonpositive intermediate denominator is infeasible.

The channel helpers compare targeted benefits `1/W` and `1/S` with a universal
benefit coefficient `beta/N`, where `N` is population size and `beta` is the
universal channel's effectiveness factor. `democratic_region` tests
`beta/N >= 1/S`. These are mathematical model outputs, not empirical predictions
of political behavior. The trainable scalar parameters support PyTorch autograd.

## Run

From the repository root, install this component's dependencies in your own
Python environment:

```sh
python -m pip install -r games/selectorate/requirements.txt
python -m pytest -q games/selectorate
```

## Scope and known limitations

`p_min_composed(top, *subs)` evaluates the continued fraction defined above,
with levels ordered top to bottom. Three ratios of 0.1 and budget 100 give
11.25; at 30 levels the requirement is approximately 11.27017. Earlier releases
used an incorrect product for deeper hierarchies. `total_attenuation(*subs)`
returns one minus the effective share of the entire sub-hierarchy.

```python
from games.selectorate import SelectorateModel, p_min_composed
levels = [SelectorateModel(W=10, S=100, B=100) for _ in range(3)]
p_min_composed(*levels)  # tensor(11.2500), with autograd
```

Parameters require finite `0 < W <= S` and `B >= 0`. Parameters are rechecked
on evaluation after optimization. A nonpositive hierarchy denominator raises
`ValueError`; it is not clamped to a tiny positive number. A finite final
requirement above B is returned unchanged, indicating an over-budget hierarchy.
Flat full inclusion W=S is allowed; using it as an attenuating sub-level is not.
Scalar output follows the model's dtype/device. Gradients are retained inside
the valid domain. Checks include unequal levels, depth, infeasibility, and
finite-difference sensitivities.

See [validation report](../../VALIDATION.md) for the exact checks and their results.
Dependencies retain their own licenses. The original project code is covered
by [PolyForm Noncommercial 1.0.0](../../LICENSE); preserve [NOTICE](../../NOTICE).
For citation details see [CITATION.cff](../../CITATION.cff).
