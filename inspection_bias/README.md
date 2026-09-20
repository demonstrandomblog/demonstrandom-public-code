# Inspection bias: functional-information fits

This package implements the fitting portion of **Inertial Growth of Functional
Information**, a provisional draft at
`demonstrandom/ml/posts/inspection_bias_functional_information/index.qmd`.
The published [Inspection Bias](https://demonstrandom.com/ml/posts/inspection_bias/index.html)
post provides the length-biased-sampling background. The recovered README's
older title, "The Law of Increasing Functional Information Is a Theorem", is
superseded by the current draft title.

> **AI warning:** Experimental research code with AI assistance. Historical
> generation details are not established for every file; automated checks do
> not establish a full human or mathematical review. See [AI_NOTICE.md](../AI_NOTICE.md).

Original code: [PolyForm Noncommercial 1.0.0](../LICENSE), with [required notices](../NOTICE)
and [citation metadata](../CITATION.cff). Third-party data keeps its own notices.

## Run and reuse

From the repository root:

```sh
python -m pip install -r inspection_bias/requirements.txt
python -m pytest -q inspection_bias
python -m inspection_bias.fit
python -m inspection_bias.plot --output functional_information.png
```

```python
from inspection_bias.persistence import compute_I, power_law_factory
from inspection_bias.fit import fit_all
results = fit_all()
weight = power_law_factory(results["power_law"]["params"])
bits = compute_I(k_max=15, m=72, n=135, log2_w=weight)
```

`compute_I` calculates the logarithm of total possibility-space weight divided
by persistence-weighted mass. `w(k)` is the draft's **effective weight**, combining
formation/discovery and persistence. The draft already explains this interpretation.
The fitter uses the nine supplied geological stages and two parameters per model.

| Model | Reproduced RMSE (bits) |
|---|---:|
| Power law | 0.72 |
| Exponential | 0.90 |
| Gaussian | 1.35 |
| Factorial | 2.02 |

The power exponent is 34.763 (34.8 at the draft's precision). Tests compare
small combinatorial cases against direct counts, verify constant weights and
stable log sums, and recompute fit residuals. A failed optimizer now raises an
error instead of returning an apparently completed fit. The command produces
the draft's observed-versus-predicted figure with those same parameters.

## Source and data

Recovered from research checkpoint `19d1371`; the numerical model and stage
values are preserved. The current draft explicitly imports these routines.
[DATA_NOTICE.md](DATA_NOTICE.md) records the Table 1 source and reuse terms.
The blog and draft were not edited or deployed.
