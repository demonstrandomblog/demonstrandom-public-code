# Inspection bias: functional-information fits

Fits persistence-weighted functional-information models to nine geological stages.
[Inspection Bias](https://demonstrandom.com/ml/posts/inspection_bias/index.html)
provides the length-biased-sampling background. The fitting model is provisional.

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
by persistence-weighted mass. `w(k)` is an **effective weight**, combining
formation/discovery and persistence.
The fitter uses the nine supplied geological stages and two parameters per model.

| Model | Reproduced RMSE (bits) |
|---|---:|
| Power law | 0.72 |
| Exponential | 0.90 |
| Gaussian | 1.35 |
| Factorial | 2.02 |

The fitted power exponent is 34.763. Tests compare
small combinatorial cases against direct counts, verify constant weights and
stable log sums, and recompute fit residuals. A failed optimizer now raises an
error instead of returning an apparently completed fit. The command produces
an observed-versus-predicted figure with those same parameters.

## Data

[DATA_NOTICE.md](DATA_NOTICE.md) records the Table 1 source and reuse terms.
