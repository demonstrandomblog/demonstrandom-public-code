# Inspection bias: functional-information fits

Fits persistence-weighted functional-information models to nine geological stages.
[Inspection Bias](https://demonstrandom.com/ml/posts/inspection_bias/index.html)
provides the length-biased-sampling background. The fitting model is provisional.

> **AI warning:** Experimental research code with AI assistance. Historical
> generation details are not established for every file; automated checks do
> not establish a full human or mathematical review. See [AI_NOTICE.md](../AI_NOTICE.md).

Original code: [PolyForm Noncommercial 1.0.0](../LICENSE), with [required notices](../NOTICE)
and [citation metadata](../CITATION.cff). Third-party data keeps its own notices.

## Model and inputs

A formula of complexity `k` uses `k` distinct elements selected from `m`
available elements and coefficients selected from `1..n`. Its possibility-space
weight is `Phi(k) = comb(m,k) * n!/(n-k)! * correction(k)`. The tabulated
corrections for `k=1..6` are `1, 0.64, 0.836, 0.924, 0.965, 0.983`; the
implementation uses 1 for larger `k`. Use integers satisfying
`1 <= k_max <= min(m,n)`.

For `k=1..k_max`, define `N = sum(Phi(k))` and
`M = sum(Phi(k)*w(k))`. The output is `I = log2(N/M)` bits, evaluated with
stable logarithmic sums. The callable `log2_w(k)` supplies the logarithm of
the effective weight. This weight combines formation, discovery, and persistence;
fitting it does not identify those mechanisms separately.

The four two-parameter models are `A*k**(-gamma)`, `A*exp(-beta*k)`,
`A*exp(-beta*k*k)`, and `A/(k!)**alpha`. Each fit minimizes squared errors
in information across the nine bundled observations. Rows of `HAZEN_WONG_DATA`
store `(k_max, m, n, observed_count, observed_information_bits)`; the fit uses
the information column. These are provisional models fitted to the same data
used to report error, without held-out predictive validation.

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
