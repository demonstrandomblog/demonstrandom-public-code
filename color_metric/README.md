# Color metric and Killing fields

Companion to [Do We See the Same Colors?](https://demonstrandom.com/theory_of_mind/posts/color_qualia_riemannian/index.html).

> **AI warning:** Experimental research code with AI assistance. Historical
> generation details are not established for every file; automated checks do
> not establish a full human or mathematical review. See [AI_NOTICE.md](../AI_NOTICE.md).

Original code: [PolyForm Noncommercial 1.0.0](../LICENSE), with [required notices](../NOTICE)
and [citation metadata](../CITATION.cff). Third-party data keeps its own notices.

## Run

From the repository root:

```sh
python -m pip install -r color_metric/requirements.txt
python -m pytest -q color_metric
python -m color_metric --output color_metric_results
```

The command saves `results.json` and `singular_values.png`, comparing two
interpolations with the same 25 measurements, 332 grid points, degree-3 ansatz,
and finite-difference step 0.005. Importing the module does not run the search.

```python
from color_metric.metric import MetricField, gamut_grid, killing_matrix
metric = MetricField()  # positive-definite log-Euclidean interpolation
A = killing_matrix(metric, gamut_grid())  # shape (996, 20)
```

## Article comparison and correction

`article-component` reproduces the printed singular-value table: largest
1,849,601.0144 and smallest 1,753.2664. However, its component-wise spline
produces nonpositive metric eigenvalues at **160 of 332** grid points.
That table therefore does not validate a Riemannian metric over the full grid.
This mode is retained explicitly for reproduction and diagnosis.

The default `log-euclidean` mode interpolates symmetric matrix logarithms and
exponentiates them, preserving positive definiteness. It changes the fitted
field and its spectrum; it is an implementation correction, not the original
article calculation. Both modes have rank 20 at relative SVD tolerance 1e-10.
The saved output records the full spectrum and threshold so they can be inspected.

The independent reference check recovers the two translations and rotation
of the Euclidean plane.
Tests also check ellipse axes and positivity at every grid/finite-difference
point for the corrected field.

## Data

[The table notice](data/NOTICE) identifies MacAdam, Wyszecki & Stiles, and the
LuxPy transcription. Its [upstream GPL-3.0 notice](data/LICENSE) is retained
separately; no LuxPy executable code is included. The sampled spectral-locus
coordinates are the article's constants.
