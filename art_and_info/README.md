# Cultural saturation: counting and overlap figures

Companion to [Are We Approaching Cultural Saturation?](https://demonstrandom.com/essays/posts/cultural_saturation/index.html).
Hamming and weighted-Hamming counting calculations with five overlap plots.

> **AI warning:** Experimental research code with AI assistance. Historical
> generation details are not established for every file; automated checks do
> not establish a full human or mathematical review. See [AI_NOTICE.md](../AI_NOTICE.md).

Original code: [PolyForm Noncommercial 1.0.0](../LICENSE), with [required notices](../NOTICE)
and [citation metadata](../CITATION.cff). Third-party data keeps its own notices.

## Model and notation

A configuration is a string of `k` binary features. Hamming distance counts
the features on which two strings differ. A radius-`r` ball contains
`V(k,r) = sum(comb(k,i), i=0..r)` strings. Given a positive integer `N`, the
code finds the first radius satisfying `N * V(k,r) >= 2**k` and reports the
overlap proxy `1 - r/k`. This is a counting model for hypothetical feature
spaces; it does not measure similarities between actual works of art.

Weighted distance assigns feature `i` the weight `i**(-beta)`, for `i=1..k`.
Dynamic programming counts subsets after rounding each weight to a positive
integer using `scale`. The weighted overlap proxy is
`1 - (r_int/scale) / sum(i**(-beta), i=1..k)`.
Larger scales increase both resolution and memory use. NumPy and Matplotlib
support the plots; the unweighted counts use exact Python integers.

## Run and reuse

From the repository root:

```sh
python -m pip install -r art_and_info/requirements.txt
python -m pytest -q art_and_info
python -m art_and_info --output cultural_saturation_figures
```

```python
from art_and_info.hamming_radius_plot import min_radius_for_covering, fractional_overlap
min_radius_for_covering(50, 5_000_000)  # 8
fractional_overlap(50, 5_000_000)       # 0.84
```

The generated files correspond to `Overlap_Fraction.png`, `Overlap_vs_N.png`,
and weighted `Overlap_vs_N_0_5.png`, `Overlap_vs_N_1.png`, `Overlap_vs_N_2.png`.
The first plot uses k=5..60 in steps of 5 and log10(N)=0..9. The unweighted
cross-sections use k=5..100 and log10(N)=0..12. Weighted plots use k=5..40,
beta=0.5/1/2 and integer-weight scale 500. Weighted plots also use log10(N)=0..12.

The `min_radius_for_covering` helpers use the **volume criterion** N*V >= 2**k;
they do not construct an actual optimal covering code. Weighted subset counts
are exact for the rounded integer weights; the real-weight diameter remains
the sum of i**(-beta) for i=1..k.

Verification enumerates every binary string for small dimensions, checks weighted
subset counts independently, and reproduces the article's 0.84 example. Integer
comparisons replace floating-point threshold division; negative weighted radii
now return zero. Plot functions return Matplotlib figures for saving or display.
