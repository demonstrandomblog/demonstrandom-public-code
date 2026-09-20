# Cultural saturation: counting and overlap figures

Companion to [Are We Approaching Cultural Saturation?](https://demonstrandom.com/essays/posts/cultural_saturation/index.html).
This release contains the Hamming and weighted-Hamming counting calculations
and five overlap plots. The separate energy/temperature sketches are not part
of this component.

> **AI warning:** Experimental research code with AI assistance. Historical
> generation details are not established for every file; automated checks do
> not establish a full human or mathematical review. See [AI_NOTICE.md](../AI_NOTICE.md).

Original code: [PolyForm Noncommercial 1.0.0](../LICENSE), with [required notices](../NOTICE)
and [citation metadata](../CITATION.cff). Third-party data keeps its own notices.

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
beta=0.5/1/2 and integer-weight scale 500. Weighted plots also use log10(N)=0..12. These recover the calculation and plotting conventions;
pixel-identical historical image exports are not established.

The article already labels the packing argument as a rough model. The helper
names `min_radius_for_covering` refer to its **volume criterion** N*V >= 2**k;
they do not construct an actual optimal covering code. Weighted subset counts
are exact for the rounded integer weights; the real-weight diameter remains
the article's sum of i**(-beta).

Verification enumerates every binary string for small dimensions, checks weighted
subset counts independently, and reproduces the article's 0.84 example. Integer
comparisons replace floating-point threshold division; negative weighted radii
now return zero. Plot functions return Matplotlib figures for saving or display.

Recovered from the `19d1371` research checkpoint. No artwork, external datasets,
or third-party implementation is bundled. The blog was not changed.
