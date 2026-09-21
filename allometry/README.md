# Mammalian mass and oxygen consumption

This helper plots positive finite measurements from a local PanTHERIA TSV.
By default it pairs mass reported with the metabolic measurements
(`5-2_BasalMetRateMass_g`) with basal oxygen consumption
(`18-1_BasalMetRate_mLO2hr`). The dataset metadata defines that mass as belonging
to the individuals used for the metabolic measurements. General adult body mass
(`5-1_AdultBodyMass_g`) is a different species-level quantity.

Only mass is converted, from grams to kilograms. Oxygen consumption remains in
milliliters of oxygen per hour. It is not labeled as watts. Missing matched mass
is excluded rather than silently replaced by a general adult-mass estimate.

Use Python 3.11 or newer. From the repository root:

```sh
python -m pip install "./allometry[test]"
python -m allometry.allometry PanTHERIA_1-0_WR05_Aug2008.txt --output pantheria_mass_bmr.png
python -m allometry.allometry PanTHERIA_1-0_WR05_Aug2008.txt --mass-source adult --output pantheria_adult_mass.png
python -m pytest -q allometry
```

Download the data separately from the
[PanTHERIA archive](https://esapubs.org/archive/ecol/E090/184/).
The [dataset metadata](https://esapubs.org/archive/ecol/E090/184/metadata.htm)
defines the fields, missing-value sentinel, and source citation. The helper
does not fetch or redistribute the dataset. Rows with missing, nonfinite,
zero, or negative mass/rate are omitted.

This is a supplementary data example for the scaling discussion in
[Algebra and Allometry](https://demonstrandom.com/symmetry/posts/allometry/).
It does not reproduce the historical Kleiber or Huxley figures shown there.

The scatter plot is descriptive. It does not fit an exponent, adjust for
phylogeny, or establish a causal scaling law. The code was developed with
AI assistance; automated checks do not replace scientific review.

The measurements are plotted as supplied after the stated filtering; no outlier
correction is applied. A fitted biological analysis requires its own
data-quality assessment and modeling assumptions.

## Files and provenance

The reader returns a two-column NumPy array: mass in kilograms and oxygen
consumption in milliliters per hour. The plotting function writes to the path
supplied by the caller and returns the number of plotted rows.

Tests use small synthetic TSV files to verify units, rejected measurements,
required columns, and the distinction between study mass and adult mass.
[provenance.json](provenance.json) records the source and prepared hashes.

Original code and documentation use
[PolyForm Noncommercial 1.0.0](LICENSE). Preserve [NOTICE](NOTICE).
PanTHERIA data and installed dependencies retain their own terms.
