# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Experimental AI-assisted code; full human and scientific review is not established.
"""Plot mammalian body mass and oxygen-consumption rate from a PanTHERIA TSV.

By default, pair 5-2_BasalMetRateMass_g with 18-1_BasalMetRate_mLO2hr:
the mass and oxygen consumption refer to the same measured individuals.
The optional adult source uses the general 5-1_AdultBodyMass_g species estimate.
Mass is converted from grams to kilograms. Metabolic rate stays in mL O2/hour;
conversion to watts would require an independently specified caloric equivalent.
Nonfinite and nonpositive measurements, including the -999 sentinel, are omitted.
"""
import argparse
import csv
from pathlib import Path
import numpy as np

MASS_COLUMNS = {'bmr': '5-2_BasalMetRateMass_g', 'adult': '5-1_AdultBodyMass_g'}
MASS_COLUMN = MASS_COLUMNS['bmr']
BMR_COLUMN = '18-1_BasalMetRate_mLO2hr'


def read_measurements(path, mass_source='bmr'):
    """Return positive finite (mass_kg, oxygen_ml_per_hour) rows.

    The default selects the mass reported with metabolic measurements.
    Select 'adult' explicitly for general adult-mass estimates. Missing study
    mass is excluded; it is never silently replaced by a different mass source.
    """
    if mass_source not in MASS_COLUMNS:
        raise ValueError("mass_source must be 'bmr' or 'adult'")
    mass_column = MASS_COLUMNS[mass_source]
    rows = []
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream, delimiter='\t')
        if not {mass_column, BMR_COLUMN}.issubset(reader.fieldnames or []):
            raise ValueError('Input lacks the required PanTHERIA mass/rate columns')
        for row in reader:
            try:
                mass, rate = float(row[mass_column]), float(row[BMR_COLUMN])
            except (ValueError, TypeError):
                continue
            if np.isfinite(mass) and np.isfinite(rate) and mass > 0 and rate > 0:
                rows.append((mass/1000.0, rate))
    return np.asarray(rows, dtype=float).reshape(-1, 2)


def plot_pantheria(input_path, output_path, mass_source='bmr'):
    """Save a base-10 log scatter plot and return the number of valid rows."""
    import matplotlib.pyplot as plt
    data = read_measurements(input_path, mass_source)
    if not len(data):
        raise ValueError('No positive finite mass/rate pairs')
    fig, ax = plt.subplots(figsize=(5, 5), dpi=120)
    ax.scatter(*np.log10(data).T, s=8, alpha=0.6, linewidths=0)
    mass_symbol = r'M_{\mathrm{BMR}}' if mass_source == 'bmr' else r'M_{\mathrm{adult}}'
    ax.set_xlabel(r'$\log_{10}(' + mass_symbol + r'/\mathrm{kg})$')
    ax.set_ylabel(r'$\log_{10}(B/[\mathrm{mL}\,\mathrm{O}_2\,\mathrm{h}^{-1}])$')
    ax.set_title('Mammalian mass and basal oxygen consumption')
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return len(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='Local PanTHERIA TSV')
    parser.add_argument('--output', type=Path, default=Path('pantheria_mass_bmr.png'))
    parser.add_argument('--mass-source', choices=tuple(MASS_COLUMNS), default='bmr',
                        help='bmr: mass accompanying metabolic measurement; adult: general species mass')
    args = parser.parse_args()
    print(f'Plotted {plot_pantheria(args.input, args.output, args.mass_source)} mass/rate pairs to {args.output}')


if __name__ == '__main__':
    main()
