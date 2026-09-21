# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Experimental AI-assisted code; full human and scientific review is not established.
"""Check units, invalid measurements, and plotting without network access."""
import numpy as np
import pytest
from .allometry import MASS_COLUMN, BMR_COLUMN, read_measurements, plot_pantheria


def test_units_and_filtering(tmp_path):
    source=tmp_path/'data.tsv'
    source.write_text(MASS_COLUMN+'\t'+BMR_COLUMN+'\n1000\t60\n250\t12\n-999\t5\n5\t0\nNaN\t2\n5\tinf\n\t5\n')
    np.testing.assert_array_equal(read_measurements(source), [[1,60],[0.25,12]])
    output=tmp_path/'plot.png'
    assert plot_pantheria(source,output)==2
    assert output.read_bytes().startswith(b'\x89PNG')


def test_empty_and_invalid_schema(tmp_path):
    source=tmp_path/'bad.tsv'
    source.write_text('mass\trate\n1\t2\n')
    with pytest.raises(ValueError,match='columns'):
        read_measurements(source)
    source.write_text(MASS_COLUMN+'\t'+BMR_COLUMN+'\n-999\t-999\n')
    assert read_measurements(source).shape==(0,2)
    with pytest.raises(ValueError,match='No positive'):
        plot_pantheria(source,tmp_path/'empty.png')

def test_default_uses_metabolic_study_mass_without_adult_fallback(tmp_path):
    source=tmp_path/'two_mass_fields.tsv'
    source.write_text('5-1_AdultBodyMass_g\t5-2_BasalMetRateMass_g\t'+BMR_COLUMN+
                      '\n9000\t1000\t60\n5000\t-999\t12\n')
    np.testing.assert_array_equal(read_measurements(source), [[1,60]])
    np.testing.assert_array_equal(read_measurements(source,'adult'), [[9,60],[5,12]])
    assert plot_pantheria(source,tmp_path/'adult.png','adult')==2
    with pytest.raises(ValueError,match='mass_source'):
        read_measurements(source,'unknown')
