# AI-assisted research code; no blanket human or mathematical review is claimed.
# See README.md and the repository AI_NOTICE.md for scope and limitations.
"""Small analytic regressions; not broad numerical-conditioning coverage."""
import numpy as np
import pytest
from .data_driven_systems import (
    dmd, dmd_arnoldi, edmd, finite_difference, heat_diffusion,
    kdmd, linear_kernel, polynomial_basis, sindy, sindyc,
)


def test_dmd_recovers_exact_linear_dynamics():
    operator = np.diag([0.8, 1.1])
    snapshots = np.random.default_rng(4).normal(size=(2, 30))
    modes, values, amplitudes, _ = dmd(snapshots, operator @ snapshots)
    np.testing.assert_allclose(np.sort(values), [0.8, 1.1], atol=1e-12)
    np.testing.assert_allclose(operator @ modes, modes * values, atol=1e-12)
    np.testing.assert_allclose(modes @ amplitudes, snapshots[:, 0], atol=1e-12)


def test_edmd_recovers_linear_observable_spectrum():
    snapshots = np.random.default_rng(4).normal(size=(2, 30))
    basis, _ = polynomial_basis(2, degree=1)
    _, values, *_ = edmd(snapshots, np.diag([0.8, 1.1]) @ snapshots, basis)
    np.testing.assert_allclose(np.sort(values), [0.8, 1.0, 1.1], atol=1e-12)


def test_linear_kernel_dmd_recovers_nonzero_spectrum():
    snapshots = np.random.default_rng(4).normal(size=(2, 30))
    values, _, _ = kdmd(snapshots, np.diag([0.8, 1.1]) @ snapshots, linear_kernel, r=2)
    values = values[np.abs(values) > 1e-10]
    np.testing.assert_allclose(np.sort(values.real), [0.8, 1.1], atol=1e-12)
    np.testing.assert_allclose(values.imag, 0, atol=1e-12)


def test_sindy_recovers_oscillator_from_exact_derivatives():
    states = np.random.default_rng(6).normal(size=(2, 100))
    derivatives = np.column_stack([states[1], -4 * states[0]])
    coefficients, names = sindy(states, derivatives, poly_order=2, lambda_reg=0.1)
    expected = np.zeros_like(coefficients)
    expected[names.index('x_1'), 0] = 1
    expected[names.index('x_0'), 1] = -4
    np.testing.assert_allclose(coefficients, expected, atol=1e-12)


def test_sindyc_recovers_control_driven_trajectory():
    time = np.linspace(0, 8, 8001)
    states = np.sin(time)[None, :]
    controls = np.cos(time)[None, :]
    coefficients, names = sindyc(states, controls, time[1]-time[0], poly_order_x=1, include_cross=False, lambda_reg=0.1)
    expected = np.zeros_like(coefficients)
    expected[names.index('u_0'), 0] = 1
    np.testing.assert_allclose(coefficients, expected, atol=1e-5)


def test_heat_diffusion_uses_requested_physical_parameters():
    field = heat_diffusion(time=0.5, m=5, n=5, alpha=0.2, dt=0.5, dx=2)
    expected = np.zeros((5, 5))
    expected[2, 2] = 90
    expected[1, 2] = expected[3, 2] = expected[2, 1] = expected[2, 3] = 2.5
    np.testing.assert_allclose(field, expected, atol=1e-12)


def test_arnoldi_remains_explicitly_unimplemented():
    with pytest.raises(NotImplementedError, match='use dmd'):
        dmd_arnoldi(np.eye(2), np.eye(2), 2)
