# AI-assisted experimental code; full human and mathematical review is not established.
"""
Fit candidate persistence functions to Hazen-Wong mineral evolution data.

Usage:
    python -m inspection_bias.fit
"""

import numpy as np
from scipy.optimize import minimize

from .persistence import (
    HAZEN_WONG_DATA, STAGE_NAMES, MODELS,
    compute_I, compute_log2_N,
)


def fit_model(name, factory, p0, data=None):
    """
    Fit a persistence model to Hazen-Wong data by minimizing
    sum of squared errors in functional information (bits).

    Returns (params, rmse, predictions).
    """
    if data is None:
        data = HAZEN_WONG_DATA

    def loss(params):
        log2_w = factory(params)
        return sum(
            (compute_I(k, m, n, log2_w) - I_obs) ** 2
            for k, m, n, M, I_obs in data
        )

    result = minimize(loss, p0, method="Nelder-Mead")
    if not result.success:
        raise RuntimeError(f"{name} fit did not converge: {result.message}")
    log2_w = factory(result.x)

    predictions = []
    for k, m, n, M, I_obs in data:
        I_pred = compute_I(k, m, n, log2_w)
        predictions.append((I_obs, I_pred))

    rmse = np.sqrt(result.fun / len(data))
    return result.x, rmse, predictions


def fit_all(data=None):
    """Fit all candidate models. Returns dict of results."""
    results = {}
    for name, spec in MODELS.items():
        params, rmse, preds = fit_model(
            name, spec["factory"], spec["p0"], data
        )
        results[name] = {
            "params": params,
            "param_names": spec["param_names"],
            "formula": spec["formula"],
            "rmse": rmse,
            "predictions": preds,
        }
    return results


def print_results(results):
    """Pretty-print fit results."""
    print(f"\n{'Model':<35} {'RMSE (bits)':>12}")
    print("-" * 50)
    for name, r in sorted(results.items(), key=lambda x: x[1]["rmse"]):
        params_str = ", ".join(
            f"{pn}={v:.3f}" for pn, v in zip(r["param_names"], r["params"])
        )
        print(f"{r['formula']:<35} {r['rmse']:>8.2f}    ({params_str})")

    # Detailed table for best model
    best_name = min(results, key=lambda k: results[k]["rmse"])
    best = results[best_name]
    print(f"\nBest model: {best['formula']} (RMSE = {best['rmse']:.2f} bits)")
    print(f"\n{'Stage':<22} {'I_obs':>7} {'I_pred':>7} {'Error':>7}")
    print("-" * 45)
    for stage, (I_obs, I_pred) in zip(STAGE_NAMES, best["predictions"]):
        print(f"{stage:<22} {I_obs:>7.1f} {I_pred:>7.1f} {I_pred - I_obs:>+7.1f}")


def phenomenological_scaling(data=None):
    """Check M ~ N^alpha phenomenological scaling."""
    if data is None:
        data = HAZEN_WONG_DATA

    log2_Ns = np.array([compute_log2_N(k, m, n) for k, m, n, M, I in data])
    log2_Ms = np.log2([M for k, m, n, M, I in data])

    coeffs = np.polyfit(log2_Ns, log2_Ms, 1)
    log2_M_pred = np.polyval(coeffs, log2_Ns)
    ss_res = np.sum((log2_Ms - log2_M_pred) ** 2)
    ss_tot = np.sum((log2_Ms - log2_Ms.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    print(f"\nPhenomenological scaling: M ~ N^{coeffs[0]:.4f}")
    print(f"  log2(M) = {coeffs[0]:.4f} * log2(N) + {coeffs[1]:.4f}")
    print(f"  R^2 = {r2:.4f}")
    print(f"  Equivalently: I ~ {1 - coeffs[0]:.4f} * log2(N)")
    return coeffs, r2


if __name__ == "__main__":
    results = fit_all()
    print_results(results)
    phenomenological_scaling()
