# AI-assisted research code; no blanket human or mathematical review is claimed.
# See this component's README.md and the repository AI_NOTICE.md before relying on results.
"""Tests for selectorate theory module (channel-based model)."""

import torch
from .selectorate import (
    SelectorateModel, p_min_composed, total_attenuation,
    decision_count, channel_coefficients, democratic_region,
)


def test_flat_basics():
    """p_min = B * W/S, loyalty margin = 0 at equilibrium."""
    m = SelectorateModel(W=10.0, S=100.0, B=100.0)
    eq = m()
    assert abs(eq.p_min.item() - 10.0) < 1e-4
    assert abs(eq.loyalty_margin.item()) < 1e-4
    assert abs(eq.rents.item() - 90.0) < 1e-4
    print("[ok] flat model basics")


def test_budget_invariance():
    """p_min/B should not depend on B."""
    m1 = SelectorateModel(W=20.0, S=100.0, B=100.0)
    m2 = SelectorateModel(W=20.0, S=100.0, B=1000.0)
    r1 = m1().p_min.item() / m1.B.item()
    r2 = m2().p_min.item() / m2.B.item()
    assert abs(r1 - r2) < 1e-6
    print("[ok] budget invariance")


def test_population_scaling():
    """(W, S) -> (kW, kS) leaves p_min/B invariant."""
    m1 = SelectorateModel(W=10.0, S=100.0, B=100.0)
    m2 = SelectorateModel(W=50.0, S=500.0, B=100.0)
    r1 = m1().p_min.item() / m1.B.item()
    r2 = m2().p_min.item() / m2.B.item()
    assert abs(r1 - r2) < 1e-6
    print("[ok] population scaling")


def test_monotonicity():
    """p_min increases with W (holding S, B fixed)."""
    prev = 0.0
    for w in [5, 10, 20, 40, 60, 80]:
        m = SelectorateModel(W=float(w), S=100.0, B=100.0)
        p = m().p_min.item()
        assert p > prev, f"p_min should increase: W={w}, p_min={p}"
        prev = p
    print("[ok] monotonicity in W")


def test_tau_and_kappa():
    """tau = 1/(1-r), kappa = 1-r, tau*kappa = 1."""
    m = SelectorateModel(W=10.0, S=100.0, B=100.0)
    tau = m.tau().item()
    kappa = m.kappa().item()
    assert abs(tau - 1.0 / 0.9) < 1e-4
    assert abs(kappa - 0.9) < 1e-4
    assert abs(tau * kappa - 1.0) < 1e-4
    print("[ok] tau and kappa")


def test_composition_flat():
    """Composing with no sub-levels matches flat model."""
    top = SelectorateModel(W=10.0, S=100.0, B=100.0)
    assert abs(top.p_min().item() - p_min_composed(top).item()) < 1e-4
    print("[ok] composition matches flat")


def test_composition_two_level():
    """Two-level hierarchy inflates p_min by tau."""
    top = SelectorateModel(W=10.0, S=100.0, B=100.0)
    sub = SelectorateModel(W=10.0, S=100.0)
    p = p_min_composed(top, sub).item()
    expected = 10.0 / 0.9  # p_min * tau
    assert abs(p - expected) < 1e-3
    print("[ok] two-level composition")


def test_composition_exponential():
    """p_min grows exponentially with hierarchy depth."""
    top = SelectorateModel(W=10.0, S=100.0, B=100.0)
    tau = 1.0 / 0.9
    for n_sub in [1, 2, 4, 9]:
        subs = [SelectorateModel(W=10.0, S=100.0) for _ in range(n_sub)]
        p = p_min_composed(top, *subs).item()
        expected = 10.0 * tau ** n_sub
        assert abs(p - expected) < 1e-2, f"n={n_sub}: {p} vs {expected}"
    print("[ok] exponential growth with depth")


def test_total_attenuation():
    """Product of kappa values."""
    subs = [SelectorateModel(W=10.0, S=100.0) for _ in range(3)]
    kappa = total_attenuation(*subs).item()
    assert abs(kappa - 0.9 ** 3) < 1e-4
    print("[ok] total attenuation")


def test_democratic_sub_costs_more():
    """Democratic sub-leaders cost more than autocratic ones."""
    top = SelectorateModel(W=10.0, S=100.0, B=100.0)
    sub_auto = SelectorateModel(W=10.0, S=100.0)
    sub_dem = SelectorateModel(W=80.0, S=100.0)
    p_auto = p_min_composed(top, sub_auto).item()
    p_dem = p_min_composed(top, sub_dem).item()
    assert p_dem > p_auto
    print("[ok] democratic sub-leaders cost more")


def test_channel_coefficients():
    """Three coefficients computed correctly."""
    c = channel_coefficients(W=10, S=100, N=1000, beta=0.5)
    assert abs(c['loyalty_targeted'] - 0.1) < 1e-6
    assert abs(c['defection_targeted'] - 0.01) < 1e-6
    assert abs(c['universal'] - 0.0005) < 1e-6
    assert c['loyalty_targeted'] > c['defection_targeted']
    print("[ok] channel coefficients")


def test_democratic_region():
    """Democratic region: beta/N >= 1/S."""
    assert not democratic_region(S=100, N=1000, beta=0.5)
    assert democratic_region(S=100, N=40, beta=0.5)
    assert democratic_region(S=100, N=50, beta=0.5)  # boundary
    print("[ok] democratic region")


def test_decision_count():
    """Decision counts are positive and loyalty dominates."""
    dc = decision_count(0.1, 100.0, 3)
    assert dc['allocation'] > 0
    assert dc['loyalty'] > dc['allocation']
    assert dc['total'] == dc['allocation'] + dc['loyalty']
    assert dc['population'] > 0
    print("[ok] decision counts")


def test_gradient_signs():
    """Key comparative statics as gradient signs."""
    # d(rents)/dW < 0
    m = SelectorateModel(W=10.0, S=100.0, B=100.0)
    eq = m()
    eq.rents.backward()
    assert m.W.grad.item() < 0, "d(rents)/dW should be negative"

    # dp/dW1 > 0 in two-level
    top = SelectorateModel(W=10.0, S=100.0, B=100.0)
    sub = SelectorateModel(W=10.0, S=100.0)
    p = p_min_composed(top, sub)
    p.backward()
    assert top.W.grad.item() > 0, "dp/dW1 should be positive"
    assert sub.W.grad.item() > 0, "dp/dW2 should be positive"
    print("[ok] gradient signs")


if __name__ == "__main__":
    test_flat_basics()
    test_budget_invariance()
    test_population_scaling()
    test_monotonicity()
    test_tau_and_kappa()
    test_composition_flat()
    test_composition_two_level()
    test_composition_exponential()
    test_total_attenuation()
    test_democratic_sub_costs_more()
    test_channel_coefficients()
    test_democratic_region()
    test_decision_count()
    test_gradient_signs()
    print("\nAll tests passed.")
