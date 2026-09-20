# AI-assisted research code; no blanket human or mathematical review is claimed.
# See this component's README.md and the repository AI_NOTICE.md before relying on results.
"""Tests for selectorate theory module (channel-based model)."""

import pytest
import torch
from .selectorate import (
    SelectorateModel, p_min_composed, total_attenuation, r_eff_from_levels,
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


def test_composition_continued_fraction():
    expected = {1: 10.0, 2: 100/9, 3: 11.25, 30: 50*(1-0.6**0.5)}
    for depth, value in expected.items():
        levels = [SelectorateModel(W=10, S=100, B=100).double() for _ in range(depth)]
        assert p_min_composed(*levels).item() == pytest.approx(value, abs=1e-11)


def test_unequal_levels_and_attenuation():
    # .2/(1 - .3/(1 - .1)) = .3; reverse order yields .16.
    levels = [SelectorateModel(W=w, S=10, B=100).double() for w in (2, 3, 1)]
    assert p_min_composed(*levels).item() == pytest.approx(30)
    assert p_min_composed(*reversed(levels)).item() == pytest.approx(16)
    assert total_attenuation(*levels[1:]).item() == pytest.approx(2/3)
    assert total_attenuation().item() == 1
    assert total_attenuation(*[SelectorateModel() for _ in range(3)]).item() == pytest.approx(.8875)


def test_boundaries_and_budget():
    full = SelectorateModel(W=1, S=1, B=100)
    assert p_min_composed(full).item() == 100
    with pytest.raises(ValueError, match="attenuation"):
        p_min_composed(SelectorateModel(), full)
    with pytest.raises(ValueError, match="attenuation"):
        full.tau()
    levels = [SelectorateModel(W=w, S=10, B=100).double() for w in (9, 2)]
    # A finite but over-budget requirement is reported, not silently capped.
    assert p_min_composed(*levels).item() == pytest.approx(112.5)
    with pytest.raises(ValueError, match="attenuation"):
        p_min_composed(SelectorateModel(), *levels)
    with pytest.raises(ValueError, match="attenuation"):
        total_attenuation(*levels)
    with pytest.raises(ValueError, match="one level"):
        r_eff_from_levels()
    assert p_min_composed(SelectorateModel(B=0)).item() == 0


@pytest.mark.parametrize("kwargs", [{"W":0}, {"W":-1}, {"W":101}, {"S":0}, {"B":-1}, {"B":float("inf")}, {"W":float("nan")}])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        SelectorateModel(**kwargs)


def test_nested_gradients_and_dtype():
    levels = [SelectorateModel(W=w, S=10, B=100).double() for w in (2, 3, 1)]
    result = p_min_composed(*levels)
    assert result.dtype == torch.float64 and result.ndim == 0
    result.backward()
    for level in levels:
        analytic = level.W.grad.item()
        with torch.no_grad():
            original = level.W.item()
            level.W.fill_(original + 1e-5)
            plus = p_min_composed(*levels).item()
            level.W.fill_(original - 1e-5)
            minus = p_min_composed(*levels).item()
            level.W.fill_(original)
        assert analytic == pytest.approx((plus-minus)/2e-5, rel=1e-8)
    with torch.no_grad():
        levels[1].W.fill_(-1)
    with pytest.raises(ValueError):
        p_min_composed(*levels)


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
    test_composition_continued_fraction()
    test_unequal_levels_and_attenuation()
    test_democratic_sub_costs_more()
    test_channel_coefficients()
    test_democratic_region()
    test_decision_count()
    test_gradient_signs()
    print("\nAll tests passed.")
