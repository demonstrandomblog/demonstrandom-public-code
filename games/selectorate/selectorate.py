# AI-assisted research code; no blanket human or mathematical review is claimed.
# See this component's README.md and the repository AI_NOTICE.md before relying on results.
"""Selectorate theory model -- channel-based, with hierarchy as attenuation."""

import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class SelectorateEquilibrium:
    p_min: torch.Tensor              # minimum targeted transfer
    rents: torch.Tensor              # B - p_min
    coalition_payoff: torch.Tensor   # p_min / W
    defection_payoff: torch.Tensor   # B / S
    inclusion_prob: torch.Tensor     # W / S
    loyalty_margin: torch.Tensor     # coalition - defection


class SelectorateModel(nn.Module):
    """One-decision selectorate model: leader chooses targeted transfer p."""

    def __init__(self, W=10.0, S=100.0, B=100.0):
        super().__init__()
        self.W = nn.Parameter(torch.tensor(W))
        self.S = nn.Parameter(torch.tensor(S))
        self.B = nn.Parameter(torch.tensor(B))

    @property
    def r(self):
        return self.W / self.S

    def p_min(self):
        """Minimum targeted transfer to retain coalition loyalty."""
        return self.B * self.r

    def kappa(self):
        """Attenuation factor: fraction of transfer that passes through."""
        return 1.0 - self.r

    def tau(self):
        """Hierarchy tax: the cost multiplier this level imposes on the level above."""
        return 1.0 / torch.clamp(1.0 - self.r, min=1e-8)

    def forward(self):
        p = self.p_min()
        rents = self.B - p
        coalition_pay = p / self.W
        defection_pay = self.B / self.S
        return SelectorateEquilibrium(
            p_min=p, rents=rents,
            coalition_payoff=coalition_pay, defection_payoff=defection_pay,
            inclusion_prob=self.r,
            loyalty_margin=coalition_pay - defection_pay,
        )


def p_min_composed(top, *sub_levels):
    """Top-level p_min accounting for hierarchy attenuation.

    Each sub-level contributes a multiplicative hierarchy tax tau_k = 1/(1-r_k).
    """
    tau_total = torch.ones(1)
    for level in sub_levels:
        tau_total = tau_total * level.tau()
    return top.B * top.r * tau_total


def total_attenuation(*sub_levels):
    """Total attenuation across sub-levels: product of kappa_k = (1-r_k)."""
    kappa_total = torch.ones(1)
    for level in sub_levels:
        kappa_total = kappa_total * level.kappa()
    return kappa_total


def decision_count(r, s, n):
    """Total decisions across n levels with span s and coalition ratio r.

    Returns dict with allocation, loyalty, total, and population counts.
    """
    rs = r * s
    if abs(rs - 1.0) < 1e-10:
        alloc = float(n)
        loyal = float(n * rs)
    else:
        alloc = ((rs)**n - 1) / (rs - 1)
        loyal = rs * ((rs)**n - 1) / (rs - 1)
    return {
        'allocation': alloc,
        'loyalty': loyal,
        'total': alloc + loyal,
        'population': (rs)**(n-1) * s * (1 - r),
    }


# ---------------------------------------------------------------------------
# Channel coefficients (two-control model)
# ---------------------------------------------------------------------------

def channel_coefficients(W, S, N, beta=0.5):
    """Return the three key coefficients: 1/W, 1/S, beta/N.

    These determine the full selectorate geometry.
    """
    return {
        'loyalty_targeted': 1.0 / W,
        'defection_targeted': 1.0 / S,
        'universal': beta / N,
    }


def democratic_region(S, N, beta=0.5):
    """Whether the adversarial benchmark shifts to the universal channel.

    Returns True when beta/N >= 1/S, i.e., N <= beta*S.
    """
    return beta / N >= 1.0 / S


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_flat_model():
    print("=" * 60)
    print("FLAT MODEL TESTS")
    print("=" * 60)

    model = SelectorateModel(W=10.0, S=100.0, B=100.0)
    eq = model()

    # p_min = B * W/S = 100 * 0.1 = 10
    expected_p_min = 10.0
    print(f"p_min: {eq.p_min.item():.2f} (expected {expected_p_min:.2f})")
    assert abs(eq.p_min.item() - expected_p_min) < 1e-4

    # Loyalty margin should be ~0 at p_min (binding constraint)
    assert abs(eq.loyalty_margin.item()) < 1e-4
    print("  -> Loyalty margin ~ 0 at p_min [ok]")

    # Budget scaling: p_min scales linearly with B
    model2 = SelectorateModel(W=10.0, S=100.0, B=1000.0)
    assert abs(eq.p_min.item() / model.B.item() -
               model2().p_min.item() / model2.B.item()) < 1e-6
    print("  -> Budget scaling (p_min/B invariant) [ok]")

    # Population scaling
    model3 = SelectorateModel(W=50.0, S=500.0, B=100.0)
    assert abs(eq.p_min.item() / model.B.item() -
               model3().p_min.item() / model3.B.item()) < 1e-6
    print("  -> Population scaling [ok]")

    # tau = 1/(1-r) = 1/0.9 ~ 1.111
    tau_val = model.tau()
    expected_tau = 1.0 / (1.0 - 0.1)
    print(f"  tau = {tau_val.item():.4f} (expected {expected_tau:.4f})")
    assert abs(tau_val.item() - expected_tau) < 1e-4
    assert tau_val.item() >= 1.0
    print("  -> tau >= 1 [ok]")

    # kappa = 1 - r = 0.9
    kappa_val = model.kappa()
    print(f"  kappa = {kappa_val.item():.4f} (expected 0.9)")
    assert abs(kappa_val.item() - 0.9) < 1e-4
    print("  -> kappa = 1 - r [ok]")

    # Gradient
    model_grad = SelectorateModel(W=10.0, S=100.0, B=100.0)
    eq_grad = model_grad()
    eq_grad.rents.backward()
    print(f"\nd(rents)/dW = {model_grad.W.grad.item():.4f}")
    assert model_grad.W.grad.item() < 0, "d(rents)/dW should be negative"
    print("  -> d(rents)/dW < 0 [ok] (more W, less rents)")

    # Numerical examples
    print("\nNumerical examples:")
    for r, label in [(0.1, "Autocracy"), (0.3, "Junta"), (0.6, "Broad coalition"),
                     (0.8, "Near-democracy"), (1.0, "Full inclusion")]:
        W, S = r * 100, 100.0
        m = SelectorateModel(W=W, S=S, B=100.0)
        e = m()
        print(f"  r={r:.1f} ({label}): p_min={e.p_min.item():.1f}, "
              f"rents={e.rents.item():.1f}")
    print()


def test_composition():
    print("=" * 60)
    print("COMPOSITION TESTS")
    print("=" * 60)

    # Single level -- compose with no sub-levels should match flat
    top = SelectorateModel(W=10.0, S=100.0, B=100.0)
    p_flat = top.p_min()
    p_composed = p_min_composed(top)
    print(f"Flat (no sub-levels): p_min={p_flat.item():.2f}, "
          f"composed={p_composed.item():.2f}")
    assert abs(p_flat.item() - p_composed.item()) < 1e-4
    print("  -> Matches flat model [ok]")

    # Two-level: autocratic sub-leaders (r=0.1)
    sub_auto = SelectorateModel(W=10.0, S=100.0)
    p_2 = p_min_composed(top, sub_auto)
    tau_auto = sub_auto.tau()
    expected_tau = 1.0 / (1.0 - 0.1)
    kappa_auto = sub_auto.kappa()
    print(f"\nTwo-level, autocratic sub (r=0.1):")
    print(f"  kappa = {kappa_auto.item():.4f}")
    print(f"  tau = {tau_auto.item():.4f} (expected {expected_tau:.4f})")
    print(f"  p_min = {p_2.item():.4f} (expected {10.0 * expected_tau:.4f})")
    assert abs(tau_auto.item() - expected_tau) < 1e-4
    assert abs(p_2.item() - 10.0 * expected_tau) < 1e-3

    # Two-level: democratic sub-leaders (r=0.8)
    sub_dem = SelectorateModel(W=80.0, S=100.0)
    p_dem = p_min_composed(top, sub_dem)
    tau_dem = sub_dem.tau()
    expected_tau_dem = 1.0 / (1.0 - 0.8)
    print(f"\nTwo-level, democratic sub (r=0.8):")
    print(f"  kappa = {sub_dem.kappa().item():.4f}")
    print(f"  tau = {tau_dem.item():.4f} (expected {expected_tau_dem:.4f})")
    print(f"  p_min = {p_dem.item():.4f}")
    assert abs(tau_dem.item() - expected_tau_dem) < 1e-4
    assert p_dem.item() > p_2.item()
    print("  -> Democratic sub-leaders cost more [ok]")

    # Multi-level: compose many identical sub-levels
    print(f"\nMulti-level with identical autocratic sub-levels:")
    for n_sub in [0, 1, 2, 4, 9]:
        subs = [SelectorateModel(W=10.0, S=100.0) for _ in range(n_sub)]
        p = p_min_composed(top, *subs)
        kappa_tot = total_attenuation(*subs) if subs else torch.ones(1)
        expected_p = 10.0 * expected_tau ** n_sub
        print(f"  n={n_sub+1} levels: p_min={p.item():.4f} "
              f"(expected {expected_p:.4f}), kappa_tot={kappa_tot.item():.6f}")
        assert abs(p.item() - expected_p) < 1e-2
    print("  -> Exponential growth of p_min [ok]")

    # Total attenuation
    print(f"\nTotal attenuation:")
    subs_3 = [SelectorateModel(W=10.0, S=100.0) for _ in range(3)]
    kappa_3 = total_attenuation(*subs_3)
    expected_kappa = 0.9 ** 3
    print(f"  3 levels, r=0.1: kappa_tot = {kappa_3.item():.6f} "
          f"(expected {expected_kappa:.6f})")
    assert abs(kappa_3.item() - expected_kappa) < 1e-4
    print("  -> Multiplicative attenuation [ok]")

    # Decision counts
    print(f"\nDecision counts (s=100, r=0.1):")
    for n in [1, 2, 3, 5, 10]:
        dc = decision_count(0.1, 100.0, n)
        print(f"  n={n}: alloc={dc['allocation']:.0f}, loyalty={dc['loyalty']:.0f}, "
              f"total={dc['total']:.0f}, pop={dc['population']:.0f}")

    print()


def test_channels():
    print("=" * 60)
    print("CHANNEL COEFFICIENT TESTS")
    print("=" * 60)

    # Standard case
    coeffs = channel_coefficients(W=10, S=100, N=1000, beta=0.5)
    print(f"W=10, S=100, N=1000, beta=0.5:")
    print(f"  1/W = {coeffs['loyalty_targeted']:.4f}")
    print(f"  1/S = {coeffs['defection_targeted']:.4f}")
    print(f"  beta/N = {coeffs['universal']:.4f}")
    assert coeffs['loyalty_targeted'] > coeffs['defection_targeted']
    print("  -> 1/W > 1/S [ok] (incumbent advantage in targeted channel)")

    # Democratic region test
    assert not democratic_region(S=100, N=1000, beta=0.5)
    print(f"\n  N=1000, S=100, beta=0.5: democratic_region = False")
    print(f"    (1/S = 0.01 > beta/N = 0.0005)")

    assert democratic_region(S=100, N=40, beta=0.5)
    print(f"  N=40, S=100, beta=0.5: democratic_region = True")
    print(f"    (1/S = 0.01 < beta/N = 0.0125)")

    # Edge case: S = N/beta exactly
    assert democratic_region(S=100, N=50, beta=0.5)
    print(f"  N=50, S=100, beta=0.5: democratic_region = True (boundary)")

    print()


def test_gradients():
    print("=" * 60)
    print("GRADIENT TESTS")
    print("=" * 60)

    # Two-level: gradient of top p_min w.r.t. sub-level parameters
    top = SelectorateModel(W=10.0, S=100.0, B=100.0)
    sub = SelectorateModel(W=10.0, S=100.0)
    p = p_min_composed(top, sub)
    p.backward()

    print(f"p_min = {p.item():.6f}")
    print(f"dp/dW1 = {top.W.grad.item():.6f}")
    print(f"dp/dS1 = {top.S.grad.item():.6f}")
    print(f"dp/dW2 = {sub.W.grad.item():.6f}")
    print(f"dp/dS2 = {sub.S.grad.item():.6f}")

    assert top.W.grad.item() > 0, "dp/dW1 should be positive"
    print("  -> dp/dW1 > 0 [ok] (larger top coalition costs more)")
    assert sub.W.grad.item() > 0, "dp/dW2 should be positive"
    print("  -> dp/dW2 > 0 [ok] (larger sub-coalition costs more)")

    # Three-level
    print(f"\nThree-level model:")
    top3 = SelectorateModel(W=10.0, S=100.0, B=100.0)
    sub3a = SelectorateModel(W=10.0, S=100.0)
    sub3b = SelectorateModel(W=10.0, S=100.0)
    p3 = p_min_composed(top3, sub3a, sub3b)
    p3.backward()
    print(f"  p_min = {p3.item():.6f}")
    print(f"  dp/dW1 = {top3.W.grad.item():.6f}")
    print(f"  dp/dW2 = {sub3a.W.grad.item():.6f}")
    print(f"  dp/dW3 = {sub3b.W.grad.item():.6f}")
    assert sub3a.W.grad.item() > 0
    assert sub3b.W.grad.item() > 0
    print("  -> All sub-level W gradients positive [ok]")

    # Budget invariance: p_min/B doesn't depend on B
    print(f"\nBudget invariance:")
    top_b = SelectorateModel(W=10.0, S=100.0, B=100.0)
    ratio = top_b.p_min() / top_b.B
    ratio.backward()
    assert top_b.B.grad is None or abs(top_b.B.grad.item()) < 1e-6, \
        "B should not affect p_min/B"
    print("  -> p_min/B independent of B [ok] (budget invariance)")

    print()


if __name__ == "__main__":
    test_flat_model()
    test_composition()
    test_channels()
    test_gradients()
    print("All tests passed.")
