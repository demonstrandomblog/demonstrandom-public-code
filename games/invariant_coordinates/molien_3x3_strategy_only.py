# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Experimental AI-assisted research code; human review is incomplete.
"""Exact Molien coefficients for three players with three strategies each.
Players remain distinguished. Independent permutations of the three strategy
axes form (S3)^3 of order 216. Removing each player's constant payoff leaves
78 coordinates. Newton identities and conjugacy-class averaging use Fractions."""

from fractions import Fraction


# S_3 conjugacy classes: (label, class_size).
S3_CLASSES = [('id', 1), ('tau', 3), ('gamma', 2)]


def fp_pow(cls_label: str, k: int) -> int:
    """Fixed points of sigma^k where sigma is in the named S_3 class."""
    if cls_label == 'id':
        return 3
    if cls_label == 'tau':
        return 3 if k % 2 == 0 else 1
    if cls_label == 'gamma':
        return 3 if k % 3 == 0 else 0
    raise ValueError(cls_label)


def molien_33_strategy_only(max_deg: int) -> list:
    G_order = 216
    h_total = [Fraction(0)] * (max_deg + 1)

    for c1, s1 in S3_CLASSES:
        for c2, s2 in S3_CLASSES:
            for c3, s3 in S3_CLASSES:
                class_size = s1 * s2 * s3
                # chi_mz(g^k) = chi_full(g^k) - 3
                chi_mz = [None] + [
                    3 * fp_pow(c1, k) * fp_pow(c2, k) * fp_pow(c3, k) - 3
                    for k in range(1, max_deg + 1)
                ]
                # Newton's identity: d * h_d^g = sum_{k=1..d} chi_mz(g^k) * h_{d-k}^g
                hg = [Fraction(0)] * (max_deg + 1)
                hg[0] = Fraction(1)
                for d in range(1, max_deg + 1):
                    s = sum(chi_mz[k] * hg[d - k] for k in range(1, d + 1))
                    hg[d] = Fraction(s, d)
                for d in range(max_deg + 1):
                    h_total[d] += class_size * hg[d]

    result = []
    for d in range(max_deg + 1):
        val = h_total[d] / G_order
        assert val.denominator == 1, f"non-integer h_{d}: {val}"
        result.append(val.numerator)
    return result


def main():
    print("Molien coefficients for (3,3)-games under (S_3)^3 on mean-zero subspace")
    print("=" * 78)
    h = molien_33_strategy_only(max_deg=4)
    for d, hd in enumerate(h):
        print(f"  h_{d} = {hd}")

    expected = [1, 0, 42, 556, 9057]
    print()
    print(f"Expected from paper: {expected}")
    print(f"Match: {h == expected}")
    assert h == expected, f"Mismatch: got {h}, expected {expected}"
    print("VERIFIED.")


if __name__ == '__main__':
    main()
