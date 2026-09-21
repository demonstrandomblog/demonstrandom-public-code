# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Experimental AI-assisted research code; human review is incomplete.
"""Count degree-three invariant dimensions by contrast-block triples.
Three players each have three strategies. A block label specifies payoff player
and a nonempty subset of strategy axes. Character formulas distinguish tensor,
symmetric-square, and symmetric-cube contributions. The result counts a graded
space; it does not certify a generating set in all degrees."""

from collections import Counter
from itertools import combinations_with_replacement


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

# 7 contrast types for (3,3)-games. Using 0-indexed axes.
TYPES = [
    (0,), (1,), (2,),         # 3 main effects (|S| = 1)
    (0, 1), (0, 2), (1, 2),   # 3 pairwise interactions (|S| = 2)
    (0, 1, 2),                # 1 three-way interaction (|S| = 3)
]
TYPE_LABELS = ["{1}", "{2}", "{3}", "{1,2}", "{1,3}", "{2,3}", "{1,2,3}"]

# Character of W_3 = standard 2-dim rep of S_3, evaluated on conjugacy classes:
#   id (size 1):           tr(g | W_3) = 2,     tr(g^2 | W_3) = 2,    tr(g^3 | W_3) = 2
#   transpositions (3):    tr(g | W_3) = 0,     tr(g^2 | W_3) = 2,    tr(g^3 | W_3) = 0
#   3-cycles (2):          tr(g | W_3) = -1,    tr(g^2 | W_3) = -1,   tr(g^3 | W_3) = 2
S3_CLASS_DATA = [
    # (size, chi(g), chi(g^2), chi(g^3))
    (1, 2, 2, 2),
    (3, 0, 2, 0),
    (2, -1, -1, 2),
]


def axis_count(types, axis):
    """How many of the given types contain the given axis."""
    return sum(1 for S in types if axis in S)


# ---------------------------------------------------------------------------
# Invariant dimension formulas via character theory
# ---------------------------------------------------------------------------

def inv_dim_tensor(c):
    if c == 0:
        return 1
    total = sum(sz * chi_g ** c for sz, chi_g, _, _ in S3_CLASS_DATA)
    assert total % 6 == 0
    return total // 6


def inv_dim_sym2(c):
    if c == 0:
        return 1
    total = 0
    for sz, chi_g, chi_g2, _ in S3_CLASS_DATA:
        total += sz * (chi_g ** (2 * c) + chi_g2 ** c)
    assert total % (2 * 6) == 0
    return total // (2 * 6)


def inv_dim_sym3(c):
    if c == 0:
        return 1
    total = 0
    for sz, chi_g, chi_g2, chi_g3 in S3_CLASS_DATA:
        total += sz * (
            chi_g ** (3 * c)
            + 3 * chi_g ** c * chi_g2 ** c
            + 2 * chi_g3 ** c
        )
    assert total % (6 * 6) == 0
    return total // (6 * 6)


def inv_dim_tensor_two_factors(c_a, c_b):
    return inv_dim_tensor(c_a + c_b)


def inv_dim_sym2_with_third(c_rep, c_other):
    total = 0
    for sz, chi_g, chi_g2, _ in S3_CLASS_DATA:
        v_a_chi = chi_g ** c_rep
        v_a_chi_g2 = chi_g2 ** c_rep
        sym2_chi = (v_a_chi * v_a_chi + v_a_chi_g2) // 2 if (v_a_chi * v_a_chi + v_a_chi_g2) % 2 == 0 else None
        # Handle the half by keeping it as rational
        sym2_chi_num = v_a_chi ** 2 + v_a_chi_g2
        v_b_chi = chi_g ** c_other
        total += sz * sym2_chi_num * v_b_chi
    assert total % (2 * 6) == 0
    return total // (2 * 6)


# ---------------------------------------------------------------------------
# Per-block-triple invariant dim
# ---------------------------------------------------------------------------

def block_triple_invariant_dim(triple_of_blocks):
    cnt = Counter(triple_of_blocks)
    sizes = sorted(cnt.values(), reverse=True)
    shape = tuple(sizes)

    if shape == (1, 1, 1):
        types_in_triple = [TYPES[t] for (t, _) in triple_of_blocks]
        per_axis = []
        for axis in range(3):
            c = axis_count(types_in_triple, axis)
            per_axis.append(inv_dim_tensor(c))
        return per_axis[0] * per_axis[1] * per_axis[2], shape

    elif shape == (2, 1):
        rep_block = next(b for b, c in cnt.items() if c == 2)
        other_block = next(b for b, c in cnt.items() if c == 1)
        rep_type = TYPES[rep_block[0]]
        other_type = TYPES[other_block[0]]
        per_axis = []
        for axis in range(3):
            c_rep = 1 if axis in rep_type else 0
            c_other = 1 if axis in other_type else 0
            per_axis.append(inv_dim_sym2_with_third(c_rep, c_other))
        return per_axis[0] * per_axis[1] * per_axis[2], shape

    elif shape == (3,):
        rep_type = TYPES[triple_of_blocks[0][0]]
        per_axis = []
        for axis in range(3):
            c = 1 if axis in rep_type else 0
            per_axis.append(inv_dim_sym3(c))
        return per_axis[0] * per_axis[1] * per_axis[2], shape

    else:
        raise ValueError(f"unknown partition shape {shape}")


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------

def enumerate_block_triples():
    blocks = [(t, p) for t in range(7) for p in range(3)]  # 21 blocks
    for triple in combinations_with_replacement(blocks, 3):
        types_in_triple = [TYPES[t] for (t, _) in triple]
        inv_dim, shape = block_triple_invariant_dim(triple)
        yield triple, types_in_triple, shape, inv_dim


def main():
    print("Enumeration of degree-3 generators for R[V^0]^{(S_3)^3}")
    print("=" * 78)
    print()

    total = 0
    by_type_combo = {}
    by_shape = Counter()
    total_block_triples = 0
    triples_with_invariants = 0

    for triple, types_in_triple, shape, inv_dim in enumerate_block_triples():
        total_block_triples += 1
        if inv_dim > 0:
            triples_with_invariants += 1
        total += inv_dim
        type_combo = tuple(sorted(t for (t, _) in triple))
        by_type_combo.setdefault(type_combo, {"total": 0, "n_triples": 0, "shapes": Counter()})
        by_type_combo[type_combo]["total"] += inv_dim
        by_type_combo[type_combo]["n_triples"] += 1
        by_type_combo[type_combo]["shapes"][shape] += 1
        by_shape[shape] += inv_dim

    print(f"Total block triples enumerated: {total_block_triples}")
    print(f"  (expected: C(21+2, 3) = {(21 * 22 * 23) // 6})")
    print(f"Triples with at least one invariant: {triples_with_invariants}")
    print(f"Total degree-3 invariant dim: {total}")
    print(f"  (expected from Molien: 556)")
    if total != 556:
        print(f"  *** MISMATCH: enumeration gives {total}, not 556 ***")
    else:
        print(f"  VERIFIED.")
    print()

    print("Contribution by partition shape (over blocks):")
    for shape, cnt in by_shape.most_common():
        shape_str = "all-distinct (1,1,1)" if shape == (1,1,1) else \
                    "one-repeated (2,1)" if shape == (2,1) else \
                    "all-same (3)" if shape == (3,) else str(shape)
        print(f"  {shape_str}: {cnt}")
    print()

    print("=" * 78)
    print("Per type-combo breakdown")
    print("=" * 78)
    print()
    print(f"{'Type combo':<40s} {'block triples':>14s} {'inv dim':>10s}")
    print("-" * 78)
    sorted_combos = sorted(by_type_combo.items(), key=lambda kv: -kv[1]["total"])
    cumulative = 0
    for type_combo, info in sorted_combos:
        labels = [TYPE_LABELS[t] for t in type_combo]
        label_str = " . ".join(labels)
        print(f"{label_str:<40s} {info['n_triples']:>14d} {info['total']:>10d}")
        cumulative += info["total"]
    print("-" * 78)
    print(f"{'TOTAL':<40s} {total_block_triples:>14d} {cumulative:>10d}")
    print()

    # Spotlight: the largest contributors
    print("Top 10 type combos by contribution:")
    for type_combo, info in sorted_combos[:10]:
        labels = [TYPE_LABELS[t] for t in type_combo]
        shapes_str = ", ".join(f"{s}:{c}" for s, c in info["shapes"].items())
        print(f"  {' . '.join(labels):<35s} -> {info['total']:>4d} (shapes: {shapes_str})")


if __name__ == "__main__":
    main()
