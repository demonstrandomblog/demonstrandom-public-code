# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Experimental AI-assisted research code; human review is incomplete.
"""Finite-example classification using contrast-family diagnostics.
Reads the JSON written by atlas_3x3. Family labels are nonempty subsets of the
three strategy axes; their matrices are Gram matrices across payoff players.
Sign and rank patterns use floating tolerances. Greedy deletion distinguishes
the supplied named games only, with no minimum-cardinality or global guarantee."""

import json
from itertools import combinations
import numpy as np


TOL = 1e-9


def sign3(x):
    """Three-valued sign: -1, 0, +1."""
    if x > TOL:
        return 1
    if x < -TOL:
        return -1
    return 0


def build_features(atlas):
    family_keys = ["{1}", "{2}", "{3}", "{1,2}", "{1,3}", "{2,3}", "{1,2,3}"]
    features = {}

    for name, rec in atlas.items():
        f = {}
        for S in family_keys:
            fm = rec["family_matrices"][S]
            M = np.array(fm["matrix"])
            tr = fm["trace"]
            rk = fm["rank_numerical"]
            offdiag_vals = M[~np.eye(3, dtype=bool)]
            offdiag_signs = set(sign3(v) for v in offdiag_vals)
            if 0 in offdiag_signs and len(offdiag_signs) > 1:
                offdiag_signs.discard(0)
            offdiag_sign_summary = (
                2 if len(offdiag_signs) > 1
                else next(iter(offdiag_signs)) if offdiag_signs
                else 0
            )

            f[f"layer1_{S}"] = int(rk > 0 or abs(tr) > TOL)
            f[f"rank_{S}"] = rk
            f[f"trace_sign_{S}"] = sign3(tr)
            f[f"offdiag_sign_{S}"] = offdiag_sign_summary
            f[f"det_sign_{S}"] = sign3(fm["det"])

        # Cross-family trace comparisons (a few useful ones)
        traces = {S: rec["family_matrices"][S]["trace"] for S in family_keys}
        f["cmp_main_vs_pair"] = sign3(
            traces["{1}"] + traces["{2}"] + traces["{3}"]
            - traces["{1,2}"] - traces["{1,3}"] - traces["{2,3}"]
        )
        f["cmp_pair_vs_three"] = sign3(
            traces["{1,2}"] + traces["{1,3}"] + traces["{2,3}"]
            - 3 * traces["{1,2,3}"]
        )

        # Degree-3 sign vector
        for dname, dval in rec["degree3_diagnostics"].items():
            f[f"d3_{dname}_sign"] = sign3(dval)

        features[name] = f

    return features


def feature_matrix(features, feature_keys):
    """Build a 2D array of feature values: rows = games, cols = features."""
    games = list(features.keys())
    mat = np.array([[features[g][k] for k in feature_keys] for g in games])
    return games, mat


def patterns_distinct(games, mat):
    """Return True if every pair of games has a distinct feature row."""
    seen = {}
    for i, g in enumerate(games):
        row = tuple(mat[i])
        if row in seen:
            return False, (seen[row], g)
        seen[row] = g
    return True, None


def minimal_classifying_subset(features, feature_keys, verbose=False):
    """Greedy backward elimination: drop features that aren't needed."""
    games, mat = feature_matrix(features, feature_keys)
    keep = list(range(len(feature_keys)))

    distinct, collision = patterns_distinct(games, mat[:, keep])
    if not distinct:
        if verbose:
            print(f"  WARNING: feature set does not separate all games; "
                  f"collision between {collision}")
        return [feature_keys[i] for i in keep]

    # Try to drop one feature at a time, lowest-index first
    changed = True
    while changed:
        changed = False
        for i in list(keep):
            trial = [j for j in keep if j != i]
            distinct, _ = patterns_distinct(games, mat[:, trial])
            if distinct:
                keep = trial
                changed = True
                if verbose:
                    print(f"  dropped {feature_keys[i]}; {len(keep)} features remain")
                break

    return [feature_keys[i] for i in keep]


def layer1_signature(atlas):
    """7-bit family-activation pattern per game (which contrast types nonzero)."""
    family_keys = ["{1}", "{2}", "{3}", "{1,2}", "{1,3}", "{2,3}", "{1,2,3}"]
    out = {}
    for name, rec in atlas.items():
        sig = tuple(
            int(rec["family_matrices"][S]["rank_numerical"] > 0
                or abs(rec["family_matrices"][S]["trace"]) > TOL)
            for S in family_keys
        )
        out[name] = sig
    return out, family_keys


def layer2_signature(atlas):
    """Layer-2 signature: (rank, offdiag-sign) per nonzero family."""
    family_keys = ["{1}", "{2}", "{3}", "{1,2}", "{1,3}", "{2,3}", "{1,2,3}"]
    out = {}
    for name, rec in atlas.items():
        sig = []
        for S in family_keys:
            fm = rec["family_matrices"][S]
            M = np.array(fm["matrix"])
            rk = fm["rank_numerical"]
            if rk == 0 and abs(fm["trace"]) < TOL:
                sig.append(("0", 0))
                continue
            offdiag = M[~np.eye(3, dtype=bool)]
            signs = set(sign3(v) for v in offdiag)
            if 0 in signs and len(signs) > 1:
                signs.discard(0)
            if len(signs) > 1:
                offdiag_label = "mixed"
            elif signs == {1}:
                offdiag_label = "+"
            elif signs == {-1}:
                offdiag_label = "-"
            else:
                offdiag_label = "0"
            sig.append((str(rk), offdiag_label))
        out[name] = tuple(sig)
    return out, family_keys


def report(atlas_path="atlas_3x3_results.json"):
    with open(atlas_path) as f:
        atlas = json.load(f)

    games = list(atlas.keys())
    print(f"Classifying invariants for {len(games)} named (3,3)-games\n")

    # ------- Layer 1 -------
    print("=" * 78)
    print("Layer 1: family activation pattern (which of 7 contrast types nonzero)")
    print("=" * 78)
    sigs_l1, fam_keys = layer1_signature(atlas)
    header = "Game".ljust(35) + " | " + "  ".join(s.ljust(7) for s in fam_keys)
    print(header)
    print("-" * len(header))
    pattern_groups = {}
    for name in games:
        sig = sigs_l1[name]
        pattern_groups.setdefault(sig, []).append(name)
        flags = "  ".join(("on" if b else "  ").ljust(7) for b in sig)
        nshort = name.replace("3p_", "").replace("_", " ")
        print(f"{nshort:<35s} | {flags}")
    print(f"\n  Distinct Layer-1 patterns: {len(pattern_groups)}")
    for sig, members in sorted(pattern_groups.items()):
        sig_str = "".join(str(b) for b in sig)
        labels = [m.replace("3p_", "").replace("_", " ") for m in members]
        print(f"    pattern {sig_str}: {labels}")

    # ------- Layer 2 -------
    print()
    print("=" * 78)
    print("Layer 2: per-family (rank, off-diag sign)")
    print("=" * 78)
    sigs_l2, _ = layer2_signature(atlas)
    l2_groups = {}
    for name in games:
        l2_groups.setdefault(sigs_l2[name], []).append(name)
    print(f"  Distinct Layer-2 signatures: {len(l2_groups)}")
    for sig, members in sorted(l2_groups.items()):
        sig_str = " ".join(f"{S}=({r},{o})" for S, (r, o) in zip(fam_keys, sig))
        labels = [m.replace("3p_", "").replace("_", " ") for m in members]
        print(f"    {sig_str}")
        for m in labels:
            print(f"      - {m}")
    if len(l2_groups) == len(games):
        print("  Layer 2 ALONE distinguishes all 13 games.")
    else:
        n_collisions = sum(1 for v in l2_groups.values() if len(v) > 1)
        print(f"  Layer 2 leaves {n_collisions} group(s) unresolved; "
              f"add Layer-3 degree-3 signs to discriminate.")

    # ------- Minimal classifying subset (greedy) -------
    print()
    print("=" * 78)
    print("Minimal classifying subset (greedy backward elimination)")
    print("=" * 78)
    features = build_features(atlas)
    all_keys = sorted(next(iter(features.values())).keys())
    # Stable order: layer1 first, then ranks, signs, then degree-3
    def order_key(k):
        if k.startswith("layer1_"): return (0, k)
        if k.startswith("rank_"):   return (1, k)
        if k.startswith("trace_sign_"):   return (2, k)
        if k.startswith("offdiag_sign_"): return (3, k)
        if k.startswith("det_sign_"):     return (4, k)
        if k.startswith("cmp_"):          return (5, k)
        if k.startswith("d3_"):           return (6, k)
        return (7, k)
    ordered_keys = sorted(all_keys, key=order_key)

    # First check whether the full set distinguishes
    full_games, full_mat = feature_matrix(features, ordered_keys)
    distinct_full, collision = patterns_distinct(full_games, full_mat)
    print(f"  Full feature set ({len(ordered_keys)} features) "
          f"distinguishes all games: {distinct_full}")
    if not distinct_full:
        print(f"    Unresolvable collision: {collision}")
    else:
        minimal = minimal_classifying_subset(features, ordered_keys, verbose=False)
        print(f"  Minimal classifying subset: {len(minimal)} features")
        for k in minimal:
            print(f"    - {k}")

    # Also: which features in the minimal set come from each layer?
    print()
    print("  Layer breakdown of minimal classifying set:")
    layer_counts = {}
    for k in minimal:
        layer = order_key(k)[0]
        layer_counts.setdefault(layer, []).append(k)
    layer_names = {0: "L1 activation", 1: "L2 rank", 2: "L2 trace-sign",
                   3: "L2 offdiag-sign", 4: "L2 det-sign", 5: "cross-cmp",
                   6: "L3 degree-3"}
    for layer in sorted(layer_counts):
        print(f"    {layer_names[layer]}: {len(layer_counts[layer])}")
        for k in layer_counts[layer]:
            print(f"      - {k}")


if __name__ == "__main__":
    report()
