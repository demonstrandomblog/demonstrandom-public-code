# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Experimental AI-assisted research code; human review is incomplete.
"""Sampled ranks of products of a 93-coordinate game fingerprint and stratum diagnostics.
Games have shape (3,3,3,3), payoff player first. Features have degrees 2,3,6;
some are identically zero. Evaluation-matrix rank is bounded by the number of
points and is tolerance-dependent. Agreement with a Molien coefficient is
numerical evidence only; a sampled rank deficit does not certify an algebraic gap."""

from itertools import combinations_with_replacement
import json
import numpy as np

from .orbit_separation_3x3 import (
    fingerprint,
    degree2_features,
    degree3_features_extended,
    apply_group_element,
)
from .molien_3x3_strategy_only import molien_33_strategy_only


# ---------------------------------------------------------------------------
# (1) Subalgebra Hilbert series via numerical rank
# ---------------------------------------------------------------------------


def compute_subalgebra_hilbert(max_degree=6, N_points=800, tol=1e-6, verbose=True):
    # Gram entries are quadratic; trace(M^3) and det(M) are sextic.
    fp_degrees = [2] * 42 + [3] * 49 + [6] * 2
    n_inv = len(fp_degrees)

    if verbose:
        print(f"Evaluating fingerprint at {N_points} random V^0 points ...",
              flush=True)
    rng = np.random.default_rng(11)
    sample_pts = [rng.standard_normal((3, 3, 3, 3)) for _ in range(N_points)]
    F_eval = np.array([fingerprint(u) for u in sample_pts])  # (N_points, 93)

    # For each total degree d, enumerate monomials in the 93 invariants
    # whose total *fingerprint degree* (sum of fp_degrees of chosen invariants)
    # equals d. Evaluate each monomial as a product across the sample.
    if verbose:
        print(f"Enumerating monomials in subalgebra coordinates up to degree {max_degree} ...",
              flush=True)
    hilb = [0] * (max_degree + 1)
    hilb[0] = 1  # constants

    # For each abstract polynomial degree d, the contributing monomials are
    # m_{i1} m_{i2} ... m_{ik} where i_1 <= i_2 <= ... and fp_degrees sum to d.
    # Enumerate by length k = 1, 2, 3, ... and total degree.
    for d in range(1, max_degree + 1):
        # Build all unordered tuples of fingerprint indices whose degree sums to d.
        mono_evals = []
        # Minimum degree is two, so at most d//2 factors contribute.
        for k in range(1, d // 2 + 1):
            for idxs in combinations_with_replacement(range(n_inv), k):
                if sum(fp_degrees[i] for i in idxs) == d:
                    val = np.ones(N_points)
                    for i in idxs:
                        val *= F_eval[:, i]
                    mono_evals.append(val)
        if not mono_evals:
            hilb[d] = 0
            if verbose:
                print(f"  degree {d}: no monomials -> dim 0", flush=True)
            continue

        M = np.array(mono_evals)  # (n_monos, N_points)
        rank = int(np.linalg.matrix_rank(M, tol=tol))
        hilb[d] = rank
        if verbose:
            print(f"  degree {d}: {len(mono_evals)} monomials, rank = {rank}",
                  flush=True)

    return hilb


def compare_to_molien(subalgebra_hilbert, max_degree):
    """Compare subalgebra Hilbert series to full Molien series."""
    full = molien_33_strategy_only(max_degree)
    print()
    print("Hilbert series comparison: subalgebra vs full invariant ring")
    print("=" * 70)
    print(f"{'degree':>6} {'subalgebra':>12} {'full ring':>12} {'gap':>8}")
    print("-" * 70)
    for d in range(max_degree + 1):
        gap = full[d] - subalgebra_hilbert[d]
        flag = "" if gap == 0 else "  <-- gap" if gap > 0 else "  ERROR"
        print(f"{d:>6} {subalgebra_hilbert[d]:>12} {full[d]:>12} {gap:>+8}{flag}")
    print()
    total_gap_low = sum(full[d] - subalgebra_hilbert[d] for d in range(min(5, len(full))))
    return full, total_gap_low


# ---------------------------------------------------------------------------
# (2) Stress-tests on special strata
# ---------------------------------------------------------------------------


def _gen_random(rng):
    return rng.standard_normal((3, 3, 3, 3))


def _gen_symmetric_payoff(rng):
    # Generate via a single random function on (own_strategy, multiset of others)
    # multiset of 2 strategies from {0,1,2}: 6 possibilities
    # Total entries: 3 (own) * 6 (multiset) = 18 random numbers
    base = rng.standard_normal(18)
    u = np.zeros((3, 3, 3, 3))
    multiset_index = {}
    idx = 0
    for a in range(3):
        for b in range(a, 3):
            multiset_index[(a, b)] = idx
            multiset_index[(b, a)] = idx
            idx += 1
    for p in range(3):
        for s1 in range(3):
            for s2 in range(3):
                for s3 in range(3):
                    profile = [s1, s2, s3]
                    own = profile[p]
                    others = sorted(profile[q] for q in range(3) if q != p)
                    mi = multiset_index[(others[0], others[1])]
                    u[p, s1, s2, s3] = base[own * 6 + mi]
    return u


def _gen_low_rank_payoff(rng, rank=2):
    u = np.zeros((3, 3, 3, 3))
    for p in range(3):
        for _ in range(rank):
            v1 = rng.standard_normal(3)
            v2 = rng.standard_normal(3)
            v3 = rng.standard_normal(3)
            u[p] += np.einsum('i,j,k->ijk', v1, v2, v3)
    return u


def _gen_sparse_payoff(rng, density=0.3):
    """Sparse: zero out (1 - density) fraction of payoff entries."""
    u = rng.standard_normal((3, 3, 3, 3))
    mask = rng.random((3, 3, 3, 3)) < density
    return u * mask


def _gen_near_degenerate(rng, eps=0.05):
    """Near-degenerate: small perturbation of a degenerate (all-zero) tensor."""
    return eps * rng.standard_normal((3, 3, 3, 3))


STRATA = {
    "random": _gen_random,
    "symmetric": _gen_symmetric_payoff,
    "low_rank": _gen_low_rank_payoff,
    "sparse": _gen_sparse_payoff,
    "near_degenerate": _gen_near_degenerate,
}


def stress_test(N_per_stratum=50, tol=1e-6, verbose=True):
    rng = np.random.default_rng(99)
    results = {}
    if verbose:
        print("Stress tests on special strata")
        print("=" * 70)
    for stratum_name, gen in STRATA.items():
        if verbose:
            print(f"  stratum: {stratum_name} ({N_per_stratum} orbits) ...",
                  flush=True)
        fps = []
        labels = []
        for i in range(N_per_stratum):
            u = gen(rng)
            fps.append(fingerprint(u))
            labels.append(i)
            sigma1 = rng.permutation(3)
            sigma2 = rng.permutation(3)
            sigma3 = rng.permutation(3)
            u_rel = apply_group_element(u, sigma1, sigma2, sigma3)
            fps.append(fingerprint(u_rel))
            labels.append(i)

        fps = np.array(fps)
        labels = np.array(labels)
        n = len(fps)

        false_merges = 0
        max_in_orbit = 0.0
        min_cross_orbit = float("inf")
        for i in range(n - 1):
            diffs = np.max(np.abs(fps[i + 1:] - fps[i]), axis=1)
            same_orbit = (labels[i + 1:] == labels[i])
            in_orbit_diffs = diffs[same_orbit]
            cross_orbit_diffs = diffs[~same_orbit]
            if in_orbit_diffs.size > 0:
                max_in_orbit = max(max_in_orbit, float(in_orbit_diffs.max()))
            if cross_orbit_diffs.size > 0:
                min_cross_orbit = min(min_cross_orbit, float(cross_orbit_diffs.min()))
            false_merges += int((cross_orbit_diffs < tol).sum())

        results[stratum_name] = {
            "false_merges": false_merges,
            "max_in_orbit": max_in_orbit,
            "min_cross_orbit": min_cross_orbit,
        }
        if verbose:
            print(f"    false merges: {false_merges}")
            print(f"    max in-orbit diff: {max_in_orbit:.2e} (should be ~ machine eps)")
            print(f"    min cross-orbit diff: {min_cross_orbit:.2e}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("(3a) Stronger separation verification for the 93-invariant fingerprint")
    print("=" * 78)
    print()

    # Hilbert-series check
    print("[1/2] Subalgebra Hilbert-series check (vs full Molien)")
    hilb = compute_subalgebra_hilbert(max_degree=4, N_points=400, verbose=True)
    full, total_gap = compare_to_molien(hilb, max_degree=4)
    if total_gap == 0:
        print("PASS: subalgebra Hilbert series matches full Molien up to degree 4.")
        print("Sampled numerical ranks agree through degree 4; this is not an exact proof.")
    else:
        print(f"Numerical rank deficit through degree 4: {total_gap}.")
        print("Sampled rank is below Molien; finite sample size can itself cause a deficit.")
        print("Degree-four rank is capped by the 400 sampled points.")
    print()

    # Stress-test check
    print("[2/2] Stress-test on special strata")
    stress = stress_test(N_per_stratum=50, tol=1e-6)
    print()
    print("Summary of stress tests:")
    print(f"{'stratum':<20s} {'false_merges':>14s} {'max_in_orbit':>14s} {'min_cross':>14s}")
    print("-" * 70)
    total_merges = 0
    for stratum, info in stress.items():
        total_merges += info["false_merges"]
        print(f"{stratum:<20s} {info['false_merges']:>14d} "
              f"{info['max_in_orbit']:>14.2e} {info['min_cross_orbit']:>14.2e}")
    print()
    if total_merges == 0:
        print("PASS: no false merges in any stratum. No collision was found in these sampled games.")
    else:
        print(f"FAIL: {total_merges} false merges. The 93 invariants are not sufficient")
        print("on every stratum tested.")


if __name__ == "__main__":
    main()
