# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Experimental AI-assisted research code; human review is incomplete.
"""A 93-coordinate diagnostic fingerprint for three-player, three-strategy games.
Input u[p,s1,s2,s3] has shape (3,3,3,3). There are 42 quadratics, 49 cubic
slots, and two degree-six slots. Nine cubic slots are determinants of centered
3-by-3 pair blocks and vanish identically. The final two features are trace(M^3)
and det(M) for a quadratic Gram matrix. Random samples and greedy feature
deletion are empirical diagnostics, not proofs of orbit separation."""

from itertools import combinations, combinations_with_replacement
import json
import numpy as np

from .contrast_blocks_3x3 import (
    mean_zero_payoff, project_contrast_block, family_matrix, all_family_matrices
)


def _project_block_local(u_p, S):
    """Full coordinates of a contrast block, with constant axes removed; shape (3,)*len(S)."""
    full = project_contrast_block(u_p, S)
    sl = [slice(None)] * 3
    for axis in range(3):
        if axis not in S:
            sl[axis] = 0
    return full[tuple(sl)]


def degree2_features(u):
    """42 invariants: upper triangle of M_S for each non-empty S."""
    out = []
    for r in range(1, 4):
        for S in combinations((0, 1, 2), r):
            M = family_matrix(u, S)
            for i in range(3):
                for j in range(i, 3):
                    out.append(M[i, j])
    return np.array(out)


def degree3_features_extended(u):
    u0 = mean_zero_payoff(u)
    out = []

    # Main-effect polarizations
    for axis in (0, 1, 2):
        S = (axis,)
        Ts = [_project_block_local(u0[p], S) for p in range(3)]
        for p, q, r in combinations_with_replacement(range(3), 3):
            out.append(float(np.sum(Ts[p] * Ts[q] * Ts[r])))

    # Pairwise-block determinants
    for pair in combinations((0, 1, 2), 2):
        for p in range(3):
            T = _project_block_local(u0[p], pair)
            out.append(float(np.linalg.det(T)))

    # Three-way contractions
    Ds = [_project_block_local(u0[p], (0, 1, 2)) for p in range(3)]
    for p, q, r in combinations_with_replacement(range(3), 3):
        out.append(float(np.sum(Ds[p] * Ds[q] * Ds[r])))

    M3 = family_matrix(u, (0, 1, 2))
    out.append(float(np.trace(M3 @ M3 @ M3)))
    out.append(float(np.linalg.det(M3)))

    return np.array(out)


def fingerprint(u):
    """42 quadratics, 49 cubic slots, and two sextic Gram-matrix features."""
    return np.concatenate([degree2_features(u), degree3_features_extended(u)])


def apply_group_element(u, sigma1, sigma2, sigma3):
    inv1 = np.argsort(sigma1)
    inv2 = np.argsort(sigma2)
    inv3 = np.argsort(sigma3)
    return u[:, inv1][:, :, inv2][:, :, :, inv3]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_orbit_invariance(N=30, tol=1e-8):
    """For N random games, verify the fingerprint is (S_3)^3-invariant."""
    rng = np.random.default_rng(0)
    max_err = 0.0
    for trial in range(N):
        u = rng.standard_normal((3, 3, 3, 3))
        fp_u = fingerprint(u)
        # Random group element
        sigma1 = rng.permutation(3)
        sigma2 = rng.permutation(3)
        sigma3 = rng.permutation(3)
        u_rel = apply_group_element(u, sigma1, sigma2, sigma3)
        fp_rel = fingerprint(u_rel)
        err = float(np.max(np.abs(fp_u - fp_rel)))
        max_err = max(max_err, err)
    return max_err


def test_separation_random(N=500, tol=1e-6):
    rng = np.random.default_rng(1)
    fps = []
    labels = []
    for i in range(N):
        u = rng.standard_normal((3, 3, 3, 3))
        fp_u = fingerprint(u)
        fps.append(fp_u)
        labels.append(i)
        sigma1 = rng.permutation(3)
        sigma2 = rng.permutation(3)
        sigma3 = rng.permutation(3)
        u_rel = apply_group_element(u, sigma1, sigma2, sigma3)
        fps.append(fingerprint(u_rel))
        labels.append(i)
    fps = np.array(fps)
    labels = np.array(labels)
    M = len(fps)

    # Compare every pair; record cases where fingerprints match
    pair_matches = []
    pair_mismatches_in_same_orbit = []
    for i in range(M):
        for j in range(i + 1, M):
            diff = float(np.max(np.abs(fps[i] - fps[j])))
            same_orbit = (labels[i] == labels[j])
            close = diff < tol
            if close and not same_orbit:
                pair_matches.append((i, j, diff))
            if not close and same_orbit:
                pair_mismatches_in_same_orbit.append((i, j, diff))

    return {
        "false_merges": pair_matches,
        "broken_orbit_mates": pair_mismatches_in_same_orbit,
        "n_games": N,
        "n_total": M,
    }


def test_atlas_separation(atlas_path="atlas_3x3_results.json"):
    # Rebuild atlas fingerprints from scratch (the JSON has limited diagnostics)
    from .atlas_3x3 import NAMED_GAMES
    fps = {}
    for name, builder in NAMED_GAMES.items():
        u = builder()
        fps[name] = fingerprint(u)
    games = list(fps.keys())
    n = len(games)
    collisions = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = float(np.max(np.abs(fps[games[i]] - fps[games[j]])))
            if diff < 1e-6:
                collisions.append((games[i], games[j], diff))
    return collisions, fps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _structured_orbit_samples(N_random=80):
    from .atlas_3x3 import NAMED_GAMES
    samples = []
    rng = np.random.default_rng(7)

    for _ in range(N_random):
        samples.append(rng.standard_normal((3, 3, 3, 3)))

    for builder in NAMED_GAMES.values():
        u = builder()
        samples.append(u)
        # Perturbations of atlas games
        for eps in (0.01, 0.1):
            samples.append(u + eps * rng.standard_normal((3, 3, 3, 3)))

    # Sparse / structured games
    for _ in range(20):
        u = rng.standard_normal((3, 3, 3, 3))
        mask = rng.random((3, 3, 3, 3)) < 0.3
        samples.append(u * mask)

    return samples


def minimum_separating_subset(N=200, tol=1e-6, verbose=False):
    """Greedy sample-label distinction; return None on a collision. No global guarantee."""
    rng = np.random.default_rng(123)
    base_samples = _structured_orbit_samples(N_random=N)
    fps_full = []
    labels = []
    for i, u in enumerate(base_samples):
        fps_full.append(fingerprint(u))
        labels.append(i)
        # Add the orbit mate
        sigma1 = rng.permutation(3)
        sigma2 = rng.permutation(3)
        sigma3 = rng.permutation(3)
        u_rel = apply_group_element(u, sigma1, sigma2, sigma3)
        fps_full.append(fingerprint(u_rel))
        labels.append(i)
    fps_full = np.array(fps_full)
    labels = np.array(labels)

    n_features = fps_full.shape[1]
    keep = set(range(n_features))

    def separates(idxs):
        if not idxs:
            return False
        sub = fps_full[:, sorted(idxs)]
        # Vectorized pairwise check: max-norm distances
        # For each pair (i, j) with i < j: dist = max(|sub[i] - sub[j]|)
        # Find the minimum cross-orbit distance.
        n = sub.shape[0]
        for i in range(n - 1):
            diffs = np.max(np.abs(sub[i + 1:] - sub[i]), axis=1)
            # Mask pairs in same orbit
            same_orbit = (labels[i + 1:] == labels[i])
            cross_diffs = diffs[~same_orbit]
            if cross_diffs.size > 0 and cross_diffs.min() < tol:
                return False
        return True

    if not separates(keep):
        return None  # At least two sample labels have indistinguishable fingerprints.

    changed = True
    while changed:
        changed = False
        # Try dropping features in order
        for i in sorted(keep):
            trial = keep - {i}
            if separates(trial):
                keep = trial
                changed = True
                if verbose:
                    print(f"    dropped feature {i}; {len(keep)} remain", flush=True)
                break
    return sorted(keep)


def main():
    print("Orbit-separation tests for (3,3)-games")
    print("=" * 70)

    # Sample fingerprint size
    rng = np.random.default_rng(99)
    u_sample = rng.standard_normal((3, 3, 3, 3))
    fp_sample = fingerprint(u_sample)
    print(f"Fingerprint size: {fp_sample.shape[0]} invariants")
    print(f"  ({degree2_features(u_sample).shape[0]} degree-2 + "
          f"{degree3_features_extended(u_sample).shape[0]} cubic/sextic slots)")
    print()

    # Test A: orbit invariance
    print("(A) Orbit invariance: applying random g in (S_3)^3 to random games ...")
    max_err = test_orbit_invariance(N=30)
    print(f"  max |fingerprint(u) - fingerprint(g.u)| over 30 trials: {max_err:.2e}")
    assert max_err < 1e-8, "fingerprint is NOT G-invariant"
    print("  PASS: fingerprint is (S_3)^3-invariant.")
    print()

    # Test B: collision check on random sample
    print("(B) Collision check on N=500 random games + orbit-mates ...")
    result = test_separation_random(N=500, tol=1e-6)
    n_false = len(result["false_merges"])
    n_broken = len(result["broken_orbit_mates"])
    print(f"  pairs in different orbits with same fingerprint (FALSE MERGES): {n_false}")
    print(f"  pairs in same orbit with different fingerprint (BROKEN INVARIANCE): {n_broken}")
    assert n_broken == 0
    if n_false == 0:
        print("  PASS: no false merges. No collisions were found in this finite sample.")
    else:
        print(f"  FAIL: {n_false} false merges. Examples:")
        for i, j, d in result["false_merges"][:3]:
            print(f"    games {i} and {j} differ by {d:.2e} but in different orbits")
    print()

    # Test C: atlas separation
    print("(C) Atlas separation: all 13 named games have distinct fingerprints?")
    collisions, fps = test_atlas_separation()
    if collisions:
        print(f"  COLLISIONS: {len(collisions)} pairs")
        for a, b, d in collisions:
            print(f"    {a} <-> {b}: max diff {d:.2e}")
    else:
        print("  PASS: all 13 atlas games have distinct fingerprints.")
    print()

    # Test D: empirical minimum separating subset against a STRESS-TEST sample.
    # Note: greedy elimination on a finite sample returns few features, since
    # any non-constant invariant has distinct values on N generic points. The
    # resulting count is not a meaningful lower bound on the minimum separating
    # set size for the full orbit space. The low-degree atlas uses 42 + 556 =
    # 598 invariants (degrees 2 and 3), which span the degree-<=3 invariant
    # subspace; orbit separation is not claimed for this set.
    print("(D) Greedy minimum on stress-test sample ...")
    print("    (random + atlas + perturbations + sparse, with orbit-mates)")
    kept = minimum_separating_subset(N=80, tol=1e-4, verbose=False)
    print(f"  Greedy result on this sample: {len(kept) if kept is not None else 'no separating subset'} features suffice.")
    print(f"  CAVEAT: this is sample-specific, not a true separation lower bound.")
    print("  The separate 598-coordinate atlas spans degrees 2 and 3 only.")
    print("  Neither atlas has a global separation guarantee here.")
    print()

    print("=" * 70)
    print("Summary:")
    print(f"  Fingerprint = {fp_sample.shape[0]} invariants "
          f"({degree2_features(u_sample).shape[0]} deg-2 + "
          f"{degree3_features_extended(u_sample).shape[0]} cubic/sextic slots)")
    print(f"  Orbit-invariant: max err {max_err:.2e}")
    print(f"  False merges on random sample (N=500, 1000 games total): {n_false}")
    print(f"  Atlas separation: {len(collisions)} collisions")
    print(f"  Greedy sample subset: {len(kept) if kept is not None else 'no separating subset'} features")


if __name__ == "__main__":
    main()
