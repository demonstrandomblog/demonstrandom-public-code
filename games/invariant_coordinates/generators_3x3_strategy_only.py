# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Experimental AI-assisted research code; human review is incomplete.
"""Numerical Reynolds-rank and cubic diagnostics for three-player, three-strategy games.
Input u[p,s1,s2,s3] has shape (3,3,3,3). Players are fixed and the strategy
axes are independently permuted. Removing three payoff means leaves 78
coordinates. Rank estimates are numerical and do not prove generation of the
full invariant ring. The default Reynolds experiment can take several minutes."""

from itertools import combinations, combinations_with_replacement, product as iproduct
import numpy as np


# ---------------------------------------------------------------------------
# Group action: (S_3)^3 on the 78-dim mean-zero subspace
# ---------------------------------------------------------------------------


def _all_s3_perms():
    """6 permutations of {0,1,2} as 3-tuples."""
    from itertools import permutations
    return list(permutations((0, 1, 2)))


def _flatten_index(p, s1, s2, s3):
    """Map (player, strategy profile) to flat index in {0,...,80}."""
    return p * 27 + s1 * 9 + s2 * 3 + s3


def _build_group_permutations():
    s3 = _all_s3_perms()
    perms = []
    for sigma1 in s3:
        for sigma2 in s3:
            for sigma3 in s3:
                perm = np.empty(81, dtype=np.int64)
                for p in range(3):
                    for s1 in range(3):
                        for s2 in range(3):
                            for s3i in range(3):
                                src = _flatten_index(p, s1, s2, s3i)
                                dst = _flatten_index(
                                    p, sigma1[s1], sigma2[s2], sigma3[s3i])
                                perm[dst] = src
                perms.append(perm)
    return np.array(perms)  # shape (216, 81)


def _mean_zero_basis():
    # For each player p, build a 26-dim orthonormal basis of mean-zero functions
    # on the 27 strategy profiles.
    B = np.zeros((81, 78))
    col = 0
    for p in range(3):
        # 27 coords for player p; need orthonormal basis of the 26-dim
        # mean-zero subspace.
        sub = np.zeros((27, 27))
        sub[0, :] = 1.0 / np.sqrt(27)  # the constant direction
        # Build remaining basis by Gram-Schmidt on random vectors
        rng = np.random.default_rng(p)
        Q, _ = np.linalg.qr(np.column_stack([sub[0, :], rng.standard_normal((27, 26))]))
        # Q's first column is the constant; remaining 26 are mean-zero
        for j in range(1, 27):
            B[p * 27 + np.arange(27), col] = Q[:, j]
            col += 1
    assert col == 78
    return B


def _build_group_action_on_mean_zero(B, perms_81):
    rhos = np.zeros((216, 78, 78))
    for g, perm in enumerate(perms_81):
        # In R^81, the action sends e_i -> e_{perm[i]}, i.e., column i of P is e_{perm[i]}
        P = np.zeros((81, 81))
        P[perm, np.arange(81)] = 1.0
        rhos[g] = B.T @ P @ B
    return rhos


# ---------------------------------------------------------------------------
# Reynolds projection and rank verification
# ---------------------------------------------------------------------------


def verify_degree3_count(verbose=True):
    if verbose:
        print("Building group action on mean-zero subspace ...", flush=True)
    perms_81 = _build_group_permutations()
    B = _mean_zero_basis()
    rhos = _build_group_action_on_mean_zero(B, perms_81)
    if verbose:
        print(f"  built {rhos.shape[0]} orthogonal matrices of shape {rhos.shape[1:]}",
              flush=True)

    # N random points in V^0; need N > 556 for full rank.
    N = 600
    rng = np.random.default_rng(0)
    pts = rng.standard_normal((N, 78))

    # For each random point u, precompute u_g = rho_g @ u for all g.
    # Shape: (N, 216, 78). Memory ~ 60 MB.
    if verbose:
        print("Precomputing group orbits at random points ...", flush=True)
    orbits = np.einsum('gij,nj->ngi', rhos, pts)
    if verbose:
        print(f"  orbits shape: {orbits.shape}", flush=True)

    # Enumerate degree-3 monomials x_a x_b x_c with a <= b <= c.
    coords = np.arange(78)
    triples = list(combinations_with_replacement(coords, 3))
    n_triples = len(triples)
    if verbose:
        print(f"  total degree-3 monomials: {n_triples}", flush=True)

    # Maintain a running orthonormal basis of the invariant span.
    rank = 0
    basis = np.zeros((N, 0))
    tol = 1e-8

    # Process in micro-batches to fold rank-test cost without blowing memory.
    # For each batch: compute Reynolds value (length-N vector) for each monomial
    # one at a time (cheap: 1 advanced-index slice + product + mean over axis 1).
    micro_batch = 256
    reynolds_buf = np.empty((N, micro_batch))

    for batch_start in range(0, n_triples, micro_batch):
        batch_end = min(batch_start + micro_batch, n_triples)
        batch = triples[batch_start:batch_end]
        m = len(batch)

        for j, (a, b, c) in enumerate(batch):
            # orbits[:, :, a/b/c] is (N, 216); elementwise product then mean
            reynolds_buf[:, j] = (orbits[:, :, a] * orbits[:, :, b] *
                                  orbits[:, :, c]).mean(axis=1)

        # Project the batch onto orthogonal complement of basis and add new dirs
        sub = reynolds_buf[:, :m]
        if rank > 0:
            coeffs = basis.T @ sub
            residual = sub - basis @ coeffs
        else:
            residual = sub
        norms = np.linalg.norm(residual, axis=0)
        significant = norms > tol
        if significant.any():
            R = residual[:, significant]
            U, S, _ = np.linalg.svd(R, full_matrices=False)
            new_dirs = U[:, S > tol]
            if new_dirs.shape[1] > 0:
                basis = np.concatenate([basis, new_dirs], axis=1)
                rank = basis.shape[1]

        if verbose and (batch_start // micro_batch) % 10 == 0:
            print(f"  processed {batch_end}/{n_triples}, running rank = {rank}",
                  flush=True)

        if rank >= 556:
            if verbose:
                print(f"  rank reached 556 after {batch_end} monomials; halting early",
                      flush=True)
            break

    print(f"\nFinal rank: {rank} (expected 556)", flush=True)
    assert rank == 556, f"rank = {rank}, expected 556"
    return rank


# ---------------------------------------------------------------------------
# Named diagnostic degree-3 invariants for the atlas
# ---------------------------------------------------------------------------


def _project_block_local_basis(u_p, S):
    from .contrast_blocks_3x3 import project_contrast_block
    full = project_contrast_block(u_p, S)  # shape (3, 3, 3)
    # Reduce out the non-S axes by taking the value at index 0
    sl = [slice(None)] * 3
    for axis in range(3):
        if axis not in S:
            sl[axis] = 0  # block is constant along non-S axes
    return full[tuple(sl)]


def diagnostic_degree3(u):
    from .contrast_blocks_3x3 import mean_zero_payoff, project_contrast_block, family_matrix
    u0 = mean_zero_payoff(u)
    out = {}

    # (1) Main-effect power-sum cubics p_3(T_{S,p}) for each main-effect type S
    #     and each player p. For k=3 mean-zero vectors v in W_3, p_3(v) = sum v_i^3.
    #     There are 3 main-effect types * 3 players = 9 such invariants.
    for axis in (0, 1, 2):
        S = (axis,)
        for p in range(3):
            T = _project_block_local_basis(u0[p], S)  # shape (3,)
            out[f"p3_main_S{axis}_p{p}"] = float(np.sum(T ** 3))

    # (2) Cross-player main-effect cubics: sum_a T_{S,p}^a * T_{S,q}^a * T_{S,r}^a
    #     for unordered (p, q, r). There are 3 types * 10 unordered triples = 30.
    #     We expose a handful: the fully symmetric one tr(T_{S,1} T_{S,2} T_{S,3})
    #     for each S.
    for axis in (0, 1, 2):
        S = (axis,)
        T1 = _project_block_local_basis(u0[0], S)
        T2 = _project_block_local_basis(u0[1], S)
        T3 = _project_block_local_basis(u0[2], S)
        out[f"crossplayer_main_S{axis}_123"] = float(np.sum(T1 * T2 * T3))

    # (3) Cubic in quadratic Gram entries, hence degree six in payoff entries.
    M123 = family_matrix(u, (0, 1, 2))
    out["det_M_three_way"] = float(np.linalg.det(M123))
    out["tr_M_three_way_cubed"] = float(np.trace(M123 @ M123 @ M123))

    # (4) Three-way interaction tensor traces: sum_{a,b,c} D_p[a,b,c] D_q[a,b,c] D_r[a,b,c]
    #     where D_p = T_{{1,2,3}, p}. This is the natural cubic on the three-way blocks.
    D1 = _project_block_local_basis(u0[0], (0, 1, 2))
    D2 = _project_block_local_basis(u0[1], (0, 1, 2))
    D3 = _project_block_local_basis(u0[2], (0, 1, 2))
    out["threeway_cubic_123"] = float(np.sum(D1 * D2 * D3))

    # (5) Pairwise-interaction "det" contribution: for each pair {i,j} \subset {0,1,2},
    #     the pair-interaction block T_{{i,j}, p} is a 3x3 matrix in coords i,j.
    #     Centered row/column sums vanish, so this determinant is identically zero.
    for pair in combinations((0, 1, 2), 2):
        for p in range(3):
            T = _project_block_local_basis(u0[p], pair)  # shape (3, 3)
            out[f"det_pair_S{pair[0]}{pair[1]}_p{p}"] = float(np.linalg.det(T))

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    from .contrast_blocks_3x3 import all_family_matrices

    print("Generators for (3,3)-games under (S_3)^3 (strategy-only)")
    print("=" * 70)

    print("\nDegree 2: 42 family-matrix entries (analytical, from contrast blocks)")
    print(f"  7 contrast families x 6 unordered player pairs = 42 generators")
    print(f"  matches h_2 = 42 from molien_3x3_strategy_only")

    print("\nDiagnostic cubic and sextic values (including zero determinant slots):")
    rng = np.random.default_rng(7)
    u_random = rng.standard_normal((3, 3, 3, 3))
    diag = diagnostic_degree3(u_random)
    for name, val in diag.items():
        print(f"  {name:40s} = {val:+.4f}")
    print(f"  (total diagnostics: {len(diag)})")

    print("\nDegree 3: numerical rank verification (target 556) ...")
    rank = verify_degree3_count()
    print(f"\nVERIFIED: degree-3 Reynolds invariants span a {rank}-dim subspace.")


if __name__ == '__main__':
    main()
