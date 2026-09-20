# AI-assisted research code; see README.md and the repository AI_NOTICE.md.
import math
from functools import lru_cache

import numpy as np
import matplotlib.pyplot as plt


# -------- base helpers for weighted metric --------

def semantic_diameter(k: int, beta: float) -> float:
    """
    S_k(beta) = sum_{i=0}^{k-1} (i+1)^(-beta)
    Maximum possible weighted distance between two k-bit strings.
    """
    return sum((i + 1) ** (-beta) for i in range(k))


def integer_weights(k: int, beta: float, scale: int) -> list[int]:
    """
    Discretize weights w_i = (i+1)^(-beta) to positive integers.

    We use:
        w_i_int = max(1, round(scale * (i+1)^(-beta)))

    so that:
        sum(w_i_int) / scale ~ S_k(beta)
    """
    if k < 0 or int(k) != k or scale <= 0 or int(scale) != scale or not math.isfinite(beta) or beta < 0:
        raise ValueError("Require integer k >= 0, integer scale > 0, and finite beta >= 0")
    weights = []
    for i in range(int(k)):
        w_real = (i + 1) ** (-beta)
        w_int = int(round(scale * w_real))
        if w_int <= 0:
            w_int = 1
        weights.append(w_int)
    return weights


# -------- combinatorics for weighted balls (no normal approx) --------

@lru_cache(maxsize=None)
def weighted_cumulative_counts(k: int, beta: float, scale: int) -> list[int]:
    """
    DP over subsets to count, for each integer weighted radius r_int, how many
    bitstrings have weighted distance <= r_int from 0...0, using discretized weights.

    Returns a list `cum` of length W_tot+1, where:
        cum[r_int] = number of subsets with total integer weight <= r_int.

    This is exact *for the discretized weights*.
    """
    weights = integer_weights(k, beta, scale)
    if k == 0:
        return [1]  # only the empty string
    W_tot = sum(weights)

    # dp[s] = number of subsets with exact integer weight s
    dp = [0] * (W_tot + 1)
    dp[0] = 1

    for w in weights:
        # typical knapsack-style DP: iterate backwards to avoid double-counting
        for s in range(W_tot, w - 1, -1):
            dp[s] += dp[s - w]

    # cumulative counts: cum[r] = sum_{s<=r} dp[s]
    cum = [0] * (W_tot + 1)
    running = 0
    for s in range(W_tot + 1):
        running += dp[s]
        cum[s] = running

    return cum


def weighted_ball_volume(k: int, beta: float, scale: int, r_int: int) -> int:
    """
    V_beta(k, r_int) = number of bitstrings within integer weighted radius r_int.
    """
    cum = weighted_cumulative_counts(k, beta, scale)
    if r_int < 0:
        return 0
    r_int = min(r_int, len(cum) - 1)
    return cum[r_int]


def min_weighted_radius_for_covering(k: int, N: int, beta: float, scale: int) -> int:
    """
    Given semantic dimension k, number of stories N, and weighting beta,
    find the smallest integer radius r_int such that

        V_beta(k, r_int) >= 2^k / N

    where V_beta is the weighted ball volume using discretized weights.
    """
    if k < 0 or int(k) != k or N <= 0 or int(N) != N:
        raise ValueError("Require integer k >= 0 and integer N > 0")
    if k == 0:
        return 0

    k, N = int(k), int(N)
    target = 1 << k  # Compare integer counts without floating-point overflow.
    cum = weighted_cumulative_counts(k, beta, scale)

    for r_int, vol in enumerate(cum):
        if vol * N >= target:
            return r_int
    return len(cum) - 1


def fractional_overlap_weighted(k: int, N: int, beta: float, scale: int) -> float:
    """
    Weighted fractional overlap:

        d_f^{(w)} = 1 - R / S_k(beta),

    where R is the (real) weighted covering radius corresponding to
    the integer radius r_int returned by min_weighted_radius_for_covering.
    """
    if k < 0 or int(k) != k or N <= 0 or int(N) != N:
        raise ValueError("Require integer k >= 0 and integer N > 0")
    if k == 0:
        return 0.0

    # real diameter
    S_k = semantic_diameter(k, beta)
    if S_k <= 0:
        return 0.0

    r_int = min_weighted_radius_for_covering(k, N, beta, scale)
    R_real = r_int / float(scale)

    d_f_w = 1.0 - R_real / S_k
    # numeric safety
    if d_f_w < 0.0:
        d_f_w = 0.0
    if d_f_w > 1.0:
        d_f_w = 1.0
    return d_f_w


# -------- weighted heatmap: x = log10 N, y = k --------

def plot_weighted_overlap_heatmap(
    beta: float = 1.0,
    scale: int = 200,
    k_values=None,
    log10N_values=None,
    figsize=(8, 6),
):
    """
    Heatmap of d_f^{(w)} as a function of log10 N (x-axis) and k (y-axis),
    for a given beta and discretization scale.

    scale controls the resolution of the discrete approximation:
    larger scale -> more accurate, but slower DP.
    """
    if k_values is None:
        k_values = np.arange(5, 41, 5)  # keep k modest; DP cost grows with k and scale

    if log10N_values is None:
        log10N_values = np.linspace(0, 9, 91)  # 1 to 1e9

    k_values = np.array(k_values, dtype=int)
    log10N_values = np.array(log10N_values, dtype=float)

    d_f_matrix = np.zeros((len(k_values), len(log10N_values)))

    for i, k in enumerate(k_values):
        for j, logN in enumerate(log10N_values):
            N = int(round(10 ** logN))
            d_f_matrix[i, j] = fractional_overlap_weighted(k, N, beta, scale)

    plt.figure(figsize=figsize)
    im = plt.imshow(
        d_f_matrix,
        origin="lower",
        aspect="auto",
        extent=[
            log10N_values[0], log10N_values[-1],
            k_values[0],      k_values[-1],
        ],
        vmin=0, vmax=1,
    )
    cbar = plt.colorbar(im)
    cbar.set_label(r"Weighted fractional overlap $d_f^{(w)}$")

    plt.xlabel(r"$\log_{10} N$")
    plt.ylabel("Semantic bits $k$")
    plt.title(
        rf"Weighted overlap $d_f^{{(w)}}$ vs $\log_{{10}} N$ and $k$ (beta={beta}, scale={scale})"
    )
    plt.tight_layout()
    return plt.gcf()


# -------- weighted cross-sections: d_f^{(w)} vs log10 N for many k --------

def plot_weighted_overlap_vs_N_every_5_k(
    beta: float = 1.0,
    scale: int = 200,
    log10N_min=0,
    log10N_max=12,
    num_points=200,
    figsize=(10, 6),
):
    """
    Plot d_f^{(w)} vs log10(N) for k = 5, 10, ..., 40 at a fixed beta.
    We keep k <= 40 by default because the DP cost grows with k and 'scale'.
    """
    k_list = list(range(5, 41, 5))
    log10N_values = np.linspace(log10N_min, log10N_max, num_points)

    plt.figure(figsize=figsize)

    global_max_x = 0.0

    for k in k_list:
        xs = []
        ys = []
        for logN in log10N_values:
            N = int(round(10 ** logN))
            val = fractional_overlap_weighted(k, N, beta, scale)
            xs.append(logN)
            ys.append(val)

        valid_indices = [idx for idx, v in enumerate(ys) if v < 1.0]
        if valid_indices:
            last_valid_index = valid_indices[-1]
            max_x_for_k = xs[last_valid_index]
            global_max_x = max(global_max_x, max_x_for_k)

        plt.plot(xs, ys, label=f"k={k}")

    if global_max_x > 0:
        plt.xlim(0, global_max_x)
    else:
        plt.xlim(log10N_min, log10N_max)

    plt.xlabel(r"$\log_{10} N$")
    plt.ylabel(r"Weighted fractional overlap $d_f^{(w)}$")
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.title(
        rf"Weighted overlap vs $N$ for $k = 5, 10, \dots, 40$ (beta={beta}, scale={scale})"
    )
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    return plt.gcf()


if __name__ == "__main__":
    plot_weighted_overlap_vs_N_every_5_k(beta=2, scale=500)
    plt.show()
