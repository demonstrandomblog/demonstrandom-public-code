# AI-assisted research code; see README.md and the repository AI_NOTICE.md.
import math
from functools import lru_cache

import numpy as np
import matplotlib.pyplot as plt


# -------- core combinatorics --------

@lru_cache(maxsize=None)
def hamming_ball_volume(k: int, r: int) -> int:
    """V(k, r) = sum_{i=0}^r C(k, i) in {0,1}^k."""
    if k < 0 or int(k) != k or int(r) != r:
        raise ValueError("Require integer k >= 0 and integer radius")
    k, r = int(k), int(r)
    r = min(r, k)
    return sum(math.comb(k, i) for i in range(r + 1))


def min_radius_for_covering(k: int, N: int) -> int:
    """
    Given semantic dimension k and number of stories N,
    find smallest r such that

        V(k, r) >= 2^k / N

    (back-of-the-envelope even-dispersion assumption).
    """
    if k < 0 or int(k) != k or int(N) != N:
        raise ValueError("k and N must be integers, with k nonnegative")
    if N <= 0:
        raise ValueError("N must be positive")

    k, N = int(k), int(N)
    target = 1 << k  # Compare integer counts without floating-point overflow.

    for r in range(k + 1):
        if hamming_ball_volume(k, r) * N >= target:
            return r
    return k


def fractional_overlap(k: int, N: int) -> float:
    """d_f = 1 - r/k."""
    if k <= 0:
        raise ValueError("Overlap requires k > 0")
    r = min_radius_for_covering(k, N)
    return 1.0 - r / k


# -------- heatmap: x = log10 N, y = k --------

def plot_overlap_heatmap(
    k_values=None,
    log10N_values=None,
    figsize=(8, 6)
):
    """
    Heatmap of d_f as a function of log10 N (x-axis) and k (y-axis).
    Uses semantic dimensions 5 through 60 in steps of 5.
    """
    if k_values is None:
        # Semantic dimensions used in the article's heatmap.
        k_values = np.arange(5, 61, 5)  # 5,10,...,60

    if log10N_values is None:
        # from 1 story up to 1e9 stories by default
        log10N_values = np.linspace(0, 9, 91)

    k_values = np.array(k_values, dtype=int)
    log10N_values = np.array(log10N_values, dtype=float)

    # rows: k index, cols: log10N index
    d_f_matrix = np.zeros((len(k_values), len(log10N_values)))

    for i, k in enumerate(k_values):
        for j, logN in enumerate(log10N_values):
            N = int(round(10 ** logN))
            d_f_matrix[i, j] = fractional_overlap(k, N)

    plt.figure(figsize=figsize)
    im = plt.imshow(
        d_f_matrix,
        origin="lower",
        aspect="auto",
        extent=[
            log10N_values[0], log10N_values[-1],
            k_values[0],      k_values[-1],
        ],
        vmin=0, vmax=1
    )
    cbar = plt.colorbar(im)
    cbar.set_label("Fractional overlap $d_f$")

    plt.xlabel(r"$\log_{10} N$")
    plt.ylabel("Semantic bits $k$")
    plt.title("Overlap fraction $d_f$ as a function of $\\log_{10} N$ and $k$")
    plt.tight_layout()
    return plt.gcf()


# -------- cross-sections: d_f vs log10 N for many k --------
def plot_overlap_vs_N_every_5_k(
    log10N_min=0,
    log10N_max=12,
    num_points=200,
    figsize=(10, 6)
):
    """
    Plot d_f vs log10(N) for k = 5, 10, ..., 100.
    Trim x-axis so it only shows the region where curves exist.
    """

    k_list = list(range(5, 101, 5))
    log10N_values = np.linspace(log10N_min, log10N_max, num_points)

    plt.figure(figsize=figsize)

    # Track rightmost valid x across all curves
    global_max_x = 0

    for k in k_list:
        xs = []
        ys = []
        for logN in log10N_values:
            N = int(round(10 ** logN))
            val = fractional_overlap(k, N)
            xs.append(logN)
            ys.append(val)

        # Find the last x where the curve actually varies (before the constant-1 plateau)
        # A point is valid as long as val < 1
        valid_indices = [i for i, v in enumerate(ys) if v < 1]
        if valid_indices:
            last_valid_index = valid_indices[-1]
            max_x_for_k = xs[last_valid_index]
            global_max_x = max(global_max_x, max_x_for_k)

        # Plot whole curve (all sampled values)
        plt.plot(xs, ys, label=f"k={k}")

    # Set xlim to: [0, global_max_x]
    plt.xlim(0, global_max_x)

    plt.xlabel(r"$\log_{10} N$")
    plt.ylabel("Fractional overlap $d_f$")
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.title(r"Overlap vs $N$ for $k = 5, 10, \dots, 100$ (trimmed)")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    return plt.gcf()


if __name__ == "__main__":
    plot_overlap_heatmap()
    plot_overlap_vs_N_every_5_k()
    plt.show()
