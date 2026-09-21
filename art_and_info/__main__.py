# AI-assisted experimental code; full human and mathematical review is not established.
"""Generate the five counting/overlap figures associated with Cultural Saturation."""
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from .hamming_radius_plot import plot_overlap_heatmap, plot_overlap_vs_N_every_5_k, min_radius_for_covering, fractional_overlap
from .weighted_hamming_radius_plot import plot_weighted_overlap_vs_N_every_5_k


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=Path("cultural_saturation_figures"))
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    figures=[("Overlap_Fraction.png",plot_overlap_heatmap()),("Overlap_vs_N.png",plot_overlap_vs_N_every_5_k())]
    for beta,suffix in [(.5,"0_5"),(1.,"1"),(2.,"2")]:
        figures.append((f"Overlap_vs_N_{suffix}.png",plot_weighted_overlap_vs_N_every_5_k(beta=beta,scale=500)))
    for name,fig in figures:
        fig.savefig(args.output/name,dpi=160);plt.close(fig)
    print(f"k=50, N=5000000: radius={min_radius_for_covering(50,5000000)}, overlap={fractional_overlap(50,5000000):.2f}")
    print(f"Saved {len(figures)} figures in {args.output}")


if __name__ == "__main__":
    main()
