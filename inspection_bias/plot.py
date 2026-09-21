# AI-assisted experimental code; full human and mathematical review is not established.
"""Plot observed functional information and power-law model predictions."""
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from .fit import fit_all
from .persistence import STAGE_NAMES


def make_figure():
    result = fit_all()["power_law"]
    observed, predicted = zip(*result["predictions"])
    stages = list(range(1, len(observed)+1))
    fig, ax = plt.subplots(figsize=(8,5))
    ax.scatter(stages, observed, s=60, zorder=3, label="Observed (Hazen-Wong)", color="black")
    ax.plot(stages, predicted, "o--", color="steelblue", markersize=5, label="Predicted (power law)")
    for stage, name, value in zip(stages, STAGE_NAMES, observed):
        ax.annotate(name.split(":")[0].strip(), (stage,value), textcoords="offset points", xytext=(8,5), fontsize=7, color="gray")
    ax.set(xlabel="Geological Stage", ylabel="Functional Information (bits)", ylim=(0,160), xlim=(.5,9.9))
    ax.legend(frameon=False);fig.tight_layout()
    return fig


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("functional_information.png"))
    args=parser.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    fig=make_figure();fig.savefig(args.output,dpi=160);plt.close(fig)
    print(args.output)
