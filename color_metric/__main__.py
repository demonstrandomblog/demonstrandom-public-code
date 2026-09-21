# AI-assisted experimental code; full human and mathematical review is not established.
"""Run both interpolation modes and save a spectrum comparison and JSON results."""
import argparse,json
from pathlib import Path
import matplotlib.pyplot as plt
from .metric import evaluate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=Path("color_metric_results"))
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    results=[evaluate(mode) for mode in ("article-component","log-euclidean")]
    fig,ax=plt.subplots(figsize=(8,5))
    for result in results:
        s=result['singular_values']
        ax.semilogy(range(len(s)),[v/s[0] for v in s],"o-",label=result['mode'])
        print(f"{result['mode']}: {result['shape']}, rank {result['rank']}, nonpositive metric points {result['nonpositive_metric_points']}")
    ax.set(xlabel="Singular value index",ylabel="Singular value / largest",title="Polynomial Killing-equation search (degree 3)")
    ax.legend();fig.tight_layout();fig.savefig(args.output/'singular_values.png',dpi=160);plt.close(fig)
    (args.output/'results.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=="__main__":
    main()
