# Copyright (c) 2024-2026 Kevin T. Procopio and contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# AI assistance/review status: see AI_NOTICE.md at the repository root.

"""Reproduce the three article scenarios and normal-form payoff matrix."""
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from .core import Arena, NormalFormConverter, build_stag_hunt, plot_trajectory, plot_payoff_heatmap

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('outputs/differential_games'))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    game,space = build_stag_hunt()
    arena = Arena(game,dt=.02,max_time=15)
    for label,c1,c2 in [('cooperate','ChaseStag','ChaseStag'),('defect','ChaseHare','ChaseHare'),('asymmetric','ChaseStag','ChaseHare')]:
        profile = dict(c1=c1,c2=c2,stag='Flee',hare1='Flee',hare2='Flee')
        trajectory,payoffs = arena.play(profile)
        print(f"{label}: c1={payoffs['c1']}, c2={payoffs['c2']}")
        plot_trajectory(trajectory,space,title=label.capitalize())
        plt.gcf().savefig(args.output_dir/f'{label}.png',dpi=150,bbox_inches='tight')
        plt.close('all')
    matrix = NormalFormConverter.to_payoff_matrix(arena,['c1','c2'])
    print(matrix)
    plot_payoff_heatmap(matrix,['c1','c2'],game.get_strategy_sets())
    plt.gcf().savefig(args.output_dir/'payoffs.png',dpi=150,bbox_inches='tight')
    plt.close('all')

if __name__ == '__main__':
    main()
