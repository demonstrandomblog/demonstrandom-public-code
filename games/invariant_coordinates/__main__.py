"""Print exact small-game counts and bounded game-coordinate examples."""
from .rg_from_invariants import enumerate_rg_types, mean_zero, eval_generators
from .molien_3x3_strategy_only import molien_33_strategy_only
from .candidate_atlases import manifest


def main():
    print('Strict ordinal 2x2 types with players fixed:',len(enumerate_rg_types()))
    print('3-player, 3-strategy Molien degrees 0..4:',molien_33_strategy_only(4))
    print('17 coordinates for A=[[3,0],[5,1]], B=[[3,5],[0,1]]:')
    print(eval_generators(*mean_zero(3,0,5,1,3,5,0,1)))
    for atlas in manifest()['atlases']:
        print(atlas['name'],':',atlas['mean_zero_count'],'nonzero mean-zero coordinates')
    print('Stored atlases are candidates; finite-sample checks do not prove global separation.')


if __name__=='__main__':
    main()
