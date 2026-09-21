# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Experimental AI-assisted research code; human review is incomplete.
"""Greedy feature deletion on 144 strict ordinal 2x2 representatives.
Features are signs of 17 polynomial generators and nine squared-magnitude
comparisons, evaluated on payoffs 1,2,3,4. The returned subset is sample-specific
and is not a minimum-cardinality or global orbit-separation certificate."""
import numpy as np
from itertools import permutations


from .rg_from_invariants import mean_zero, eval_generators, ordinal_type, rg_type, sign, enumerate_rg_types


def test_features(rg_map, indices, all_features):
    pat_to_rg = {}
    for rgt, feats in all_features.items():
        pat = tuple(feats[i] for i in indices)
        if pat in pat_to_rg and pat_to_rg[pat] != rgt:
            return False
        pat_to_rg[pat] = rgt
    return len(pat_to_rg) == len(rg_map)


if __name__ == '__main__':
    rg_map = enumerate_rg_types()
    FEAT_NAMES = [
        'rA2', 'rArB', 'cA2', 'cAcB', 'dA2', 'dAdB', 'rB2', 'cB2', 'dB2',
        'cAdArA', 'cAdBrA', 'cBdArA', 'cBdBrA', 'cAdArB', 'cAdBrB', 'cBdArB', 'cBdBrB',
        'rA2-cA2', 'rA2-dA2', 'cA2-dA2',
        'rB2-cB2', 'rB2-dB2', 'cB2-dB2',
        'rA2-rB2', 'cA2-cB2', 'dA2-dB2',
    ]

    all_features = {}
    for rgt, (a, b) in rg_map.items():
        gens = eval_generators(*mean_zero(*a, *b))
        feats = [sign(g) for g in gens]
        feats += [sign(gens[0] - gens[2]), sign(gens[0] - gens[4]), sign(gens[2] - gens[4]),
                  sign(gens[6] - gens[7]), sign(gens[6] - gens[8]), sign(gens[7] - gens[8]),
                  sign(gens[0] - gens[6]), sign(gens[2] - gens[7]), sign(gens[4] - gens[8])]
        all_features[rgt] = feats

    # Greedy backward elimination
    current = set(range(len(FEAT_NAMES)))
    for i in reversed(range(len(FEAT_NAMES))):
        trial = sorted(current - {i})
        if test_features(rg_map, trial, all_features):
            current = set(trial)
    print(f"Greedy feature subset for 144 representatives: {len(current)} features")
    for i in sorted(current):
        print(f"  {i}: {FEAT_NAMES[i]}")
