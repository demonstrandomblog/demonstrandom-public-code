# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Experimental AI-assisted research code; human review is incomplete.
"""Finite candidate-atlas experiment on 36 three-player games.

Payoffs have shape (3,3,3,3), player first. The group has 216 independent
strategy permutations, keeping players fixed. One seeded Gaussian game's orbit
tests invariance. The collision sample has 15 generic games, seven identical-
payoff games, seven zero-sum games, and seven games invariant under a shared
strategy permutation on all axes. All 630 unordered pairs are checked, with
group enumeration for close fingerprints. This does not prove global separation.
"""
import argparse
import json
import random
from itertools import permutations as iperms, product as iprod
from pathlib import Path
import numpy as np
from .candidate_atlases import evaluate, load_atlas, game_to_point
N, K = 3, 3
def get_cells():
    return [c for c in iprod(range(N), repeat=K) if c != (N - 1,) * K]


def apply_group(A_list, sigmas):
    new_list = []
    for A in A_list:
        new_A = np.zeros_like(A)
        for x in iprod(range(N), repeat=K):
            new_x = tuple(sigmas[a][x[a]] for a in range(K))
            new_A[new_x] = A[x]
        new_list.append(new_A)
    return new_list


def random_game(seed):
    random.seed(seed)
    A_list = []
    for _ in range(K):
        flat = np.array([random.gauss(0, 1) for _ in range(N ** K)])
        flat = flat - flat.mean()
        A_list.append(flat.reshape((N,) * K))
    return A_list


def player_sym(seed):
    random.seed(seed)
    flat = np.array([random.gauss(0, 1) for _ in range(N ** K)])
    flat = flat - flat.mean()
    A = flat.reshape((N,) * K)
    return [A.copy() for _ in range(K)]


def zero_sum(seed):
    random.seed(seed)
    A_list = []
    for _ in range(K - 1):
        flat = np.array([random.gauss(0, 1) for _ in range(N ** K)])
        flat = flat - flat.mean()
        A_list.append(flat.reshape((N,) * K))
    A_list.append(-sum(A_list))
    return A_list


def diag_sym(seed):
    random.seed(seed)
    A_init = []
    for _ in range(K):
        flat = np.array([random.gauss(0, 1) for _ in range(N ** K)])
        flat = flat - flat.mean()
        A_init.append(flat.reshape((N,) * K))
    A_sym = [np.zeros_like(A) for A in A_init]
    for sigma in iperms(range(N)):
        ds = tuple(sigma for _ in range(K))
        g_A = apply_group(A_init, ds)
        for p in range(K):
            A_sym[p] += g_A[p]
    A_sym = [(A / 6) for A in A_sym]
    A_sym = [A - A.mean() for A in A_sym]
    return A_sym


def games_in_same_orbit(A_v, A_w, tol=1e-7):
    for sigmas in iprod(iperms(range(N)), repeat=K):
        gw = apply_group(A_w, sigmas)
        if all(np.allclose(A_v[i], gw[i], atol=tol, rtol=0) for i in range(K)):
            return True
    return False

def sample_games():
    """Return the original 36 seeded games, preserving sample order."""
    return ([random_game(80000+s) for s in range(15)]
            + [player_sym(82000+s) for s in range(7)]
            + [zero_sum(83000+s) for s in range(7)]
            + [diag_sym(84000+s) for s in range(7)])


def verify(name):
    """Check every group image and every unordered pair in the finite sample."""
    base = random_game(70011)
    orbit = [apply_group(base, sigmas) for sigmas in iprod(iperms(range(N)), repeat=K)]
    games = sample_games()
    points = np.stack([game_to_point(g) for g in orbit+games])
    values = evaluate(load_atlas(name), points)
    inv_error = float(np.max(np.abs(values[:216]-values[0])))
    inv_relative = inv_error / max(float(np.linalg.norm(values[0])), 1.0)
    fingerprints = values[216:]
    norms = np.linalg.norm(fingerprints, axis=1, keepdims=True)
    fingerprints = fingerprints / np.maximum(norms, 1e-12)
    distances, false_merges, same_orbit_collisions = [], [], []
    for i in range(len(games)):
        for j in range(i+1, len(games)):
            distance = float(np.linalg.norm(fingerprints[i]-fingerprints[j]))
            distances.append(distance)
            if distance < 1e-6:
                target = same_orbit_collisions if games_in_same_orbit(games[i], games[j]) else false_merges
                target.append([i,j])
    return {
        'atlas': name, 'coordinates': int(values.shape[1]),
        'group_images': len(orbit), 'games': len(games), 'pairs': len(distances),
        'invariance_max_absolute_error': inv_error,
        'invariance_relative_error': inv_relative,
        'minimum_normalized_distance': min(distances),
        'median_normalized_distance': float(np.median(distances)),
        'false_merges': false_merges, 'same_orbit_collisions': same_orbit_collisions,
        'passed': bool(inv_relative < 1e-8 and not false_merges),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('atlas', choices=['atlas598','polarized302','polarized1456','all'])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    names = ['atlas598','polarized302','polarized1456'] if args.atlas == 'all' else [args.atlas]
    results = [verify(name) for name in names]
    report = {'scope':'Finite 36-game sample; no global separation claim.', 'results':results}
    text = json.dumps(report, indent=2)+'\n'
    if args.output:
        args.output.write_text(text, encoding='utf-8')
    else:
        print(text, end='')
    if not all(item['passed'] for item in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
