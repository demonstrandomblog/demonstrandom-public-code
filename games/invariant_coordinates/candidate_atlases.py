# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Experimental AI-assisted research code; human review is incomplete.
"""Stored candidate atlases for three players with three strategies each.

Games have shape (3,3,3,3), payoff player first. Subtract each player's mean,
then flatten that player's first 26 strategy profiles in C order; the omitted
(2,2,2) payoff equals minus their sum. Concatenating players gives 78 variables.
Four variable indices encode each rational monomial; index 78 is a factor of
one. Loading disables pickle. Numerical samples do not prove global separation.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

DATA = Path(__file__).with_name('data')


def manifest():
    """Return identities, counts, coefficient conventions, and data hashes."""
    return json.loads((DATA / 'manifest.json').read_text(encoding='utf-8'))


def game_to_point(game):
    """Remove payoff means and return 78 independent entries, player-major."""
    game = np.asarray(game, dtype=float)
    if game.shape != (3, 3, 3, 3) or not np.isfinite(game).all():
        raise ValueError('Expected finite payoffs of shape (3,3,3,3)')
    flat = game.reshape(3, 27)
    return (flat - flat.mean(axis=1, keepdims=True))[:, :26].reshape(78)


def load_atlas(name):
    """Read a bundled atlas, checking its hash and sparse-array structure."""
    records = {item['name']: item for item in manifest()['atlases']}
    if name not in records:
        raise ValueError(f'Unknown atlas {name!r}; choose from {tuple(records)}')
    record = records[name]
    path = DATA / record['file']
    with path.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != record['sha256']:
            raise ValueError(f'Data hash mismatch: {name}')
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in ('indices', 'numerator', 'denominator', 'offsets')}
    count, terms = record['mean_zero_count'], record['terms']
    idx, num, den, offsets = (arrays[k] for k in ('indices', 'numerator', 'denominator', 'offsets'))
    if (idx.shape != (terms, 4) or num.shape != (terms,) or den.shape != (terms,)
            or offsets.shape != (count+1,) or offsets[0] != 0 or offsets[-1] != terms
            or np.any(np.diff(offsets) <= 0) or np.any(idx > 78) or np.any(den <= 0)
            or any(a.dtype.kind not in 'iu' for a in arrays.values())):
        raise ValueError(f'Invalid sparse polynomial arrays: {name}')
    return arrays


def evaluate(atlas, points, term_chunk=4096):
    """Return (n_points,n_polys) values for finite (n_points,78) input.

    Temporary products contain at most n_points*term_chunk float64 entries.
    Direct multiplication preserves zeros and negative factors.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 78 or not np.isfinite(points).all():
        raise ValueError('Expected finite points of shape (n_points,78)')
    if not isinstance(term_chunk, int) or term_chunk < 1:
        raise ValueError('term_chunk must be a positive integer')
    padded = np.column_stack((points, np.ones(len(points))))
    offsets = atlas['offsets']
    values = np.zeros((len(points), len(offsets)-1))
    for i, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
        for a in range(int(start), int(stop), term_chunk):
            b = min(a+term_chunk, int(stop))
            idx = atlas['indices'][a:b]
            terms = padded[:, idx[:, 0]].copy()
            for factor in range(1, 4):
                terms *= padded[:, idx[:, factor]]
            coeff = atlas['numerator'][a:b] / atlas['denominator'][a:b]
            values[:, i] += terms @ coeff
    return values
