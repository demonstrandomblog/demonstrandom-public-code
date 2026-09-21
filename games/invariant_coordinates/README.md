# Invariant coordinates for normal-form games

Companion to [Invariant Coordinates for Normal-Form Games](https://demonstrandom.com/symmetry/posts/invariant_coords_normal_form_games_v2/).

An invariant coordinate is a polynomial in game payoffs whose value survives
strategy relabeling. These examples compute coordinates, exact graded dimension
counts, polynomial relations, contrast blocks, and finite-sample diagnostics.

> **AI warning:** Experimental research code developed with AI assistance.
> Full human and mathematical review is not established. Numerical ranks and
> successful sample checks do not prove a complete generating or separating set.

## Install and quick start

Use Python 3.11 or newer. From the repository root:

```sh
python -m pip install "./games/invariant_coordinates[test]"
python -m games.invariant_coordinates
python -m pytest -q games/invariant_coordinates
```

NumPy supplies numerical operations, SymPy symbolic relations, and SciPy the
numerical indifference solver. Tests use pytest. The module command is a small
example; it does not run the large Reynolds enumeration.

```python
import numpy as np
from games.invariant_coordinates.rg_from_invariants import mean_zero, eval_generators, rg_type
from games.invariant_coordinates.contrast_blocks_3x3 import all_family_matrices
from games.invariant_coordinates.candidate_atlases import load_atlas, game_to_point, evaluate

a, b = [3, 0, 5, 1], [3, 5, 0, 1]
coordinates = eval_generators(*mean_zero(*a, *b))
assert len(coordinates) == 17
assert rg_type(a, b) == rg_type(a[::-1], b[::-1])

game = np.random.default_rng(0).normal(size=(3, 3, 3, 3))
assert len(all_family_matrices(game)) == 7
values = evaluate(load_atlas("atlas598"), game_to_point(game)[None, :])
assert values.shape == (1, 598)
```

## Models and coordinate conventions

For two-player 2x2 games, each payoff vector lists cells in row-major order:
`a=(a00,a01,a10,a11)` and similarly `b`. The three contrasts are
`r=a00+a01-a10-a11`, `c=a00-a01+a10-a11`, and
`d=a00-a01-a10+a11`. They are unnormalized and already discard the payoff
mean. The six-variable order is `(rA,cA,dA,rB,cB,dB)`.

The strategy-only group independently exchanges rows and columns, keeping
players distinguished. Its 17 generators are returned in this order:

- Nine quadratics: `rA², rA*rB, cA², cA*cB, dA², dA*dB, rB², cB², dB²`.
- Eight cubics: `cA*dA*rA, cA*dB*rA, cB*dA*rA, cB*dB*rA,
  cA*dA*rB, cA*dB*rB, cB*dA*rB, cB*dB*rB`.

The player-exchange example uses the eight-element group that also transposes
the game and exchanges payoff players. It discovers 5 quadratic, 4 cubic,
8 quartic, and 5 quintic generators numerically. The nine low-degree expressions
used by `syzygies_exact` are a subset; that script does not calculate all
relations among all 22 generators.

Strict ordinal classification ranks each player's four payoffs separately.
There are 144 classes with players distinguished. `rg_type` returns `None`
when payoffs tie. The sign-feature and greedy-deletion experiments use the
144 representatives with payoffs 1,2,3,4; their result is sample-specific.

For three-player, three-strategy games, `u[p,s1,s2,s3]` has shape
`(3,3,3,3)`, with the payoff player first. Independent permutations of the
three strategy axes give 216 group elements. Players remain distinguished.
Subtracting each player's mean leaves 78 dimensions.

A contrast block is labeled by a nonempty subset `S` of strategy axes
`0,1,2`. Subtract means along axes in `S`, and average along its complement.
The seven blocks sum to the centered payoff tensor. For each `S`, a 3x3
family matrix contains Frobenius inner products between players' blocks.
Its six upper-triangle entries give 42 quadratic invariants across all families.

## Scripts

Run scripts as modules from the repository root:
`python -m games.invariant_coordinates.MODULE`. Some commands write outputs
to the current directory. To run elsewhere, add this repository to `PYTHONPATH`.

| Module | Computation |
|---|---|
| `generators_2x2_strategy_only` | Numerical Reynolds discovery of the 17 positive-degree generators; exact symbolic expressions. |
| `generators_2x2_mz` | Player-exchange discovery, retaining products of all lower-degree generators. |
| `syzygies_strategy_only`, `syzygies_exact` | Exact relations in tested degrees among 17 strategy-only generators or nine player-exchange coordinates. |
| `scaling`, `degree3_mz_strategy_only` | Exact character/combinatorial counts as the number of binary-strategy players grows. |
| `rg_from_invariants`, `rg_minimal` | Enumerate 144 strict ordinal representatives and greedily reduce their sign features. |
| `verify_ne_discriminant` | Square two-player indifference determinant product under relabeling and scaling. |
| `ne_disc_3_2_solve` | Quadratic elimination discriminant for three players with two strategies. |
| `ne_disc_degree` | Numerical scaling of square-game determinants and of Jacobians at indifference roots. |
| `verify_solvability` | Compare a strict-dominance contrast criterion with explicit best responses. |
| `molien_3x3_strategy_only`, `enumerate_3x3_generators` | Exact graded counts, including 42 quadratics and 556 cubics. |
| `contrast_blocks_3x3` | Contrast projectors, reconstruction, orthogonality, and family matrices. |
| `generators_3x3_strategy_only` | Numerical cubic Reynolds rank; approximately two minutes in the recorded CPU environment. |
| `atlas_3x3`, `classify_3x3` | Thirteen named games, their diagnostics, and finite-example classification. Run atlas before classify. Writes `atlas_3x3_results.json` and `atlas_3x3_table.md`. |
| `orbit_separation_3x3`, `hilbert_separation_3x3` | A 93-coordinate diagnostic fingerprint, sample collisions, and sampled ranks of products. |
| `typology_census_3x3` | Construct and sample contrast-family presence/sign patterns; writes JSON and Markdown tables. |
| `verify_atlases` | Reproduce the 36-game experiment with the three stored candidate atlases. |

The 93-coordinate fingerprint has 42 quadratic slots, 49 cubic slots, and
two sextic slots. Nine cubic slots are determinants of centered 3x3 blocks
and vanish identically. The sextics are `trace(M³)` and `det(M)` for a
quadratic Gram matrix. It is a separate diagnostic from the stored 598 atlas.
Numerical rank is capped by the number of sample points and depends on
tolerance. A rank deficit is not by itself an exact Hilbert-series calculation.

Indifference equations have algebraic roots outside the probability simplex.
The numerical solver searches from one initial point and does not enumerate
all roots. Its Jacobian determinant is not a polynomial discriminant.
Uniform payoff scaling gives degree `n` for that determinant with `n`
players, while the separate three-player elimination discriminant has degree six.

## Stored candidate atlases

These are candidate coordinate lists, **not certified global separating sets**.

| Data name | Count used in article | Nonzero mean-zero coordinates | Degrees |
|---|---:|---:|---|
| `atlas598` | 598 | 598 | 42 quadratics + 556 cubics |
| `polarized302` | 302 | 299 | 40 quadratics + 259 cubics |
| `polarized1456` | 1,456 | 1,453 | 35 quadratics + 203 cubics + 1,215 quartics |

The latter two full lists include three linear payoff-sum coordinates that
vanish after centering. Only their nonzero restrictions are stored here.
The original polarization coefficients were residues modulo 998244353.
They are lifted to centered signed integers, matching the numerical experiment's
convention. This lift and the sample checks do not establish a global theorem.

The [data manifest](data/manifest.json) records original and converted hashes,
counts, degrees, and coefficient conventions. Compressed NPZ files contain
numeric arrays only and load with `allow_pickle=False`. They total about
32 MB; the largest expands to about 302 MB of arrays.

A variable index is `26*p + 9*s1 + 3*s2 + s3`, excluding profile `(2,2,2)`
within each player's block. The omitted payoff is minus the other 26 entries'
sum. This is a coordinate restriction, not deletion from an unconstrained game.

Each file has four arrays:

- `indices[T,4]`: uint8 variable indices. Repetition denotes powers; 78 pads
  with a factor of one. All terms have degree at most four.
- `numerator[T]`, `denominator[T]`: int64 rational coefficients.
- `offsets[P+1]`: int64 bounds for each polynomial's contiguous terms.

`evaluate` uses bounded term batches instead of dense 78-column power arrays.
It computes floating values; cancellation and overflow remain possible at
large payoffs. The raw rational arrays are available for exact calculations.

Reproduce the recorded experiment:

```sh
python -m games.invariant_coordinates.verify_atlases all --output atlas-verification.json
```

Each atlas is tested on all 216 images of one seeded Gaussian game and every
one of the 630 unordered pairs in a fixed 36-game sample: 15 generic, seven
identical-payoff, seven zero-sum, and seven invariant under a common strategy
permutation on all axes. Fingerprints are normalized for pairwise comparison;
near collisions trigger explicit group enumeration. This finite test cannot
rule out collisions elsewhere. The [recorded results](verification.json) include
tolerances and measured errors.

## Provenance and limitations

The 21 script bodies come from the linked article's software appendix, with
local imports and documented corrections. [provenance.json](provenance.json)
identifies source blocks and prepared files.
[Article differences](../../ARTICLE_DIFFERENCES.md) records mathematical and
implementation discrepancies. The article's discussion of other algorithms
does not imply they are implemented by this directory.

Original code, documentation, and generated polynomial data use
[PolyForm Noncommercial 1.0.0](LICENSE).
Preserve [NOTICE](NOTICE); citation metadata is in
[CITATION.cff](../../CITATION.cff). Dependencies retain their own licenses.
