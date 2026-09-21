# Game structure in invariant coordinates

Companion routines for the game-class, potential, best-response, Hodge, and
cycle sections of [Invariant Coordinates for Normal-Form Games](https://demonstrandom.com/symmetry/posts/invariant_coords_normal_form_games_v2/).

> **AI warning:** Experimental code developed with AI assistance. The tests
> below do not establish a full human or mathematical review.

## Run and inputs

```sh
python -m pip install "./games/invariant_coordinates[test]"
python -m games.invariant_coordinates.structure_example
python -m pytest -q games/invariant_coordinates/test_game_structure.py
```

These modules use square two-player k-by-k games with k>=2. A is the row
player's payoff matrix; B is the column player's. Flattened payoffs concatenate
A.ravel() and B.ravel(). These functions do not silently center payoffs.
They are separate from the three-player 3x3x3 atlas API.

| Module | Operations |
|---|---|
| game_classes | Literal class predicates, strategy/player permutations, exact monomial orbit averages |
| potential_subvariety | Additive rectangle constraints, their invariant squared norm, potential reconstruction |
| hodge | Difference/curl operators and orthogonal potential, harmonic, nonstrategic projections |
| hodge_invariants | Orthonormal component bases and invariant squared component norms |
| br_type_invariants | Canonical unique-best-response maps under strategy relabeling and player exchange |
| cycles | All-maximizer response graph, unique alternating dynamics, directed cycles and labeled cycle witnesses |
| selection | Five quadratic/four cubic D4 coordinates and explicit finite-pair comparisons |

```python
import numpy as np
from games.invariant_coordinates.hodge import decompose
from games.invariant_coordinates.hodge_invariants import component_energies
from games.invariant_coordinates.potential_subvariety import recover_potential
from games.invariant_coordinates.br_type_invariants import br_type

A = np.array([[1., -1.], [-1., 1.]])
B = -A
potential, harmonic, nonstrategic = decompose(A, B)
assert np.allclose(harmonic, [A, B])
assert component_energies(A, B)["harmonic"] > 0
assert br_type(A, B) == br_type(B.T, A.T)

C = np.array([[3., 0.], [0., 2.]])
phi, f, g = recover_potential(C, C)
assert np.allclose(phi + f[None, :], C)
assert np.allclose(phi + g[:, None], C)
```

## Mathematical conventions

A potential game admits A[i,j]=Phi[i,j]+f[j] and B[i,j]=Phi[i,j]+g[i].
Thus A-B has zero additive rectangle differences. This is not a rank-one
condition or vanishing determinant condition. potential_obstruction is the
sum of squares of all rectangle differences, a degree-two relabeling invariant
whose exact zero locus over the reals is the potential-game subspace.
Floating-point predicates apply an explicit absolute tolerance.

The Hodge decomposition uses the Euclidean payoff inner product for equal
strategy counts. Nonstrategic means A may depend on the column only, while B
may depend on the row only. Its dimension is 2k, not two. The strategic potential
and harmonic dimensions are k^2-1 and (k-1)^2. The SVD rank threshold is 1e-10
on integer difference operators. component_energies returns squared norms;
these sum to the original squared norm and are invariant under relabeling.

A best-response map sends every opposing action to its unique maximizing
action. It need not be a permutation. Player exchange swaps the two maps
without inverting them. Exact ties raise for canonical types and alternating
dynamics; best_response_graph includes all tied maximizers. There is no
near-tie tolerance. Results can change discontinuously as payoffs cross a tie.
An alternating step contains two successive unilateral moves. Cycle detection
excludes incoming transient paths. A labeled cycle_witness multiplies gains
around one specified unilateral cycle and is not itself a relabeling invariant.

Monomial orbit averages divide the distinct-monomial sum by orbit size.
For integer/Fraction inputs this is an exact rational Reynolds average.
Group enumeration costs 2(k!)^2; use small k and small degree. class_diagnostics
reports overlapping predicates: symmetric_in_given_labels tests B=A.T in the
supplied labels, and does not search for an unknown relabeling.

The selection coordinates use unnormalized 2x2 contrasts
r=a00+a01-a10-a11, c=a00-a01+a10-a11, d=a00-a01-a10+a11.
Its nine coordinates are not a complete generating or separating set.
The example supplies a potential-game pair with equal quadratics but distinct
cubics, plus a distinct-orbit pair with all nine coordinates equal.
No low-degree classification or universal cycle-degree theorem is asserted.

## Source correspondence and validation

The seven modules retain the article-referenced filenames, with selected
routines corrected or rewritten into this shared API. Run structure_example
for their combined deterministic demonstration. The original long randomized
exploratory drivers and hard-coded summary claims are not part of this API.
The existing 21 software-appendix examples and stored atlases remain available.

Seventeen tests cover independently constructed Hodge components for k=2,3,4;
all 72 strategy/player relabelings for k=3; exact full-group Reynolds averages;
potential reconstruction; non-bijective best responses; ties and transient
cycles; and explicit low-degree collisions. See [article differences](../../ARTICLE_DIFFERENCES.md)
and [provenance](provenance.json) for source identities and corrections.

Original code uses [PolyForm Noncommercial 1.0.0](LICENSE).
Preserve [NOTICE](NOTICE); dependencies retain their own licenses.
