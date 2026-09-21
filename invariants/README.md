# Computational invariant theory

Companion to [Building a Minimal Computational Invariant Theory Library](https://demonstrandom.com/symmetry/posts/computational_invariant_theory/).

An invariant polynomial has the same value before and after a group action.
This teaching library computes such polynomials, their relations, and bounded
orbit diagnostics. A polynomial is a dictionary mapping exponent tuples to
`fractions.Fraction` coefficients: `{(2,0): Fraction(1)}` represents `x0**2`.
The zero polynomial is `{}`. Exponent tuples in one calculation must all have
the same length, and degrees are nonnegative integers.

> **AI warning:** Experimental research code developed with AI assistance.
> Full human and mathematical review is not established. The tests verify
> specific identities and examples, not every group or input size.

## Install and run

Use Python 3.11 or newer. From the repository root:

```sh
python -m pip install "./invariants[test]"
python -m pytest -q invariants
python -m invariants
```

Runtime operations use NumPy and Python's standard library. SymPy is a test
dependency for independent integer-rank and lattice-saturation checks.
The demo prints the symmetric-group Hilbert coefficients, three generators,
the sign-reversal example, and an exact integer kernel.

```python
from invariants.action import invariant_theory, compute_generators, compute_hilbert_series
from invariants.groups.constructors import symmetric, diagonal_torus
from invariants.spaces import polynomial_ring
from invariants.orbits import same_image

action = invariant_theory(symmetric(3), polynomial_ring(3))
generators = compute_generators(action, max_degree=3)
assert compute_hilbert_series(action, 6) == [1, 1, 2, 3, 4, 5, 7]
assert same_image(generators, (1, 2, 3), (3, 1, 2))
assert not same_image(generators, (1, 2, 3), (1, 1, 4))

torus = diagonal_torus([[-2, 1]])
basis = torus.kernel_basis()
assert (torus.W @ basis == 0).all()
```

## Supported computations

| Module | Meaning and contract |
|---|---|
| `poly` | Sparse rational arithmetic, monomial orderings, evaluation, and display with `show`. |
| `groebner` | Buchberger bases, normal forms, ideal intersection, elimination, and polynomial relations. `compute_relations` returns exponent tuples in the generator variables only. |
| `groups.finite` | Reynolds and orbit sums for explicit signed-permutation groups. Supply every group element. Numerical Molien coefficients and approximate point-orbit tests are also available. |
| `groups.torus` | An integer matrix `W` defines diagonal weights; `x**alpha` is invariant exactly when `W@alpha=0`. `kernel_basis` spans all signed integer solutions; nonnegative monomial generators require a separate degree-bounded enumeration. |
| `classical` | Inner products for orthogonal groups, determinants for special linear groups, and symplectic pairings. Coordinates concatenate the vectors. Products are reduced to an independent graded basis before counting dimensions. |
| `action` | Combines a group and polynomial space, enumerates generators and separating candidates by degree, and searches for parameters and secondary candidates. |
| `orbits` | Evaluates supplied invariant lists and checks explicit pairs. Global conclusions require a complete generating or separating set. |

`compute_syzygies` returns valid pairwise Koszul relations `f_j*e_i-f_i*e_j`.
The final coordinates represent formal `e` variables. It does **not** compute
a complete syzygy module. `compute_relations` solves the different problem of
finding polynomial equations among invariant values by elimination.

## Degree bounds and separation

`compute_generators` and `compute_separating_invariants` only inspect degrees
through `max_degree`. For finite linear groups in characteristic zero, the
group order is a sufficient bound, but enumeration at that bound can be costly.
The separator routine retains every basis invariant within the cutoff and
does not promise minimum cardinality. `minimal_separating_subset` greedily
covers supplied test pairs and raises if any pair cannot be distinguished.
Passing a finite list of pairs does not certify separation of all orbits.

A homogeneous system of parameters (HSOP) does not necessarily separate orbits.
For simultaneous sign reversal, `x**2` and `y**2` are parameters but identify
`(1,1)` with `(1,-1)`; `x*y` distinguishes them. `find_hsop` checks algebraic
independence and a zero-dimensional common zero locus, and raises if its
greedy search fails. This criterion is for finite-group actions on the full
coordinate space. Secondary candidates and null-cone tests remain degree bounded.

The torus kernel uses arbitrary-size Python integers and returns an object-dtype
array with basis vectors as columns. Integer-preserving basis transformations
ensure that primitive solutions are retained. Multiplying independently scaled
rational null vectors can miss such solutions. `hilbert_basis(max_degree)`
enumerates indecomposable nonnegative solutions only through its stated cutoff.

## Numerical and practical limits

Polynomial substitution explicitly rejects matrices outside the signed-permutation
representation. Rotation constructors with irrational entries can still be used
for numerical Molien or point-orbit diagnostics, but cannot certify exact rational
polynomials. Molien computations use floating eigenvalues and rounding; point
comparisons use a tolerance. Group closure is the caller's responsibility.

Buchberger elimination and monomial enumeration can grow quickly. Start with
small dimensions and low degrees. The worked examples cover symmetric groups,
simultaneous sign reversal, diagonal tori, and small classical-group actions.
Equivalent generating sets may differ in order, normalization, and basis choice.
The article's displayed `format_poly` name is supplied here as `poly.show`.

## Checks and provenance

Tests check the published examples, relation substitution, classical graded
dimensions, degree cutoffs, and separator failures. Kernel checks cover 125
small signed weight rows and 120 random rectangular matrices, with independent
exact rank and maximal-minor gcd tests establishing lattice completeness.
[provenance.json](provenance.json) records source identities and file hashes.

Original code and documentation: [PolyForm Noncommercial 1.0.0](LICENSE).
Preserve [NOTICE](NOTICE). Citation metadata is in [CITATION.cff](../CITATION.cff).
Dependencies retain their own licenses.
