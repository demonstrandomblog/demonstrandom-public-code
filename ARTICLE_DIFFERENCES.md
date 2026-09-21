# Article and implementation differences

Implementation differences and article errata checked on 20-21 September 2026.

| Component | Relationship to its article |
|---|---|
| Invariant theory | Exact saturated integer kernels; bounded separator candidates; zero-dimensional HSOP check; exact signed-permutation substitution; corrected relation coordinates and classical graded ranks. Pairwise Koszul relations are explicitly incomplete. |
| Game invariant coordinates | Article scripts made importable and reproducible; corrected degree-six products, fingerprint degrees, sampled-rank labels, and Jacobian scaling. Candidate atlases retain the article limitations on separation. |
| Allometry | Supplementary PanTHERIA example. The article uses historical Kleiber and Huxley figures; this helper plots separately supplied data. Metabolic-study mass is the default and oxygen-consumption units remain mL O2/hour. No fitted exponent or validation of the scaling theory is claimed. |
| EGraphs | Hash-consing distinguishes equal-hash, unequal objects. The implementation consistently names `enode_to_eclass_id`; the article constructor uses another name. The article's union-by-rank prose reverses the parent direction used by the implementation. |
| Selectorate | Hierarchy follows the written nested recursion. Invalid denominators raise rather than using the article's numerical clamp. Earlier product-form results for three or more levels are superseded. |
| Canonicalization | Exact enumeration replaces the article's hard sorting path. With player exchange there are 78 strict classes; 144 keeps players distinguished. The discrete rank/hash path is not differentiable. |
| Lie groups and variational integration | Corrected rotations, principal logs, rigid-motion exp/log, operation ordering, and matrix/coordinate APIs. Newton success requires the final discrete Euler-Lagrange residual below tolerance. The midpoint Lagrangian and free-rotor Noether calculation follow the articles. |
| Gradient learning | Each player's own gradient is computed before either Adam update, matching the article's simultaneous-update intent. The article's displayed loop uses sequential backward/optimizer calls. The example fixes seed 0, so stochastic prints differ. |
| Differential games | The Stag Hunt model reproduces the article's payoff matrix. Invalid time steps and simulation bounds raise. Simulation retains whole steps and may finish beyond the requested end time. |
| System identification | Heat diffusion uses alpha*dt/dx**2 as in the equation. SINDy implements the STLS section rather than the introduction's LASSO objective. Truncated DMD uses a reduced projection; rectangular U does not define a similarity transformation of the full system. |
| Game control | Implementation and ten example groups match the extracted article code. |
| Color metric | Original component interpolation and SVD are reproducible, but the field is indefinite at 160 grid points. Log-Euclidean interpolation preserves positivity and changes the field/spectrum. |
| Inspection bias | The linked Inspection Bias article provides background on length-biased sampling. The package fits a provisional functional-information model using effective weights for formation/discovery and persistence. |
| Cultural counting | Formula and 0.84 worked example agree. Integer threshold comparisons and negative-radius handling are corrected. Five overlap figure drivers implement the article's counting calculations. |

## Canonicalization details

The article's opening example has dominant strategies despite being labelled
Battle of the Sexes; its stated 3x+7 transformation does not produce two printed
entries. Ordinal ranking can change mixed-equilibrium probabilities. Soft ranks
are one-based before the later integer conversion.

## Variational integration details

A small Newton update alone does not count as convergence: the final equation
residual must be below tolerance, including after the last permitted iteration.
Step size and tolerance must be positive and finite.

The example retains the article's first-order initialization `q1=q0+h*v0` and
backward-difference energy estimate. The pendulum convergence test uses an
independently accurate q1 to isolate integration error. The Noether calculation
uses the article's finite-difference generator.

[Source hashes](SOURCE_PROVENANCE.json) identify the implementations; the
[validation report](VALIDATION.md) describes the tested cases.

## Invariant theory details

A rational nullspace converted one vector at a time need not span the integer
kernel. Unimodular column operations now preserve the full lattice, including
signed and primitive solutions. Tests check exact rank and lattice saturation.

The separating routine returns every basis invariant through its cutoff.
An HSOP is not a separating set: for simultaneous sign reversal, x² and y²
identify (1,1) and (1,-1), while xy distinguishes them. The HSOP search tests
zero dimensionality through pure powers in its leading ideal and raises on an
incomplete search. Degree limits remain explicit for generators and secondaries.

Exact polynomial substitution supports signed-permutation representations.
Other numerical matrices raise instead of silently approximating an exact
rational action. The numerical Molien and point-orbit paths remain approximate.
Classical graded products are reduced to an independent basis. Elimination
relations use only generator coordinates. The syzygy routine returns pairwise
Koszul relations, not a complete syzygy-module basis. The article's
format_poly is named poly.show in this implementation.

## Game invariant coordinate details

The software appendix's 21 scripts use package-relative imports; the greedy
ordinal script imports the local helper functions it previously omitted.
SciPy is required by the numerical indifference solver in addition to the
article's stated NumPy/SymPy dependencies.

The strategy-only generator routine omits the degree-zero unit from its
generator count. The player-exchange routine now retains all lower-degree
generators: quartic-times-quadratic products are necessary for the degree-six
check. It recovers 5, 4, 8, and 5 generators in degrees 2 through 5 and no new
ones at degree 6 on the recorded numerical sample. The exact relation script
uses the nine quadratic/cubic coordinates only, not all 22 generators.

The appendix describes the 93-feature fingerprint as 42 quadratics and
51 cubics. In its code, two of those 51 features have degree six:
trace(M³) and det(M), where M is a quadratic Gram matrix. Nine other slots
are determinants of centered 3x3 blocks and vanish identically. Product-degree
bookkeeping and documentation reflect the actual formulas. The default
degree-four sampled rank is capped by 400 points; it cannot by itself measure
the full algebraic gap.

The binary-strategy numerical solver measures a Jacobian determinant at one
indifference root. Under uniform payoff scaling its degree is the number of
players, not n(n−1). This differs from the separate degree-six algebraic
elimination discriminant for three players. Root finding does not establish
feasibility in the probability simplex.

The stored 302- and 1,456-coordinate polarization lists lose three linear
payoff-sum coordinates after centering, leaving 299 and 1,453 nonzero
polynomials. Numeric NPZ files replace pickle serialization and dense exponent
arrays. Polarization residues are lifted to centered signed integers as in the
numerical experiment. All 630 sample pairs are now checked; the original
diagnostic printed only the ten closest pairs. The three atlases pass the
recorded 36-game experiment, without a global separation claim.
