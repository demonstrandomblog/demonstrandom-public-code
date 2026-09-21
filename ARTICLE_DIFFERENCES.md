# Article and implementation differences

Implementation differences and article errata checked on 20-21 September 2026.

| Component | Relationship to its article |
|---|---|
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
