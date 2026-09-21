# Article and implementation differences

Checked against local published-source files on 20-21 September 2026. The blog
sources are the claim baseline. This record does not edit or deploy them.
The full implementation history remains in Git.

| Component | Relationship to its article |
|---|---|
| EGraphs | Repository fixes equal-hash/unequal-object collisions and consistently names `enode_to_eclass_id`; the article constructor uses another name. Union-by-rank prose reverses the parent direction used by both implementations. |
| Selectorate | Hierarchy now follows the written nested recursion. Invalid denominators raise rather than using the article's numerical clamp. Earlier public results for three or more levels are superseded. |
| Canonicalization | Exact enumeration replaces the article's faulty hard canonicalization. With player exchange there are 78 strict classes; the article also cites 144, which keeps players distinguished. Its differentiable-ID wording applies neither to discrete ranks nor hashing. |
| Lie groups | Repository corrects rotations, principal logs, rigid-motion exp/log, operation ordering, and matrix/coordinate APIs. The basic variational integrator is now included; final residual determines Newton success. The separate time-aware variant remains excluded. |
| System identification | Heat diffusion now uses alpha*dt/dx**2 as in the equation. SINDy implements the later STLS section; it does not solve the introduction's LASSO objective. Truncated DMD uses a reduced projection; rectangular U does not define a similarity transformation of the full system. |
| Game control | Implementation and ten example groups match the extracted article code exactly. Existing human-review status is retained. |
| Color metric | Original component interpolation and SVD are reproducible, but the field is indefinite at 160 grid points. A separately identified log-Euclidean mode fixes positivity and changes the field/spectrum. |
| Inspection bias | Fitting code accompanies the later provisional functional-information draft; the published Inspection Bias post supplies the background. The draft's effective-weight terminology replaces the recovered README's older probability-only description. |
| Cultural counting | Formula and 0.84 worked example agree. Five overlap figure drivers are included; historical export pixel identity is unverified. Integer threshold comparisons and negative-radius handling are corrected. |

Further canonicalization article errata for human review: its opening example
has dominant strategies despite being labelled Battle of the Sexes; the stated
3x+7 transformation does not produce two of the printed entries. Ordinal
ranking can change mixed-equilibrium probabilities, and soft ranks are one-based
before the later integer conversion. The repository does not silently rewrite
those examples or the article's mathematical claims.

## Published simulation companions added 21 September 2026

- **Gradient learning:** the recovered loop computes each player's own gradients
  before either Adam update, matching the article's simultaneous-update intent.
  The displayed article loop uses sequential `backward()`/optimizer calls instead.
  The public CLI fixes seed 0; exact stochastic prints therefore differ. The
  action-only policy is included; the later observation-conditioned policy is not.
- **Variational integration:** retains the midpoint Lagrangian and the published
  free-rotor Noether extension, with repaired public Lie primitives. Success now
  requires the final discrete Euler-Lagrange residual below tolerance, including
  after the last Newton iteration. Positive finite step/tolerance bounds are
  checked. The CLI saves bounded examples; it retains the article's first-order
  q1 initialization and backward-difference energy estimate.
- **Differential games:** retains the recovered Stag Hunt model and whole-step
  time convention. Invalid step sizes and simulation bounds now raise. The
  optional collision class is excluded because its source neither accesses
  state correctly nor applies its proposed position correction. The article
  scenarios do not need that class.

The corresponding articles were confirmed live before selection. Their local
published sources remain unchanged. The source snapshots and exact prepared
hashes are recorded in SOURCE_PROVENANCE.json.
