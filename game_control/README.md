# Game control

Generated from the appendix of [Engineering Game Types](https://demonstrandom.com/game_theory/posts/engineering_game_types/index.html).
The article is the authoritative source. The generator preserved 103 definitions
and the concrete code blocks from all ten vignettes.

> **AI warning:** This is experimental research code developed with AI assistance.
> Generation history and human-review coverage are incomplete; see individual
> files for additional disclosures. Passing automated checks does not establish
> a full human or mathematical review. Independently validate results you rely on.

The article specifically discloses ChatGPT-generated code with author-provided
architecture guidance. No complete human review is recorded; human-review.json
remains pending. Automated verification passed the current article vignettes,
mathematical regressions, PID checks, and polynomial-solver checks.

## Model and API

A `Game(payoffs, shape)` contains a payoff array with trailing dimensions
`(players, *shape)`, where `shape` lists each player's action count. Optional
leading dimensions represent batches. `Map(function)` wraps a transformation;
`first.then_apply(second)` evaluates `second(first(input))`.
`Game.with_payoff_rule(rule)` constructs a parameterized family, calling
`rule(base_payoffs, theta)` for each parameter vector `theta`.

`effect(game,S)` averages out action axes outside `S` and removes means along
axes in `S`, isolating that interaction component. `gram(S,p,q)` measures the
mean product of players `p` and `q`'s components. Player and action indices are
zero based. These measurements describe payoff structure; they do not by
themselves prescribe an equilibrium or learning dynamic.

```python
import numpy as np
from game_control import Game, exact, expected, gram

interaction = exact([[-1, 1], [1, -1]])
game = Game(np.stack([interaction, -interaction]), (2, 2))
assert gram((0, 1), 0, 1)(game) == -1
assert list(expected(game, [[1, 0], [1, 0]])) == [-1, 1]
```

`exact` produces NumPy object arrays of SymPy values for symbolic computation.
Use `sympy.Rational` for exact fractions. The tensor path uses PyTorch values for
numerical evaluation and differentiation; supplied payoff and measurement
functions must support the backend being used.

| Operations | Contract |
|---|---|
| `ge`, `eq`, `Region` | Define measurement inequalities/equalities; `&` intersects regions and `|` forms unions. `pullback(mapping)` expresses a target in the mapping's input coordinates. |
| `Control`, `design` | A control contains command-to-parameter mapping, admissible commands, cost, and names. `design` pulls the target back through the game family. |
| `solve_affine`, `solve_polynomial` | Solve the corresponding affine or polynomial constraint problem. Inspect the returned status and verification result; timeouts or unknown results are not feasibility certificates. |
| `search`, `local_controls` | Numerical penalty search and local differential control analysis; these do not provide global feasibility or optimality proofs. |
| `segment_ok`, `winning_layers`, `rollout` | Check paths, compute finite-state winning sets, and simulate supplied transition/policy functions. Results depend on the supplied state/action model. |
| `pid_rollout`, `pid_loss` | Simulate bounded proportional-integral-derivative feedback and evaluate its tracking loss. |
| `observation_classes`, `audit_observations` | Compare games under selected measurements and test whether those observations suffice for the supplied control problems. |

`examples.py` contains complete model definitions for intervention, inverse
design, safe reachability, robust feedback, control authority, equivalent
commands, behavioral dynamics, temporary control, distributed control, and
observation sufficiency. Run it through `run_examples.py`, which supplies the
package symbols and numerical imports used by the examples.

## Install and verify

From the repository root, install the package in your Python environment and
run the complete verification suite:

```sh
python -m pip install './game_control[polynomial]'
python game_control/verify.py
```

This executes all ten article vignettes, then the mathematical, PID, and
polynomial suites. The 89 retained assertions cover symbolic/numerical agreement,
relabeling invariance, reachability, feasibility/optimality certificates, PID
constraints and gradients, and exact polynomial edge cases. The command fails
if any check fails. Run without Python's `-O` flag, which disables assertions.
Use `--output results.json` to save a fresh report; the bundled
[verification.json](verification.json) records the release run.

To run only the article vignettes:

```sh
python game_control/run_examples.py
```

The examples are generated blocks sharing the implementation namespace;
`run_examples.py` supplies it. All verification code is included in this
directory and uses the installed package.

## Article provenance and regeneration

The implementation definitions and example bodies match the recorded article
revision; generated headers describe their local purpose.
[provenance.json](provenance.json) records its hash, 103 definitions, all ten
vignette groups, generator hashes, and verification sources.

If you have the original Quarto article source, check the extraction with:

```sh
python game_control/tools/extract_article.py check --source /path/to/engineering_game_types/index.qmd
```

The article itself is not bundled. This command requires the exact source
revision recorded in the manifest. To extract a revised article into a separate
directory, use `generate --source PATH --output DIRECTORY`; it refuses to
overwrite different content. Regeneration creates the implementation and
examples only; it does not update release metadata or establish validation.

## Limits

These are experimental examples. Numerical searches and example feasibility
checks do not establish all mathematical claims or suitability for deployment.
No complete human review is recorded. The wheel and complete verifier were
checked on CPU with Python 3.13.5 in WSL, NumPy 2.5.3, SciPy 1.18.1,
SymPy 1.14.0, PyTorch 2.7.1+cpu, and z3-solver 5.1.0.0 in an isolated environment.
See the [validation record](../VALIDATION.md) for the tested configuration.

## License and citation

Original code is offered under [PolyForm Noncommercial 1.0.0](LICENSE).
Preserve [NOTICE](NOTICE). Dependencies retain their own terms.
Cite Engineering Game Types and the repository commit used. See [CITATION.cff](../CITATION.cff) and [VALIDATION.md](../VALIDATION.md).
