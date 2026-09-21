# Demonstrandom code

Educational and experimental code accompanying [Demonstrandom](https://demonstrandom.com/).

## Components

> **AI warning:** This is experimental research code developed with AI assistance.
> Generation history and human-review coverage are incomplete; see individual
> files for additional disclosures. Passing automated checks does not establish
> a full human or mathematical review. Independently validate results you rely on.

| Component | Scope | Article |
|---|---|---|
| [Invariant theory](invariants/) | Sparse rational polynomials, integer kernels, bounded generators and orbit diagnostics | [Building a Minimal Computational Invariant Theory Library](https://demonstrandom.com/symmetry/posts/computational_invariant_theory/) |
| [Game invariant coordinates](games/invariant_coordinates/) | 2x2 and three-player game examples, contrast blocks, and three candidate atlases | [Invariant Coordinates for Normal-Form Games](https://demonstrandom.com/symmetry/posts/invariant_coords_normal_form_games_v2/) |
| [Allometry](allometry/) | PanTHERIA mass and oxygen-consumption plotting helper | [Algebra and Allometry](https://demonstrandom.com/symmetry/posts/allometry/) (supplementary example) |
| [EGraphs](reasoning/egraphs/) | Union-find, hash-consing, equality examples | [E-Graph Basics](https://demonstrandom.com/reasoning/posts/egraph/index.html) |
| [Selectorate](games/selectorate/) | Model, hierarchy, channels, gradients | [Thoughts on Selectorate Theory](https://demonstrandom.com/governance/posts/game_theory_dictatorships_selectorate/index.html) |
| [Canonicalization](games/canonicalization/) | Ordinal 2x2 game examples | [Differentiable Game Canonicalization](https://demonstrandom.com/game_theory/posts/canonical_games/index.html) |
| [Lie groups and variational integration](games/geometric_controls/) | Rotations, rigid motions, Newton integrator, pendulum and rotor | [Controls from the Geometric Perspective](https://demonstrandom.com/game_theory/posts/discrete_controls_lagrange/index.html); [Noether's Theorem and Geometric Controls](https://demonstrandom.com/game_theory/posts/noether_geometric_controls/) |
| [Gradient learning](games/gradient_learning/) | Normal-form games and simultaneous policy gradients | [Learning Equilibria by Gradient Descent](https://demonstrandom.com/game_theory/posts/gradient_learning_nash/) |
| [Differential games](games/differential_games/) | State/dynamics simulation and normal-form conversion | [Differential Games and Stag Hunt](https://demonstrandom.com/game_theory/posts/differential_stag_hunt/) |
| [System identification](systems/) | DMD, EDMD, kernel DMD, SINDy, SINDyC | [Linear Methods for Learning Dynamical Systems](https://demonstrandom.com/ml/posts/linear_methods_for_dynamical_systems/index.html) |
| [Game control](game_control/) | Engineering Game Types appendix and ten vignettes | [Engineering Game Types](https://demonstrandom.com/game_theory/posts/engineering_game_types/index.html) |
| [Color metric](color_metric/) | Polynomial Killing-field search | [Do We See the Same Colors?](https://demonstrandom.com/theory_of_mind/posts/color_qualia_riemannian/index.html) |
| [Inspection bias](inspection_bias/) | Provisional functional-information models | [Inspection Bias](https://demonstrandom.com/ml/posts/inspection_bias/index.html) (background) |
| [Cultural counting](art_and_info/) | Hamming/weighted counting and five overlap figures | [Are We Approaching Cultural Saturation?](https://demonstrandom.com/essays/posts/cultural_saturation/index.html) |

Each component documents its own model, input conventions, usage examples,
status, dependencies, and limitations. Article links provide further background. See
[VALIDATION.md](VALIDATION.md) for checks and limitations and
[AI_NOTICE.md](AI_NOTICE.md) for AI assistance and review status.
[ARTICLE_DIFFERENCES.md](ARTICLE_DIFFERENCES.md) records corrections and article comparisons.

## License and terms

Original code and accompanying documentation are available under the
[PolyForm Noncommercial License 1.0.0](LICENSE). It permits use, modification,
and redistribution for its permitted purposes, including noncommercial use.
It also expressly permits use by the organizations listed in the license,
including educational institutions and public research organizations,
regardless of their funding. Uses outside those permissions require a separate
license; [contact the author](https://demonstrandom.com/contact.html) for
commercial licensing inquiries.

This is source-available software with use restrictions. When redistributing
any part of it, pass on the license terms or their URL and the attribution
lines in [NOTICE](NOTICE).

**Use at your own risk.** This code may be incomplete or incorrect and is not
intended for production use without further review and testing. See the
[licensing and research-use notice](TERMS.md) for scope, AI assistance,
review status, warranty, and support information, and the
[third-party notices](THIRD_PARTY_NOTICES.md) for material governed by other
licenses.

## Why noncommercial?

The goal is to help people learn from the posts, reproduce results, and build
on the work. Noncommercial licensing supports that sharing while reserving
commercial reuse outside the licenses' existing permissions for a separate
conversation.

Commercial projects can be worthwhile; asking first is the intended tradeoff.
This can make the code less convenient to adopt in products or libraries whose
users need commercial rights. PolyForm's express permissions for educational
institutions, public research organizations, and other listed organizations
still apply.

This rationale adds no license conditions. The applicable license governs each
use, including its exceptions and permissions. For uses requiring a separate
license, [contact the author](https://demonstrandom.com/contact.html).

## Citation

If this code contributes to your research or writing, please cite this
repository and the relevant Demonstrandom post. [CITATION.cff](CITATION.cff)
provides citation metadata. Record the commit or release you used so readers
can reproduce your work.

For the current EGraphs examples, see the
[E-Graph citation](reasoning/egraphs/README.md#citation).
Academic citation is requested; it is separate from the license's required
redistribution notices.
