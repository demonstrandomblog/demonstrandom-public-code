# Demonstrandom code

Educational and experimental code accompanying [Demonstrandom](https://demonstrandom.com/).

## Components

> **AI warning:** This is experimental research code. AI assistance has been
> used in the project and in this cleanup; historical generation details are
> not established for every file. Existing file-level disclosures are retained.
> Passing automated checks does not establish a full human or mathematical
> review. Independently validate the behavior and results you rely on.

| Component | Scope | Status |
|---|---|---|
| [EGraphs](reasoning/egraphs/) | Union-find, hash-consing, equality examples | Teaching implementation; collision handling and test fixtures corrected |
| [Selectorate](games/selectorate/) | Model, hierarchy, channels, gradients | Nested hierarchy repaired; independent formula and gradient checks |
| [Canonicalization](games/canonicalization/) | Ordinal 2x2 game examples | Exact 2x2 orbit enumeration; all 576 strict games verified |
| [Lie groups and variational integration](games/geometric_controls/) | Rotations, rigid motions, Newton integrator, pendulum and rotor | 85 Lie checks plus 8 independent integrator checks |
| [Gradient learning](games/gradient_learning/) | Normal-form games and simultaneous policy gradients | Seeded 200-step example; independent payoffs and gradients |
| [Differential games](games/differential_games/) | State/dynamics simulation and normal-form conversion | Article Stag Hunt matrix and Euler/RK4 refinement checks |
| [System identification](systems/) | DMD, EDMD, kernel DMD, SINDy, SINDyC | Seven analytic checks and demo pass; Arnoldi unimplemented |
| [Game control](game_control/) | Engineering Game Types appendix and ten vignettes | Portable mathematical/PID/polynomial checks; human review pending |
| [Color metric](color_metric/) | Polynomial Killing-field search | Article table reproduced; positive-definite interpolation added |
| [Inspection bias](inspection_bias/) | Functional-information fitting draft | Four recorded fits and analytic checks reproduced |
| [Cultural counting](art_and_info/) | Hamming/weighted counting and five overlap figures | Brute-force count checks and article example reproduced |

Each component has its own dependency and run instructions. See
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
on the work, with credit to Kevin T. Procopio and Demonstrandom. Noncommercial
licensing supports that sharing while reserving commercial reuse outside the
licenses' existing permissions for a separate conversation.

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
