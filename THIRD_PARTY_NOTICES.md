# Third-party notices

This repository's license applies to its original material. Third-party source,
assets, and dependencies retain their own licenses and required notices.

## Current distribution

The source includes the components listed in README.md. Dependencies are
installed separately; no third-party runtime package or artwork is bundled. The two numerical tables
below retain their source notices.
EGraphs uses the Python standard library. The game components use PyTorch,
NumPy, SciPy, SymPy, and torchsort as listed in their requirements; game control
uses z3-solver for its optional polynomial solver and full verification suite.
pytest is used for verification. Those dependencies retain their upstream
licenses.

Existing AI-generation and attribution comments are preserved, including the
GPT-generated, unreviewed trainable Lie-group layer and its tests. Historical
provenance is not fully established for every file. The system-identification
Arnoldi draft retains its Claude attribution and explicit unimplemented status.
Game control retains the article's AI-generation disclosure.
The stable SO(3) logarithm repair consulted PyTorch3D's rotation-conversion
reference; PyTorch3D itself is not bundled or required. See the component's
limitations and the source provenance records.

Linked articles, papers, and implementations remain separate works. A reference
does not grant rights to copy their code, text, artwork, or data.

## Numerical tables

- Color metric: [MacAdam table notice](color_metric/data/NOTICE) and
  [upstream GPL-3.0 license](color_metric/data/LICENSE). The JSON transcription
  is separated from the original PolyForm-licensed analysis code.
- Inspection bias: [Hazen-Wong Table 1 notice](inspection_bias/DATA_NOTICE.md),
  with attribution, source link, transcription details, and CC BY-NC 4.0 terms.

## Adding or distributing third-party material

For any copied or adapted third-party code, include its upstream source,
version or revision, applicable license, original copyright notices, and any
required attribution or modification notices alongside the distributed files.
List the component here and keep its license text with the distribution.

Installed dependencies keep their upstream licenses. If a release bundles or
redistributes dependencies or assets, include the notices and other materials
required for that distribution. The top-level PolyForm license does not replace
those obligations.
