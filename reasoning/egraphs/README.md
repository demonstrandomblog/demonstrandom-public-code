# EGraphs examples

Code accompanying [E-Graph Basics](https://demonstrandom.com/reasoning/posts/egraph/index.html),
published on 4 November 2024.

These are educational examples. Validate the behavior you rely on; the presence
of code or tests does not establish correctness. See the repository's
[research-use notice](../../TERMS.md).

> **AI warning:** This is experimental research code. AI assistance has been
> used in the project and in this cleanup; historical generation details are
> not established for every file. Existing file-level disclosures are retained.
> Passing automated checks does not establish a full human or mathematical
> review. Independently validate the behavior and results you rely on.

## Run

```sh
python -m pip install pytest
python -m pytest -q reasoning/egraphs/union_find.py reasoning/egraphs/hashcons.py reasoning/egraphs/e_graphs.py
```

Hash-consing now distinguishes unequal values with colliding hashes. The tuple
identity fixture and pytest return-value warnings are corrected. EGraph extraction
uses a simple node ordering; this is not a general minimum-cost extractor.

## License

These examples are covered by the repository's
[PolyForm Noncommercial License 1.0.0](../../LICENSE).
When redistributing any part of them, include the license terms or their URL
and the required attribution lines in [NOTICE](../../NOTICE).

## Citation

If these examples contribute to your work, please cite the post and record the
repository commit or release you used.

Procopio, K. T. (2024, November 4). *E-Graph Basics*. Demonstrandom.
https://demonstrandom.com/reasoning/posts/egraph/index.html

```bibtex
@misc{procopio2024egraph,
  author = {Procopio, Kevin T.},
  title = {E-Graph Basics},
  year = {2024},
  month = nov,
  url = {https://demonstrandom.com/reasoning/posts/egraph/index.html}
}
```

Repository citation metadata is in [CITATION.cff](../../CITATION.cff).
Academic citation is requested separately from the license's redistribution
notice requirements.
