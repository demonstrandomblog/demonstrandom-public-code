# EGraphs examples

Code accompanying [E-Graph Basics](https://demonstrandom.com/reasoning/posts/egraph/index.html),
published on 4 November 2024.

These are educational examples. Validate the behavior you rely on; the presence
of code or tests does not establish correctness. See the repository's
[research-use notice](../../TERMS.md).

> **AI warning:** This is experimental research code developed with AI assistance.
> Generation history and human-review coverage are incomplete; see individual
> files for additional disclosures. Passing automated checks does not establish
> a full human or mathematical review. Independently validate results you rely on.

## Data structures and reuse

`UnionFind` maintains disjoint sets using union by rank and path compression.
Call `make_set(value)` before `find(value)` or `union(a,b)`. `HashCons.cons`
interns an immutable, hashable value so equal values share one Python object.

`EGraph` groups expression nodes into equivalence classes. An `ENode(op,args)`
contains an operator name and a tuple of child class IDs; a leaf has no children.
`add` returns a class ID, `union` asserts equality of two classes, and `rebuild`
propagates equality through parent expressions. Call `find` to compare current
representatives after merges. The module does not infer algebraic rewrite rules;
the caller supplies equalities explicitly.

```python
from reasoning.egraphs.e_graphs import EGraph, ENode

graph = EGraph()
one = graph.add(ENode("1", ()))
two = graph.add(ENode("2", ()))
left = graph.add(ENode("+", (one, two)))
right = graph.add(ENode("+", (two, one)))
graph.union(left, right)  # Assert commutativity for these two expressions.
graph.rebuild()
assert graph.find(left) == graph.find(right)
```

These small implementations use Python's standard library; pytest is needed
only for verification. Extraction orders nodes by arity and operator name and
recursively expands children. It has no cost optimizer or cycle guard, so do
not use `extract` on cyclic expression classes.

## Run

From the repository root:

```sh
python -m pip install pytest
python -m pytest -q reasoning/egraphs/union_find.py reasoning/egraphs/hashcons.py reasoning/egraphs/e_graphs.py
```

Hash-consing distinguishes unequal values with colliding hashes. EGraph
extraction uses a simple node ordering, rather than a minimum-cost search.

## License

These examples are covered by the repository's
[PolyForm Noncommercial License 1.0.0](../../LICENSE).
When redistributing any part of them, include the license terms or their URL
and the required attribution lines in [NOTICE](../../NOTICE).

## Citation

If these examples contribute to your work, please cite the post and record the
repository commit or release you used.

Repository citation metadata is in [CITATION.cff](../../CITATION.cff).
Academic citation is requested separately from the license's redistribution
notice requirements.
