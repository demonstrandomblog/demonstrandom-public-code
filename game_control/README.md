# Game control

Generated from the appendix of [Engineering Game Types](https://demonstrandom.com/game_theory/posts/engineering_game_types/index.html).
The article is the authoritative source. The generator preserved 103 definitions
and the concrete code blocks from all ten vignettes.

> **AI warning:** This is experimental research code. AI assistance has been
> used in the project and in this cleanup; historical generation details are
> not established for every file. Existing file-level disclosures are retained.
> Passing automated checks does not establish a full human or mathematical
> review. Independently validate the behavior and results you rely on.

The article specifically discloses ChatGPT-generated code with author-provided
architecture guidance. No complete human review is recorded; human-review.json
remains pending. Automated verification passed the current article vignettes,
mathematical regressions, PID checks, and polynomial-solver checks.

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
directory and uses the installed package. No research-workspace files or blog
checkout are needed to run the examples or checks.

## Article provenance and regeneration

The implementation and example bytes match the recorded article revision.
[provenance.json](provenance.json) records its hash, 103 definitions, all ten
vignette groups, original generator history, and the portable check sources.
The historical workspace paths in that record identify provenance only.

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
The full human-review status remains pending. Release checks used CPU execution
on Python 3.13.5 in WSL; other platforms, dependency versions, and GPU behavior
have not been validated. A wheel installation and source-distribution checks
used a virtual environment sharing preinstalled scientific dependencies, not
a fresh resolution/download of all dependencies.

## License and citation

Original code is offered under [PolyForm Noncommercial 1.0.0](LICENSE).
Preserve [NOTICE](NOTICE). Dependencies retain their own terms.
Cite Kevin T. Procopio, Engineering Game Types, Demonstrandom, and the repository
commit used. See [CITATION.cff](../CITATION.cff) and [VALIDATION.md](../VALIDATION.md).
