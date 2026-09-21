# System identification

Code accompanying [Demonstrandom](https://demonstrandom.com/ml/posts/linear_methods_for_dynamical_systems/index.html).

> **AI warning:** This is experimental research code developed with AI assistance.
> Generation history and human-review coverage are incomplete; see individual
> files for additional disclosures. Passing automated checks does not establish
> a full human or mathematical review. Independently validate results you rely on.

## Run

From the repository root, install this component's dependencies in your own
Python environment:

```sh
python -m pip install -r systems/requirements.txt
python -m pytest -q systems
python systems/data_driven_systems.py
```

## Scope and known limitations

DMD, EDMD, kernel DMD, SINDy, and SINDyC are teaching implementations.
Arnoldi is unimplemented and raises `NotImplementedError`.

Seven analytic regression tests cover exact linear DMD/EDMD/kernel-DMD spectra,
SINDy oscillator recovery, SINDyC control recovery, heat-diffusion parameter use,
and the explicit Arnoldi stub. The full demo also runs. The heat-diffusion
wrapper now passes its computed step size to the stepping function; the earlier
source silently used the default regardless of the supplied parameters.

The basic DMD/EDMD routines invert retained singular values directly. They do
not automatically choose a safe rank or regularization for ill-conditioned
inputs. Select the rank and validate noisy-data behavior for your application.

See [validation report](../VALIDATION.md) for the exact checks and their results.
Dependencies retain their own licenses. The original project code is covered
by [PolyForm Noncommercial 1.0.0](../LICENSE); preserve [NOTICE](../NOTICE).
For citation details see [CITATION.cff](../CITATION.cff).
