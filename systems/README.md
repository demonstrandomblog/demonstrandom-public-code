# System identification

Code accompanying [Demonstrandom](https://demonstrandom.com/ml/posts/linear_methods_for_dynamical_systems/index.html).

> **AI warning:** This is experimental research code developed with AI assistance.
> Generation history and human-review coverage are incomplete; see individual
> files for additional disclosures. Passing automated checks does not establish
> a full human or mathematical review. Independently validate results you rely on.

## Data layout and algorithms

Snapshots use NumPy arrays with shape `(state_variables, samples)`.
Columns of `X` contain states, and the matching columns of `Y` contain their
one-step successors. `collect_snapshots(step_fn, initial_state, num_steps)`
builds this pair by flattening each state. `dmd(X,Y,r)` uses a rank-`r` singular
value decomposition to approximate the map `Y = A@X`. It returns modes,
discrete-time eigenvalues, initial amplitudes, and the matrix of modes scaled
by amplitudes. That last matrix is not a full trajectory reconstruction.

Extended DMD (`edmd`) lifts snapshots through a supplied list of observables;
each observable accepts a snapshot matrix and returns one row. Kernel DMD
(`kdmd`) uses a kernel callable returning the pairwise sample Gram matrix.
The kernel implementation uses an absolute singular-value cutoff `epsilon`.

Sparse identification of nonlinear dynamics (`sindy`) regresses time
derivatives against polynomial and optional trigonometric features. It takes
`X` with shape `(variables,samples)` and `dXdt` with shape `(samples,variables)`.
It returns coefficients `Xi` with shape `(features,variables)` and feature
labels, so `Theta @ Xi` predicts derivatives. `finite_difference(X,dt)` builds
the derivative array; differentiation amplifies measurement noise. `sindyc`
adds sampled control variables to the feature library.

```python
import numpy as np
from systems.data_driven_systems import dmd

X = np.array([[1., 0., 1.], [0., 1., 1.]])
Y = np.diag([0.9, 0.5]) @ X
modes, eigenvalues, amplitudes, scaled_modes = dmd(X, Y, r=2)
assert np.allclose(np.sort(eigenvalues), [0.5, 0.9])
```

The heat example uses a five-point explicit diffusion step with
`r = alpha*dt/dx**2`; stability requires `r <= 1/4`. Boundary values remain
fixed. The helper prints a warning when this bound is exceeded.

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
