# AI-assisted research code; no blanket human or mathematical review is claimed.
"""Snapshot-based identification of linear and nonlinear dynamical systems.

DMD/EDMD/kernel DMD accept X and one-step successors Y as (variables,samples).
SINDy uses the same X layout with derivatives shaped (samples,variables).
SINDyC adds controls to the feature library. DMD/EDMD invert retained singular
values directly; select rank explicitly for ill-conditioned data. The Arnoldi
variant is unavailable. Running this file executes the numerical demonstrations.
"""
import math
import numpy as np

from itertools import combinations_with_replacement


def reduced_svd(A, r=None):
    U, S, Vt = np.linalg.svd(A, full_matrices=False)

    if r is not None:
        U = U[:,:r]
        S = S[:r]
        Vt = Vt[:r, :]

    return U, S, Vt

def heat_diffusion_step_fn(u, r=0.001):
    u_new = u.copy()
    u_new[1:-1,1:-1] = (
        u[1:-1,1:-1] +
        r * (u[2:,1:-1] + u[:-2,1:-1] + u[1:-1,2:] + u[1:-1,:-2] - 4 * u[1:-1,1:-1])
    )
    return u_new

def heat_diffusion(time, m=50, n=50, alpha=0.01, dt=0.1, dx=1.0):

    # Discretization
    steps = int(time / dt)

    # Warn when the explicit two-dimensional diffusion stability bound is exceeded.
    r = alpha * dt / dx**2
    if r > 0.25:
        print("Warning: Scheme may be unstable (r > 0.25)")

    # Initial condition: hot spot in center
    u = np.zeros((m, n))
    u[m//2, n//2] = 100

    # Time stepping
    for _ in range(steps):
        u = heat_diffusion_step_fn(u, r=r)

    return u

def reduced_frobenius_ratio(matrix, r=None):
    if r is None:
        print("Warning: r not supplied to Frobenius ratio")
        return 1

    denominator = np.linalg.norm(matrix, ord="fro")**2

    U, S, Vt = reduced_svd(matrix, r=r)
    X_r = np.matmul(np.matmul(U, np.diag(S)), Vt)

    numerator = np.linalg.norm(X_r, ord="fro")**2
    return numerator/denominator

def dmd(X, Y, r=None):
    """Fit discrete-time dynamic mode decomposition from column snapshots X,Y.

    Return (modes, eigenvalues, amplitudes, modes@diag(amplitudes)); the last
    matrix is not a time trajectory. r truncates the SVD; retained singular
    values must be nonzero because this routine inverts them directly.
    """
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    if r is not None and r < len(S):
        U = U[:, :r]
        S = S[:r]
        Vt = Vt[:r, :]
    print(U.shape, S.shape, Vt.shape)

    Atilde = U.conj().T @ Y @ Vt.conj().T @ np.diag(1/S)
    eigenvalues, eigenvectors = np.linalg.eig(Atilde)
    modes = Y @ Vt.conj().T @ np.diag(1/S) @ eigenvectors

    x0 = X[:, 0]
    amplitudes = np.linalg.lstsq(modes, x0, rcond=None)[0]

    reconstruction = (modes @ np.diag(amplitudes))

    return modes, eigenvalues, amplitudes, reconstruction

# Original source attribution: Claude. Arnoldi is unimplemented.
def dmd_arnoldi(X, Y, r):
    raise NotImplementedError("The Arnoldi DMD variant is not implemented; use dmd() instead.")
    n, m = X.shape
    V = np.zeros((n, r), dtype=X.dtype)
    H = np.zeros((r + 1, r), dtype=X.dtype)

    # Arnoldi iteration
    v1 = X[:, 0]
    V[:, 0] = v1 / np.linalg.norm(v1)

    for j in range(r-1):
        Xv = X.conj().T @ V[:, j]  # Project basis onto data (size m)
        w = Y @ Xv  # Apply Y to projection (size n)

        # Modified Gram-Schmidt
        for i in range(j+1):
            H[i, j] = V[:, i].conj() @ w
            w = w - H[i, j] * V[:, i]

        H[j+1, j] = np.linalg.norm(w)

        if H[j+1, j] > 1e-12:
            V[:, j+1] = w / H[j+1, j]
        else:
            # Early termination if numerically dependent
            r_actual = j + 1
            V = V[:, :r_actual]
            H = H[:r_actual, :r_actual]
            break

    # Extract upper Hessenberg matrix
    H_upper = H[:r, :]

    # Eigendecomposition of Hessenberg matrix
    eigenvalues, eigenvectors = np.linalg.eig(H_upper)

    # Construct DMD modes
    modes = V @ eigenvectors

    # Compute amplitudes
    amplitudes = eigenvectors / (modes.conj().T @ X[:, 0])

    return modes, eigenvalues, amplitudes

def collect_snapshots(step_fn, z0, num_steps):
    m = z0.size
    X = np.zeros((m, num_steps))
    Y = np.zeros((m, num_steps))

    state = z0.copy()
    for i in range(num_steps):
        X[:, i] = state.flatten()
        next_state = step_fn(state)
        Y[:, i] = next_state.flatten()
        state = next_state

    return X, Y

def edmd(X, Y, basis, r=None):
    obs_X = np.vstack([f(X) for f in basis])        # k × m
    obs_Y = np.vstack([f(Y) for f in basis])        # k × m

    U, S, Vt = np.linalg.svd(obs_X, full_matrices=False)

    if r is not None and r < len(S):
        U = U[:, :r]
        S = S[:r]
        Vt = Vt[:r, :]

    S_inv_diag = np.diag(1.0 / S)

    K = U.conj().T @ obs_Y @ Vt.conj().T @ S_inv_diag # r_eff x r_eff
    eig_vals, eig_vecs = np.linalg.eig(K)
    eig_vecs_K = U @ eig_vecs
    print(U.shape, Vt.shape,S_inv_diag.shape, eig_vecs_K.shape, obs_X.shape)

    modes = obs_X.T @ eig_vecs_K
    return K, eig_vals, eig_vecs, modes, eig_vecs_K

def polynomial_basis(x_dim, degree=2):
    basis = []
    symbolic = []

    basis.append(lambda x: np.ones((1, x.shape[1])))
    symbolic.append("1")

    for d in range(1, degree + 1):
        for combo in combinations_with_replacement(range(x_dim), d):
            basis.append(lambda x, combo=combo: np.prod(np.array([x[i:i+1, :] for i in combo]), axis=0))
            term = "*".join([f"x_{i}" for i in combo])
            symbolic.append(term)

    return basis, symbolic

def hermite_basis(x_dim, degree=4):
    dictionary = []
    symbolic = ["Not implemented"]

    dictionary.append(lambda x: np.ones((1, x.shape[1])))

    for i in range(x_dim):
        def create_hermite_fn(idx, n):
            if n == 0:
                return lambda x: np.ones((1, x.shape[1]))
            elif n == 1:
                return lambda x: x[idx:idx+1, :]
            else:
                h_nm1 = create_hermite_fn(idx, n-1)
                h_nm2 = create_hermite_fn(idx, n-2)
                return lambda x: x[idx:idx+1, :] * h_nm1(x) - (n-1) * h_nm2(x)

        for n in range(degree + 1):
            dictionary.append(create_hermite_fn(i, n))

    return dictionary

def radial_basis(centers, x_dim, sigma=1.0):
    dictionary = []
    symbolic = []

    dictionary.append(lambda x: np.ones((1, x.shape[1])))

    for j in range(centers.shape[1]):
        dictionary.append(lambda x, j=j: np.exp(-np.sum((x - centers[:, j:j+1])**2, axis=0, keepdims=True) / (2 * sigma**2)))
        center_str = ", ".join([f"{centers[i,j]:.2f}" for i in range(x_dim)])
        symbolic.append(f"exp(-||x - ({center_str})||^2/(2*{sigma}^2))")

    return dictionary, symbolic

# kernels

def linear_kernel(X1, X2):
    return X1.T @ X2

def rbf_kernel(X1, X2, sigma=1.0):
    gamma = 1.0 / (2 * sigma**2)
    n_samples1 = X1.shape[1]
    n_samples2 = X2.shape[1]
    K = np.zeros((n_samples1, n_samples2))
    for i in range(n_samples1):
        for j in range(n_samples2):
            diff = X1[:, i] - X2[:, j]
            K[i, j] = np.exp(-gamma * np.dot(diff, diff))
    return K

def kdmd(X, Y, kernel_fn, r=None, epsilon=1e-8):
    m_minus_1 = X.shape[1]
    G_XX = kernel_fn(X, X)
    G_YX = kernel_fn(Y, X)

    U_g, S_g, Vt_g = reduced_svd(G_XX, r=r)
    S_g_inv = np.zeros_like(S_g)
    significant_s_vals = S_g > epsilon  # Absolute singular-value cutoff.
    S_g_inv[significant_s_vals] = 1.0 / S_g[significant_s_vals]

    if r is None:
        r_eff = np.sum(significant_s_vals)
    else:
        r_eff = r

    U_g_r = U_g[:, :r_eff]
    S_g_inv_r = S_g_inv[:r_eff]

    G_XX_pinv_svd = U_g_r @ np.diag(S_g_inv_r) @ U_g_r.T

    K_op_kernel = G_YX @ G_XX_pinv_svd
    eigenvalues, eigenvectors_alpha = np.linalg.eig(K_op_kernel)
    modes_coeffs = eigenvectors_alpha

    return eigenvalues, modes_coeffs, G_XX

def lorenz_odes(x, y, z, sigma=10, rho=28, beta=8/3):
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return np.array([dxdt, dydt, dzdt])

def sho_odes(x, v, omega_sq=4.0):
    dxdt = v
    dvdt = -omega_sq * x
    return np.array([dxdt, dvdt])

def rk4_step(ode_func_f_X, y_current, dt, **ode_kwargs):
    k1 = ode_func_f_X(y_current, **ode_kwargs)
    k2 = ode_func_f_X(y_current + dt/2 * k1, **ode_kwargs)
    k3 = ode_func_f_X(y_current + dt/2 * k2, **ode_kwargs)
    k4 = ode_func_f_X(y_current + dt * k3, **ode_kwargs)
    y_next = y_current + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    return y_next

def sindy_library(X, poly_order=3, include_sine=False, include_cosine=False):
    n_vars, n_samples = X.shape

    # Start with constant term
    library_functions = [np.ones(n_samples)]
    descriptions = ['1']

    # Add polynomial terms
    for order in range(1, poly_order + 1):
        for combo in combinations_with_replacement(range(n_vars), order):
            if order == 1:
                var_idx = combo[0]
                library_functions.append(X[var_idx, :])
                descriptions.append(f'x_{var_idx}')
            else:
                term = np.ones(n_samples)
                term_desc = []
                for var_idx in combo:
                    term *= X[var_idx, :]
                    term_desc.append(f'x_{var_idx}')
                library_functions.append(term)
                descriptions.append('*'.join(term_desc))

    # Add trigonometric terms if requested
    if include_sine:
        for i in range(n_vars):
            library_functions.append(np.sin(X[i, :]))
            descriptions.append(f'sin(x_{i})')

    if include_cosine:
        for i in range(n_vars):
            library_functions.append(np.cos(X[i, :]))
            descriptions.append(f'cos(x_{i})')

    Theta = np.column_stack(library_functions)

    return Theta, descriptions

def sequential_threshold_least_squares(Theta, dXdt, lambda_reg=0.1, max_iter=10):
    n_features = dXdt.shape[1]

    # Initialize coefficient matrix
    Xi = np.linalg.lstsq(Theta, dXdt, rcond=None)[0]

    # Iterative thresholding
    for iteration in range(max_iter):

        # Find small coefficients to remove
        small_inds = np.abs(Xi) < lambda_reg

        # Set small coefficients to zero
        Xi[small_inds] = 0

        # Identify active (non-zero) coefficients for each feature
        for i in range(n_features):
            big_inds = ~small_inds[:, i]
            if np.any(big_inds):

                # Recompute non-zero coefficients using least squares
                Xi[big_inds, i] = np.linalg.lstsq(
                    Theta[:, big_inds], dXdt[:, i], rcond=None
                )[0]
            else:
                Xi[:, i] = 0

    return Xi

def finite_difference(X, dt):
    n_vars, n_samples = X.shape

    # Use central differences where possible, forward/backward at boundaries
    dXdt = np.zeros((n_samples, n_vars))

    # Forward difference at first point
    dXdt[0, :] = (X[:, 1] - X[:, 0]) / dt

    # Central differences for interior points
    for i in range(1, n_samples - 1):
        dXdt[i, :] = (X[:, i + 1] - X[:, i - 1]) / (2 * dt)

    # Backward difference at last point
    dXdt[-1, :] = (X[:, -1] - X[:, -2]) / dt

    return dXdt

def sindy(X, dXdt, poly_order=3, lambda_reg=0.1, include_sine=False,
          include_cosine=False, max_iter=10):

    # Build library of candidate functions
    """Fit sparse derivative equations using sequential thresholded least squares.

    X has shape (variables,samples), dXdt has shape (samples,variables).
    Return (Xi, descriptions), with Xi shaped (library_features,variables).
    lambda_reg thresholds coefficient magnitude; it is not an L1 penalty.
    """
    Theta, descriptions = sindy_library(
        X,
        poly_order=poly_order,
        include_sine=include_sine,
        include_cosine=include_cosine
    )

    # Sparse regression using sequential thresholded least squares
    Xi = sequential_threshold_least_squares(Theta, dXdt, lambda_reg, max_iter)

    return Xi, descriptions


def print_equations(Xi, descriptions, feature_names=None):
    n_functions, n_features = Xi.shape

    if feature_names is None:
        feature_names = [f'x_{i}' for i in range(n_features)]

    for i in range(n_features):

        # Build equation string
        terms = []
        for j in range(n_functions):
            coef = Xi[j, i]
            if abs(coef) > 1e-10:  # Only include non-zero terms
                if abs(coef - 1.0) < 1e-10:
                    terms.append(f"{descriptions[j]}")
                elif abs(coef + 1.0) < 1e-10:
                    terms.append(f"-{descriptions[j]}")
                else:
                    terms.append(f"{coef:.6f}*{descriptions[j]}")

        if terms:
            equation = " + ".join(terms).replace(" + -", " - ")
            print(f"d{feature_names[i]}/dt = {equation}")
        else:
            print(f"d{feature_names[i]}/dt = 0")
    print()

def generate_lorenz_data(initial_conditions, sigma=10, rho=28, beta=8/3, t_final=10, dt=0.01):

    def lorenz_rhs(state, sigma=sigma, rho=rho, beta=beta):
        x, y, z = state
        return np.array([
            sigma * (y - x),
            x * (rho - z) - y,
            x * y - beta * z
        ])

    # Generate training data
    t_train = np.arange(0, t_final, dt)
    n_steps = len(t_train)

    X_train = np.zeros((3, n_steps))
    X_train[:, 0] = initial_conditions

    for i in range(1, n_steps):
        k1 = lorenz_rhs(X_train[:, i-1])
        k2 = lorenz_rhs(X_train[:, i-1] + dt/2 * k1)
        k3 = lorenz_rhs(X_train[:, i-1] + dt/2 * k2)
        k4 = lorenz_rhs(X_train[:, i-1] + dt * k3)
        X_train[:, i] = X_train[:, i-1] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    return X_train


def generate_simple_harmonic_oscillator_data(omega=2.0, t_final=10, dt=0.01):
    t = np.arange(0, t_final, dt)

    # Analytical solution
    x = np.cos(omega * t)
    xdot = -omega * np.sin(omega * t)

    X_train = np.vstack([x, xdot])

    return X_train



##### SINDyC

def sindyc_library(X, U, poly_order_x=3, poly_order_u=1,
                   include_cross=True, include_sine=False, include_cosine=False):

    n_vars, n_samples = X.shape
    n_ctrl = U.shape[0]

    feats = [np.ones(n_samples)]
    descriptions = ['1']

    # Polynomials in x
    for order in range(1, poly_order_x + 1):
        for combo in combinations_with_replacement(range(n_vars), order):
            term = np.prod([X[i, :] for i in combo], axis=0)
            feats.append(term)
            descriptions.append('*'.join([f'x_{i}' for i in combo]))

    # Polynomials in u
    for order in range(1, poly_order_u + 1):
        for combo in combinations_with_replacement(range(n_ctrl), order):
            term = np.prod([U[i, :] for i in combo], axis=0)
            feats.append(term)
            descriptions.append('*'.join([f'u_{i}' for i in combo]))

    # Cross terms (bilinear by default)
    if include_cross:
        for i in range(n_vars):
            for j in range(n_ctrl):
                feats.append(X[i, :] * U[j, :])
                descriptions.append(f'x_{i}*u_{j}')

    # Optional trigs on states
    if include_sine:
        for i in range(n_vars):
            feats.append(np.sin(X[i, :])); descriptions.append(f'sin(x_{i})')
    if include_cosine:
        for i in range(n_vars):
            feats.append(np.cos(X[i, :])); descriptions.append(f'cos(x_{i})')

    Theta = np.column_stack(feats)
    return Theta, descriptions

def sindyc(X, U, dt, poly_order_x=3, poly_order_u=1,
           include_cross=True, lambda_reg=0.1, max_iter=10,
           include_sine=False, include_cosine=False):
    dXdt = finite_difference(X, dt)
    Theta, descriptions = sindyc_library(
        X, U,
        poly_order_x=poly_order_x,
        poly_order_u=poly_order_u,
        include_cross=include_cross,
        include_sine=include_sine,
        include_cosine=include_cosine
    )
    Xi = sequential_threshold_least_squares(Theta, dXdt, lambda_reg=lambda_reg, max_iter=max_iter)
    return Xi, descriptions

def predator_prey_control_rhs(state, u, alpha=1.0, beta=0.5, delta=0.5, gamma=1.0, k1=0.8, k2=0.6):
    x, y = state
    u1, u2 = u
    dx = alpha*x - beta*x*y + k1*u1
    dy = delta*x*y - gamma*y - k2*u2
    return np.array([dx, dy])

def u1(t):
    return 0.3*np.sin(0.3*t) + 0.2*np.cos(0.11*t)

def u2(t):
    return 0.25*np.sin(0.17*t + 0.7)

def simulate_predator_prey_with_control(x0, t, u1, u2, rhs=predator_prey_control_rhs):
    n  = len(t)
    dt = t[1] - t[0]
    X  = np.zeros((2, n))
    U  = np.zeros((2, n))
    X[:, 0] = np.asarray(x0, dtype=float)

    for k in range(1, n):
        u_vec = np.array([u1(t[k-1]), u2(t[k-1])], dtype=float)
        U[:, k-1] = u_vec
        X[:, k] = rk4_step(rhs, X[:, k-1], dt, u=u_vec)

    U[:, -1] = np.array([u1(t[-1]), u2(t[-1])], dtype=float)
    return X, U, dt




if __name__ == "__main__":

    # Singular-value truncation examples
    print("Lecture One")
    X = np.random.normal(size=[20,18])
    U_full, S_full, Vt_full = reduced_svd(X, r=18)

    for r in range(0, 20):
        U, S, Vt = reduced_svd(X, r=r)
        X_r = np.matmul(np.matmul(U, np.diag(S)), Vt)
        frob_norm_reconstruction_error = np.linalg.norm(X - X_r, ord="fro")
        frob_norm_error_approximator = np.sqrt(np.sum(np.square(np.diag(S_full)[r:])))
        print(r, np.std(X), np.std(X_r), frob_norm_reconstruction_error, frob_norm_error_approximator) # Reconstruction error

        two_norm_reconstruction_error = np.linalg.norm(X - X_r, ord=2)
        two_norm_error_approximator = S_full[r] if r < len(S) else 0
        print(two_norm_reconstruction_error, two_norm_error_approximator)

        # Eckart-Young-Mirsky theorem

    # Heat-diffusion snapshot examples
    print("\nLecture Two")
    heat_diffusion_example = heat_diffusion(1)
    print(heat_diffusion_example.shape)
    for r in range(0, 20):
        ratio = reduced_frobenius_ratio(heat_diffusion_example, r=r)
        print(ratio)


    # Dynamic mode decomposition example
    print("\nLecture Three: DMD")
    # min_A || Y - AX ||_fro -> This is learning the system
    # Two ways to solve this-> pseudoinverse (can compactly get from SVD)
    # Y*V*Sigma^dagger*U^star
    # Z_n = A^n Z_0
    # Then X = [Z_0, Z_1, ... Z_n-1] and Y = [Z_1, Z_2..., Z_n]

    z0 = np.zeros((10, 10))
    z0[10//2, 10//2] = 100
    X, Y = collect_snapshots(heat_diffusion_step_fn, z0, 1000)
    modes, eigenvalues, amplitudes, reconstruction = dmd(X, Y, r=5)
    print("DMD eigenvalues:", eigenvalues)


    print("\nLecture Six: Koopman operator")

    X, Y = collect_snapshots(heat_diffusion_step_fn, z0, 150)
    basis, symbolic = polynomial_basis(X.shape[0], degree=1)
    K, eig_vals, eig_vecs, modes, eig_vecs_K = edmd(X, Y, basis, r=5)
    print(eig_vals)
    print(symbolic)
    print(eig_vals, eig_vecs, eig_vecs_K)
    print(modes)


    print("Lecture 7: Kernel")
    X, Y = collect_snapshots(heat_diffusion_step_fn, z0, 150)
    basis, symbolic = polynomial_basis(X.shape[0], degree=1)
    K, eig_vals, eig_vecs = kdmd(X, Y, rbf_kernel, r=5)
    print(symbolic)
    print(eig_vals, eig_vecs, eig_vecs_K)
    print(modes)


    print("Lecture 8: SINDy Method")
    dt = 0.01

    X = generate_simple_harmonic_oscillator_data()
    dXdt = finite_difference(X, dt)
    Xi, desc = sindy(X, dXdt, poly_order=1, lambda_reg=0.1)
    print_equations(Xi, desc, feature_names=['x0', 'x1'])


    initial_conditions = [-8, 8, 27]
    X = generate_lorenz_data(initial_conditions)
    dXdt = finite_difference(X, dt)
    Xi, desc = sindy(X, dXdt, poly_order=3, lambda_reg=0.1)
    print_equations(Xi, desc, feature_names=['x', 'y', 'z'])



    # SINDy with control example

    print("\nLecture 9: SINDYc (predator–prey with control)")
    t = np.arange(0.0, 50.0, 0.01)
    u1 = lambda _t: 0.3*np.sin(0.3*_t) + 0.2*np.cos(0.11*_t)
    u2 = lambda _t: 0.25*np.sin(0.17*_t + 0.7)

    Xpp, Upp, dt_pp = simulate_predator_prey_with_control(
        x0=(1.5, 1.0),
        t=t,
        u1=u1,
        u2=u2,
        rhs=predator_prey_control_rhs
    )

    Xi_c, desc_c = sindyc(
        Xpp, Upp, dt_pp,
        poly_order_x=2,
        poly_order_u=1,
        include_cross=True,
        lambda_reg=0.05,
        max_iter=15
    )

    print_equations(Xi_c, desc_c, feature_names=['x','y'])
