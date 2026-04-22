"""Nonlinear time integration schemes for structural dynamics.

Three implicit integrators that accept a nonlinear assembly function
``assemble_fn(u) -> (K_T, f_int)`` and use Newton iteration at each
time step to solve the nonlinear equation of motion.

Backward Euler (1st order):
    Unconditionally stable, first-order accurate.
    Introduces significant numerical damping — high-frequency modes
    are rapidly attenuated.  Good for dissipation-tolerant problems.

Newmark-beta (2nd order, beta=1/4, gamma=1/2):
    Unconditionally stable, second-order accurate, no algorithmic
    damping for linear problems (trapezoidal / average acceleration).
    Energy nearly conserved for moderate nonlinearities.

Quasi-static Newton (no inertia):
    Solves a sequence of static equilibria as the external load is
    ramped from ``f_ext_start`` to ``f_ext_end``.  No mass, no
    velocity — just the equilibrium path.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve, splu

from femlab.core.newton import solve_newton_general
from femlab.core.corotational import (
    polar_decomposition_tet4,
    _corotational_local_displacement,
)
from femlab.core.element import tet4_element_stiffness


# ---------------------------------------------------------------------------
# Helper: apply Dirichlet BCs to a sparse matrix (zero rows/cols, diag=1)
# ---------------------------------------------------------------------------

def _apply_bc_sparse(K, bc_dofs):
    """Zero out rows/cols for constrained DOFs, set diagonal to 1."""
    K_lil = K.tolil()
    for dof in bc_dofs:
        K_lil[dof, :] = 0.0
        K_lil[:, dof] = 0.0
        K_lil[dof, dof] = 1.0
    return K_lil.tocsr()


# ---------------------------------------------------------------------------
# Strain energy helper for corotational elements
# ---------------------------------------------------------------------------

def compute_strain_energy_cr(
    nodes: np.ndarray,
    elements: np.ndarray,
    u: np.ndarray,
    D: np.ndarray,
) -> float:
    """Compute total strain energy using corotational local deformation.

    For each element, extracts the rotation-free local displacement
    via polar decomposition, then evaluates the quadratic strain energy
    in the corotated frame:

        PE = sum_e  0.5 * u_local^T @ K_lin_e @ u_local

    Args:
        nodes: (N, 3) reference node coordinates.
        elements: (M, 4) element connectivity.
        u: (3N,) global displacement vector.
        D: (6, 6) linear constitutive matrix.

    Returns:
        Total strain energy (scalar).
    """
    energy = 0.0
    for e in range(len(elements)):
        el_nodes = elements[e]
        X_ref = nodes[el_nodes]
        dof_map = np.empty(12, dtype=np.int64)
        for i in range(4):
            dof_map[3 * i] = 3 * el_nodes[i]
            dof_map[3 * i + 1] = 3 * el_nodes[i] + 1
            dof_map[3 * i + 2] = 3 * el_nodes[i] + 2
        u_e = u[dof_map]

        R, _U, _dN_dX, _V_ref = polar_decomposition_tet4(X_ref, u_e)
        u_local = _corotational_local_displacement(X_ref, u_e, R)
        K_lin = tet4_element_stiffness(X_ref, D)
        energy += 0.5 * u_local @ K_lin @ u_local

    return energy


# ---------------------------------------------------------------------------
# 1. Implicit backward Euler + Newton
# ---------------------------------------------------------------------------

def backward_euler_nl(
    M: sparse.csr_matrix,
    f_ext: np.ndarray,
    assemble_fn,
    u0: np.ndarray,
    v0: np.ndarray,
    dt: float,
    n_steps: int,
    bc_dofs: np.ndarray,
    bc_vals: np.ndarray | None = None,
    newton_max_iter: int = 20,
    newton_atol: float = 1e-8,
    verbose: bool = False,
) -> dict:
    """Implicit backward Euler time integration with Newton inner loop.

    First-order accurate, unconditionally stable.  Introduces numerical
    damping that attenuates oscillations over time.

    Equation of motion at t_{n+1}:
        M * a_{n+1} + f_int(u_{n+1}) = f_ext

    Backward Euler kinematic relations:
        v_{n+1} = (u_{n+1} - u_n) / dt
        a_{n+1} = (v_{n+1} - v_n) / dt

    Substituting:
        M * (u_{n+1} - u_n - dt*v_n) / dt^2 + f_int(u_{n+1}) = f_ext

    Newton residual:
        R(u) = M*(u - u_n - dt*v_n)/dt^2 + f_int(u) - f_ext

    Tangent:
        dR/du = M/dt^2 + K_T

    Args:
        M: (n, n) sparse mass matrix.
        f_ext: (n,) external force vector (constant).
        assemble_fn: Callable(u) -> (K_T_sparse, f_int_dense).
        u0: (n,) initial displacement.
        v0: (n,) initial velocity.
        dt: Time step size.
        n_steps: Number of time steps.
        bc_dofs: Indices of constrained DOFs.
        bc_vals: Prescribed displacement values (default: zeros).
        newton_max_iter: Max Newton iterations per step.
        newton_atol: Absolute residual tolerance for Newton.
        verbose: Print progress.

    Returns:
        dict with keys:
            t: (n_steps+1,) time array.
            u_hist: (n_steps+1, n) displacement history.
            v_hist: (n_steps+1, n) velocity history.
            ke_hist: (n_steps+1,) kinetic energy history.
    """
    n = len(u0)
    if bc_vals is None:
        bc_vals = np.zeros(len(bc_dofs))

    dt2 = dt * dt
    M_over_dt2 = M / dt2

    u_curr = u0.copy()
    v_curr = v0.copy()

    # Storage.
    t_hist = np.zeros(n_steps + 1)
    u_hist = np.zeros((n_steps + 1, n))
    v_hist = np.zeros((n_steps + 1, n))
    ke_hist = np.zeros(n_steps + 1)

    u_hist[0] = u0
    v_hist[0] = v0
    ke_hist[0] = 0.5 * v0.dot(M @ v0)

    print_interval = max(1, n_steps // 10)

    for step in range(1, n_steps + 1):
        # Momentum term: M * (u_n + dt*v_n) / dt^2
        rhs_inertia = M_over_dt2 @ (u_curr + dt * v_curr)

        u_trial = u_curr.copy()
        converged = False

        for k in range(newton_max_iter):
            K_T, f_int = assemble_fn(u_trial)

            R = M_over_dt2 @ u_trial - rhs_inertia + f_int - f_ext
            R[bc_dofs] = 0.0

            R_norm = float(np.linalg.norm(R))
            if R_norm < newton_atol:
                converged = True
                break

            K_eff = M_over_dt2 + K_T
            K_eff_bc = _apply_bc_sparse(K_eff, bc_dofs)
            du = spsolve(K_eff_bc, -R)
            u_trial += du

        if not converged:
            if verbose:
                print(f"  [BE] step {step}: Newton did not converge "
                      f"(||R||={R_norm:.3e}, {newton_max_iter} iters)")
            t_hist[step:] = np.nan
            u_hist[step:] = u_hist[step - 1]
            v_hist[step:] = 0.0
            ke_hist[step:] = 0.0
            break

        # Update velocity.
        v_new = (u_trial - u_curr) / dt
        v_new[bc_dofs] = 0.0
        u_trial[bc_dofs] = bc_vals

        u_curr = u_trial
        v_curr = v_new

        t_hist[step] = step * dt
        u_hist[step] = u_curr
        v_hist[step] = v_curr
        ke_hist[step] = 0.5 * v_curr.dot(M @ v_curr)

        if verbose and step % print_interval == 0:
            print(f"  [BE] step {step}/{n_steps}, t={step*dt:.4f}, "
                  f"KE={ke_hist[step]:.6e}, Newton iters={k+1}")

    return {"t": t_hist, "u_hist": u_hist, "v_hist": v_hist, "ke_hist": ke_hist}


# ---------------------------------------------------------------------------
# 2. Newmark-beta + Newton
# ---------------------------------------------------------------------------

def newmark_beta_nl(
    M: sparse.csr_matrix,
    f_ext: np.ndarray,
    assemble_fn,
    u0: np.ndarray,
    v0: np.ndarray,
    dt: float,
    n_steps: int,
    bc_dofs: np.ndarray,
    bc_vals: np.ndarray | None = None,
    beta: float = 0.25,
    gamma: float = 0.5,
    newton_max_iter: int = 20,
    newton_atol: float = 1e-8,
    verbose: bool = False,
) -> dict:
    """Implicit Newmark-beta time integration with Newton inner loop.

    Default parameters (beta=1/4, gamma=1/2): average acceleration method,
    unconditionally stable, second-order accurate, no algorithmic damping
    for linear problems.

    Newmark kinematic relations:
        u_{n+1} = u_pred + beta*dt^2*a_{n+1}
        v_{n+1} = v_pred + gamma*dt*a_{n+1}
    where:
        u_pred = u_n + dt*v_n + (0.5-beta)*dt^2*a_n
        v_pred = v_n + (1-gamma)*dt*a_n

    Newton residual (in terms of u_{n+1}):
        a_trial = (u_trial - u_pred) / (beta*dt^2)
        R(u) = M*a_trial + f_int(u) - f_ext

    Tangent:
        dR/du = M/(beta*dt^2) + K_T

    Args:
        M: (n, n) sparse mass matrix.
        f_ext: (n,) external force vector (constant).
        assemble_fn: Callable(u) -> (K_T_sparse, f_int_dense).
        u0: (n,) initial displacement.
        v0: (n,) initial velocity.
        dt: Time step size.
        n_steps: Number of time steps.
        bc_dofs: Indices of constrained DOFs.
        bc_vals: Prescribed displacement values (default: zeros).
        beta, gamma: Newmark parameters.
        newton_max_iter: Max Newton iterations per step.
        newton_atol: Absolute residual tolerance for Newton.
        verbose: Print progress.

    Returns:
        dict with same structure as backward_euler_nl.
    """
    n = len(u0)
    if bc_vals is None:
        bc_vals = np.zeros(len(bc_dofs))

    dt2 = dt * dt
    M_over_bdt2 = M / (beta * dt2)

    u_curr = u0.copy()
    v_curr = v0.copy()

    # Initial acceleration: M*a0 = f_ext - f_int(u0).
    _K0, f_int0 = assemble_fn(u0)
    rhs0 = f_ext - f_int0
    rhs0[bc_dofs] = 0.0
    M_bc = _apply_bc_sparse(M.copy(), bc_dofs)
    a_curr = spsolve(M_bc, rhs0)

    # Storage.
    t_hist = np.zeros(n_steps + 1)
    u_hist = np.zeros((n_steps + 1, n))
    v_hist = np.zeros((n_steps + 1, n))
    ke_hist = np.zeros(n_steps + 1)

    u_hist[0] = u0
    v_hist[0] = v0
    ke_hist[0] = 0.5 * v0.dot(M @ v0)

    print_interval = max(1, n_steps // 10)

    for step in range(1, n_steps + 1):
        # Predictors (used in the Newmark formula, not as Newton guess).
        u_pred = u_curr + dt * v_curr + (0.5 - beta) * dt2 * a_curr
        v_pred = v_curr + (1.0 - gamma) * dt * a_curr

        # Newton loop — start from u_curr for robustness.
        # Starting from u_pred can overshoot catastrophically when
        # the acceleration is large (common at release from large
        # deformation).  The Newmark relations (a_trial, corrector)
        # still use u_pred; only the initial guess changes.
        u_trial = u_curr.copy()
        converged = False

        for k in range(newton_max_iter):
            K_T, f_int = assemble_fn(u_trial)

            a_trial = (u_trial - u_pred) / (beta * dt2)
            R = M @ a_trial + f_int - f_ext
            R[bc_dofs] = 0.0

            R_norm = float(np.linalg.norm(R))
            if R_norm < newton_atol:
                converged = True
                break

            K_eff = M_over_bdt2 + K_T
            K_eff_bc = _apply_bc_sparse(K_eff, bc_dofs)
            du = spsolve(K_eff_bc, -R)
            u_trial += du

        if not converged:
            if verbose:
                print(f"  [NM] step {step}: Newton did not converge "
                      f"(||R||={R_norm:.3e}, {newton_max_iter} iters)")
            # Halt early — unconverged state will poison subsequent steps.
            t_hist[step:] = np.nan
            u_hist[step:] = u_hist[step - 1]
            v_hist[step:] = 0.0
            ke_hist[step:] = 0.0
            break

        # Corrector.
        a_new = (u_trial - u_pred) / (beta * dt2)
        v_new = v_pred + gamma * dt * a_new

        # Enforce BCs.
        u_trial[bc_dofs] = bc_vals
        v_new[bc_dofs] = 0.0
        a_new[bc_dofs] = 0.0

        u_curr = u_trial
        v_curr = v_new
        a_curr = a_new

        t_hist[step] = step * dt
        u_hist[step] = u_curr
        v_hist[step] = v_curr
        ke_hist[step] = 0.5 * v_curr.dot(M @ v_curr)

        if verbose and step % print_interval == 0:
            print(f"  [NM] step {step}/{n_steps}, t={step*dt:.4f}, "
                  f"KE={ke_hist[step]:.6e}, Newton iters={k+1}")

    return {"t": t_hist, "u_hist": u_hist, "v_hist": v_hist, "ke_hist": ke_hist}


# ---------------------------------------------------------------------------
# 3. Quasi-static Newton (load stepping, no inertia)
# ---------------------------------------------------------------------------

def quasi_static_nl(
    assemble_fn,
    n_dof: int,
    f_ext_start: np.ndarray,
    f_ext_end: np.ndarray,
    u0: np.ndarray,
    n_steps: int,
    bc_dofs: np.ndarray,
    bc_vals: np.ndarray,
    newton_max_iter: int = 25,
    newton_atol: float = 1e-8,
    verbose: bool = False,
) -> dict:
    """Quasi-static analysis via load stepping (no inertia).

    Solves a sequence of static equilibria as the external load is
    linearly ramped from ``f_ext_start`` to ``f_ext_end``.  Uses
    ``solve_newton_general`` at each load level.

    This produces the equilibrium path without any dynamic oscillation.

    Args:
        assemble_fn: Callable(u) -> (K_T_sparse, f_int_dense).
        n_dof: Number of degrees of freedom.
        f_ext_start: (n_dof,) external force at step 0.
        f_ext_end: (n_dof,) external force at final step.
        u0: (n_dof,) initial displacement (e.g., pre-bent shape).
        n_steps: Number of load steps.
        bc_dofs: Indices of constrained DOFs.
        bc_vals: Prescribed displacement values.
        newton_max_iter: Max Newton iterations per step.
        newton_atol: Absolute residual tolerance.
        verbose: Print progress.

    Returns:
        dict with keys:
            t: (n_steps+1,) pseudo-time array [0, 1].
            u_hist: (n_steps+1, n_dof) displacement history.
    """
    u_curr = u0.copy()

    t_hist = np.zeros(n_steps + 1)
    u_hist = np.zeros((n_steps + 1, n_dof))
    u_hist[0] = u0
    t_hist[0] = 0.0

    for step in range(1, n_steps + 1):
        frac = step / n_steps
        f_ext = (1.0 - frac) * f_ext_start + frac * f_ext_end

        result = solve_newton_general(
            n_dof, f_ext, bc_dofs, bc_vals,
            assemble_fn, u0=u_curr,
            max_iter=newton_max_iter, atol=newton_atol,
            verbose=False,
        )

        if not result["converged"] and verbose:
            print(f"  [QS] step {step}/{n_steps}: Newton did not converge "
                  f"(||R||={result['residual_norms'][-1]:.3e})")

        u_curr = result["u"]
        t_hist[step] = frac
        u_hist[step] = u_curr

        if verbose and step % max(1, n_steps // 10) == 0:
            print(f"  [QS] step {step}/{n_steps}, frac={frac:.2f}, "
                  f"Newton iters={result['iterations']}")

    return {"t": t_hist, "u_hist": u_hist}
