"""Time integration schemes for structural dynamics.

Central difference (explicit):
    Conditionally stable, uses lumped (diagonal) mass for trivial inversion.
    Stability: dt < 2 / omega_max.

Newmark-beta (implicit):
    β = 1/4, γ = 1/2 (average acceleration) — unconditionally stable, 2nd order.
    Requires solving a linear system each step.
"""

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla


def central_difference(
    M_diag: np.ndarray,
    K: sparse.csr_matrix,
    f_ext: np.ndarray,
    u0: np.ndarray,
    v0: np.ndarray,
    dt: float,
    n_steps: int,
    bc_dofs: np.ndarray | None = None,
    verbose: bool = False,
) -> dict:
    """Explicit central difference time integration.

    Uses lumped (diagonal) mass for efficient inversion.

    u_{n+1} = 2 u_n - u_{n-1} + dt^2 M^{-1} (f_ext - K u_n)

    Args:
        M_diag: (n,) diagonal of the lumped mass matrix.
        K: (n, n) sparse stiffness matrix.
        f_ext: (n,) constant external force vector.
        u0: (n,) initial displacement.
        v0: (n,) initial velocity.
        dt: Time step size.
        n_steps: Number of time steps.
        bc_dofs: DOFs to constrain to zero throughout (optional).
        verbose: Print progress every 10% of steps.

    Returns:
        dict with keys:
            't': (n_steps+1,) time array.
            'u': (n_steps+1, n) displacement history.
            'v': (n_steps+1, n) velocity history.
            'energy': (n_steps+1,) total energy history.
    """
    n = len(u0)
    M_inv = 1.0 / M_diag

    # Bootstrap: u_{-1} = u0 - dt*v0 + 0.5*dt^2 * a0
    a0 = M_inv * (f_ext - K @ u0)
    if bc_dofs is not None:
        a0[bc_dofs] = 0.0
    u_prev = u0 - dt * v0 + 0.5 * dt * dt * a0
    u_curr = u0.copy()

    # Storage
    t_hist = np.zeros(n_steps + 1)
    u_hist = np.zeros((n_steps + 1, n))
    v_hist = np.zeros((n_steps + 1, n))
    energy_hist = np.zeros(n_steps + 1)

    u_hist[0] = u0
    v_hist[0] = v0
    energy_hist[0] = _total_energy(M_diag, K, u0, v0)

    print_interval = max(1, n_steps // 10)

    for step in range(1, n_steps + 1):
        a = M_inv * (f_ext - K @ u_curr)
        if bc_dofs is not None:
            a[bc_dofs] = 0.0

        u_next = 2.0 * u_curr - u_prev + dt * dt * a
        if bc_dofs is not None:
            u_next[bc_dofs] = 0.0

        v = (u_next - u_prev) / (2.0 * dt)

        u_prev = u_curr
        u_curr = u_next

        t_hist[step] = step * dt
        u_hist[step] = u_curr
        v_hist[step] = v
        energy_hist[step] = _total_energy(M_diag, K, u_curr, v)

        if verbose and step % print_interval == 0:
            print(f"  [central_diff] step {step}/{n_steps}, "
                  f"t={step*dt:.4f}, E={energy_hist[step]:.6e}")

    return {"t": t_hist, "u": u_hist, "v": v_hist, "energy": energy_hist}


def newmark_beta(
    M: sparse.csr_matrix,
    K: sparse.csr_matrix,
    f_ext: np.ndarray,
    u0: np.ndarray,
    v0: np.ndarray,
    dt: float,
    n_steps: int,
    beta: float = 0.25,
    gamma: float = 0.5,
    bc_dofs: np.ndarray | None = None,
    verbose: bool = False,
) -> dict:
    """Implicit Newmark-beta time integration.

    Default parameters (β=1/4, γ=1/2): average acceleration method,
    unconditionally stable, second-order accurate, no numerical damping.

    Args:
        M: (n, n) sparse mass matrix (consistent or lumped).
        K: (n, n) sparse stiffness matrix.
        f_ext: (n,) constant external force vector.
        u0: (n,) initial displacement.
        v0: (n,) initial velocity.
        dt: Time step size.
        n_steps: Number of time steps.
        beta, gamma: Newmark parameters.
        bc_dofs: DOFs to constrain to zero (optional).
        verbose: Print progress.

    Returns:
        dict with same structure as central_difference.
    """
    n = len(u0)

    # Effective stiffness: K_eff = M + beta*dt^2*K
    K_eff = M + beta * dt * dt * K

    # Apply BC to effective stiffness.
    if bc_dofs is not None and len(bc_dofs) > 0:
        K_eff = K_eff.tolil()
        for dof in bc_dofs:
            K_eff[dof, :] = 0.0
            K_eff[:, dof] = 0.0
            K_eff[dof, dof] = 1.0
        K_eff = K_eff.tocsr()

    # Factor the effective stiffness once (constant for linear problems).
    K_eff_lu = spla.splu(K_eff.tocsc())

    # Initial acceleration.
    rhs0 = f_ext - K @ u0
    if bc_dofs is not None and len(bc_dofs) > 0:
        # Solve M a0 = f_ext - K u0 with BCs
        M_bc = M.tolil()
        for dof in bc_dofs:
            M_bc[dof, :] = 0.0
            M_bc[:, dof] = 0.0
            M_bc[dof, dof] = 1.0
            rhs0[dof] = 0.0
        a0 = spla.spsolve(M_bc.tocsr(), rhs0)
    else:
        a0 = spla.spsolve(M, rhs0)

    u_curr = u0.copy()
    v_curr = v0.copy()
    a_curr = a0.copy()

    # Storage
    t_hist = np.zeros(n_steps + 1)
    u_hist = np.zeros((n_steps + 1, n))
    v_hist = np.zeros((n_steps + 1, n))
    energy_hist = np.zeros(n_steps + 1)

    u_hist[0] = u0
    v_hist[0] = v0
    # For energy with sparse consistent mass, use u^T M u form.
    energy_hist[0] = _total_energy_sparse(M, K, u0, v0)

    print_interval = max(1, n_steps // 10)

    for step in range(1, n_steps + 1):
        # Predictor (displacement and velocity without new acceleration).
        u_pred = u_curr + dt * v_curr + dt * dt * (0.5 - beta) * a_curr
        v_pred = v_curr + dt * (1.0 - gamma) * a_curr

        # Solve for new acceleration.
        rhs = f_ext - K @ u_pred
        if bc_dofs is not None and len(bc_dofs) > 0:
            rhs[bc_dofs] = 0.0
        a_next = K_eff_lu.solve(rhs)

        # Corrector.
        u_next = u_pred + beta * dt * dt * a_next
        v_next = v_pred + gamma * dt * a_next

        if bc_dofs is not None and len(bc_dofs) > 0:
            u_next[bc_dofs] = 0.0
            v_next[bc_dofs] = 0.0

        u_curr = u_next
        v_curr = v_next
        a_curr = a_next

        t_hist[step] = step * dt
        u_hist[step] = u_curr
        v_hist[step] = v_curr
        energy_hist[step] = _total_energy_sparse(M, K, u_curr, v_curr)

        if verbose and step % print_interval == 0:
            print(f"  [newmark] step {step}/{n_steps}, "
                  f"t={step*dt:.4f}, E={energy_hist[step]:.6e}")

    return {"t": t_hist, "u": u_hist, "v": v_hist, "energy": energy_hist}


def _total_energy(M_diag, K, u, v):
    """Total energy = 0.5 v^T M v + 0.5 u^T K u (lumped mass)."""
    KE = 0.5 * np.dot(M_diag * v, v)
    PE = 0.5 * np.dot(u, K @ u)
    return KE + PE


def _total_energy_sparse(M, K, u, v):
    """Total energy with sparse mass matrix."""
    KE = 0.5 * v.dot(M @ v)
    PE = 0.5 * u.dot(K @ u)
    return KE + PE
