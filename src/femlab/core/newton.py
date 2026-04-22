"""Newton-Raphson solver for nonlinear static FEM problems.

Iteratively solves the nonlinear equilibrium equation:
    f_int(u) = f_ext

using the tangent stiffness K_T = df_int/du to compute Newton updates:
    K_T du = -(f_int - f_ext)
    u  <-  u + du

Convergence is quadratic when the tangent is consistent.

Two entry points:
- ``solve_newton``: hardwired to nonlinear Tet4 assembly (Milestone 3 API).
- ``solve_newton_general``: accepts any assembly callable, used by the
  corotational solver and future formulations.
"""

import numpy as np
from scipy.sparse.linalg import spsolve

from femlab.core.assembly_nl import assemble_system_tet4_nl


# ---------------------------------------------------------------------------
# General Newton solver (assembly-function agnostic)
# ---------------------------------------------------------------------------

def solve_newton_general(
    n_dof: int,
    f_ext: np.ndarray,
    bc_dofs: np.ndarray,
    bc_vals: np.ndarray,
    assemble_fn,
    u0: np.ndarray | None = None,
    max_iter: int = 25,
    atol: float = 1e-8,
    rtol: float = 1e-8,
    verbose: bool = True,
) -> dict:
    """General Newton-Raphson solver with a user-supplied assembly function.

    Args:
        n_dof: Total number of degrees of freedom.
        f_ext: (n_dof,) external force vector.
        bc_dofs: Indices of constrained DOFs.
        bc_vals: Prescribed displacement values for constrained DOFs.
        assemble_fn: Callable(u) -> (K_T_sparse, f_int).
            Must return a sparse tangent stiffness and a dense internal
            force vector for the given displacement *u*.
        u0: Initial guess (default: zeros).
        max_iter: Maximum Newton iterations.
        atol: Absolute tolerance on residual norm.
        rtol: Relative tolerance (relative to first residual norm).
        verbose: Print iteration log.

    Returns:
        Dictionary with keys:
            u: (n_dof,) converged displacement.
            converged: bool.
            iterations: int.
            residual_norms: list of residual norms per iteration.
    """
    u = u0.copy() if u0 is not None else np.zeros(n_dof)
    u[bc_dofs] = bc_vals

    residual_norms = []
    R0_norm = None

    if verbose:
        print(f"  {'Iter':>4}  {'||R||':>12}  {'||R||/||R0||':>12}  {'||du||':>12}")
        print("  " + "-" * 48)

    for iteration in range(max_iter):
        K_T, f_int = assemble_fn(u)

        R = f_int - f_ext
        R[bc_dofs] = 0.0

        R_norm = float(np.linalg.norm(R))
        residual_norms.append(R_norm)

        if R0_norm is None:
            R0_norm = R_norm if R_norm > 0.0 else 1.0

        rel = R_norm / R0_norm

        if verbose and iteration == 0:
            print(f"  {iteration:4d}  {R_norm:12.6e}  {'—':>12}  {'—':>12}")

        if R_norm < max(atol, rtol * R0_norm):
            if verbose:
                print(f"  Converged in {iteration} iteration(s).")
            return {
                "u": u,
                "converged": True,
                "iterations": iteration,
                "residual_norms": residual_norms,
            }

        K_lil = K_T.tolil()
        for dof in bc_dofs:
            K_lil[dof, :] = 0.0
            K_lil[:, dof] = 0.0
            K_lil[dof, dof] = 1.0
        K_bc = K_lil.tocsr()

        du = spsolve(K_bc, -R)
        u += du

        du_norm = float(np.linalg.norm(du))
        if verbose:
            print(f"  {iteration + 1:4d}  {R_norm:12.6e}  {rel:12.6e}  {du_norm:12.6e}")

    if verbose:
        print(f"  WARNING: did not converge in {max_iter} iterations "
              f"(||R|| = {residual_norms[-1]:.6e}).")
    return {
        "u": u,
        "converged": False,
        "iterations": max_iter,
        "residual_norms": residual_norms,
    }


# ---------------------------------------------------------------------------
# Original Milestone-3 entry point (backward-compatible)
# ---------------------------------------------------------------------------

def solve_newton(
    nodes: np.ndarray,
    elements: np.ndarray,
    f_ext: np.ndarray,
    bc_dofs: np.ndarray,
    bc_vals: np.ndarray,
    tangent_fn,
    u0: np.ndarray | None = None,
    max_iter: int = 25,
    atol: float = 1e-8,
    rtol: float = 1e-8,
    verbose: bool = True,
) -> dict:
    """Solve a nonlinear static problem using Newton-Raphson iteration.

    Args:
        nodes: (N, 3) reference node coordinates.
        elements: (M, 4) element connectivity.
        f_ext: (3N,) external force vector.
        bc_dofs: Indices of constrained DOFs.
        bc_vals: Prescribed displacement values for constrained DOFs.
        tangent_fn: Callable F -> (P, A) where P is (3,3) PK1 stress
            and A is (9,9) material tangent.
        u0: Initial guess (default: zeros).
        max_iter: Maximum Newton iterations.
        atol: Absolute tolerance on residual norm.
        rtol: Relative tolerance (relative to first residual norm).
        verbose: Print iteration log.

    Returns:
        Dictionary with keys:
            u: (3N,) converged displacement.
            converged: bool.
            iterations: int.
            residual_norms: list of residual norms per iteration.
    """
    n_dof = 3 * len(nodes)
    u = u0.copy() if u0 is not None else np.zeros(n_dof)

    # Enforce prescribed displacements.
    u[bc_dofs] = bc_vals

    residual_norms = []
    R0_norm = None

    if verbose:
        print(f"  {'Iter':>4}  {'||R||':>12}  {'||R||/||R0||':>12}  {'||du||':>12}")
        print("  " + "-" * 48)

    for iteration in range(max_iter):
        # Assemble tangent stiffness and internal force in one pass.
        K_T, f_int = assemble_system_tet4_nl(nodes, elements, u, tangent_fn)

        # Residual: R = f_int - f_ext.
        R = f_int - f_ext
        R[bc_dofs] = 0.0  # constrained DOFs are satisfied

        R_norm = float(np.linalg.norm(R))
        residual_norms.append(R_norm)

        if R0_norm is None:
            R0_norm = R_norm if R_norm > 0.0 else 1.0

        rel = R_norm / R0_norm

        if verbose and iteration == 0:
            print(f"  {iteration:4d}  {R_norm:12.6e}  {'—':>12}  {'—':>12}")

        # Check convergence.
        if R_norm < max(atol, rtol * R0_norm):
            if verbose and iteration > 0:
                pass  # already printed
            if verbose:
                print(f"  Converged in {iteration} iteration(s).")
            return {
                "u": u,
                "converged": True,
                "iterations": iteration,
                "residual_norms": residual_norms,
            }

        # Apply BCs to tangent: zero rows/cols, diagonal = 1.
        K_lil = K_T.tolil()
        for dof in bc_dofs:
            K_lil[dof, :] = 0.0
            K_lil[:, dof] = 0.0
            K_lil[dof, dof] = 1.0
        K_bc = K_lil.tocsr()

        # Solve for Newton update.
        du = spsolve(K_bc, -R)
        u += du

        du_norm = float(np.linalg.norm(du))
        if verbose:
            if iteration > 0:
                # Print the previous iteration's info was already done above
                pass
            print(f"  {iteration + 1:4d}  {R_norm:12.6e}  {rel:12.6e}  {du_norm:12.6e}")

    if verbose:
        print(f"  WARNING: did not converge in {max_iter} iterations "
              f"(||R|| = {residual_norms[-1]:.6e}).")
    return {
        "u": u,
        "converged": False,
        "iterations": max_iter,
        "residual_norms": residual_norms,
    }
