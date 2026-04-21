"""Dirichlet boundary condition application via elimination.

This module modifies the global stiffness matrix and force vector
to enforce prescribed displacements.  The elimination method zeros
out the constrained rows/columns and places a 1 on the diagonal,
adjusting the RHS accordingly.
"""

import numpy as np
from scipy import sparse


def apply_dirichlet(
    K: sparse.csr_matrix,
    f: np.ndarray,
    bc_dofs: np.ndarray,
    bc_vals: np.ndarray,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Apply Dirichlet BCs by elimination (modify K and f in place semantics).

    For each constrained DOF i with prescribed value g_i:
        1. f -= K[:, i] * g_i   (move known contribution to RHS)
        2. Zero row i and column i of K
        3. Set K[i, i] = 1, f[i] = g_i

    Args:
        K: (n, n) sparse stiffness matrix (CSR). Will be converted internally.
        f: (n,) force vector (modified in place).
        bc_dofs: 1-D array of constrained DOF indices.
        bc_vals: 1-D array of prescribed displacement values.

    Returns:
        K_mod: Modified stiffness matrix (CSR).
        f_mod: Modified force vector.
    """
    bc_dofs = np.asarray(bc_dofs, dtype=np.int64)
    bc_vals = np.asarray(bc_vals, dtype=np.float64)

    # Work in LIL format for efficient row/column modification.
    K_lil = K.tolil()
    f_mod = f.copy()

    # Step 1: Move all known contributions to RHS using the original K.
    for dof, val in zip(bc_dofs, bc_vals):
        col = np.array(K.getcol(dof).todense()).ravel()
        f_mod -= col * val

    # Step 2: Zero constrained rows/columns, set diagonal to 1.
    for dof in bc_dofs:
        K_lil[dof, :] = 0.0
        K_lil[:, dof] = 0.0
        K_lil[dof, dof] = 1.0

    # Step 3: Set prescribed values AFTER all RHS adjustments
    # to avoid cross-corruption between constrained DOFs.
    for dof, val in zip(bc_dofs, bc_vals):
        f_mod[dof] = val

    return K_lil.tocsr(), f_mod
