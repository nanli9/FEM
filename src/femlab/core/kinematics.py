"""Kinematic quantities for finite deformation analysis.

Deformation gradient F = dx/dX maps infinitesimal material line elements
from the reference configuration to the current (deformed) configuration.

For a Tet4 element, F is constant (linear displacement interpolation)
and is computed directly from the nodal displacements and reference
shape function gradients.
"""

import numpy as np

from femlab.core.basis import tet4_grad_ref


def deformation_gradient_tet4(
    X_ref: np.ndarray,
    u_e: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute the deformation gradient F for a Tet4 element.

    F = I + du/dX = I + u_nodes^T @ dN_dX^T

    where dN_dX = J0^{-1} @ dN_ref are the shape function gradients
    with respect to the reference (material) coordinates.

    Args:
        X_ref: (4, 3) reference node coordinates.
        u_e: (12,) element displacement vector [u0,v0,w0,...,u3,v3,w3].

    Returns:
        F: (3, 3) deformation gradient.
        dN_dX: (3, 4) shape function gradients w.r.t. reference coords.
        V_ref: Reference element volume.
    """
    dN_ref = tet4_grad_ref()       # (3, 4)
    J0 = dN_ref @ X_ref           # (3, 3) reference Jacobian
    det_J0 = float(np.linalg.det(J0))
    if det_J0 <= 0.0:
        raise ValueError(
            f"Non-positive reference Jacobian determinant ({det_J0:.6e}). "
            "Check element orientation."
        )
    J0_inv = np.linalg.inv(J0)
    dN_dX = J0_inv @ dN_ref       # (3, 4)
    V_ref = det_J0 / 6.0

    u_nodes = u_e.reshape(4, 3)   # (4, 3)
    F = np.eye(3) + u_nodes.T @ dN_dX.T  # (3, 3)

    return F, dN_dX, V_ref


def tet4_G_matrix(dN_dX: np.ndarray) -> np.ndarray:
    """Build the 9x12 displacement-gradient operator G for a Tet4 element.

    Maps element DOFs to vectorized deformation gradient increment:
        vec(F - I) = G @ u_e

    where vec(P)[3*i + J] = P_{iJ} (row-major, matches numpy .ravel()).

    This enables compact force and stiffness expressions:
        f_int = G^T @ vec(P) * V_ref
        K_T   = G^T @ A @ G * V_ref

    Args:
        dN_dX: (3, 4) shape function gradients w.r.t. reference coords.

    Returns:
        G: (9, 12) gradient operator matrix.
    """
    G = np.zeros((9, 12))
    for a in range(4):
        for i in range(3):
            for J in range(3):
                G[3 * i + J, 3 * a + i] = dN_dX[J, a]
    return G
