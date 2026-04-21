"""Nonlinear element routines for Tet4 under finite deformation.

Computes internal force and tangent stiffness using the deformation
gradient F and a hyperelastic material model (total Lagrangian formulation).

f_int = G^T @ vec(P) * V_ref
K_T   = G^T @ A @ G * V_ref

where G (9x12) maps DOFs to vec(F-I), P is the first Piola-Kirchhoff
stress, and A = dP/dF is the material tangent (9x9).
"""

import numpy as np

from femlab.core.kinematics import deformation_gradient_tet4, tet4_G_matrix


def tet4_internal_force_nl(
    X_ref: np.ndarray,
    u_e: np.ndarray,
    pk1_fn,
) -> np.ndarray:
    """Compute the internal force vector for a Tet4 element (nonlinear).

    Args:
        X_ref: (4, 3) reference node coordinates.
        u_e: (12,) element displacement vector.
        pk1_fn: Callable F -> P (3x3 first Piola-Kirchhoff stress).

    Returns:
        f_int: (12,) element internal force vector.
    """
    F, dN_dX, V_ref = deformation_gradient_tet4(X_ref, u_e)
    P = pk1_fn(F)
    G = tet4_G_matrix(dN_dX)
    return G.T @ P.ravel() * V_ref


def tet4_tangent_stiffness_nl(
    X_ref: np.ndarray,
    u_e: np.ndarray,
    tangent_fn,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute tangent stiffness and internal force for a Tet4 element.

    Returns both to avoid redundant deformation gradient computation.

    Args:
        X_ref: (4, 3) reference node coordinates.
        u_e: (12,) element displacement vector.
        tangent_fn: Callable F -> (P, A) where P is (3,3) and A is (9,9).

    Returns:
        K_T: (12, 12) element tangent stiffness matrix.
        f_int: (12,) element internal force vector.
    """
    F, dN_dX, V_ref = deformation_gradient_tet4(X_ref, u_e)
    P, A = tangent_fn(F)
    G = tet4_G_matrix(dN_dX)
    K_T = G.T @ A @ G * V_ref
    f_int = G.T @ P.ravel() * V_ref
    return K_T, f_int
