"""Corotational FEM for Tet4 elements.

Extracts the rigid body rotation from each element's deformation gradient
via polar decomposition F = R·U, applies linear stiffness in the rotated
(corotational) frame, and transforms back to the global frame.

This approach handles large rotations exactly while using a simple linear
constitutive law.  It is less expensive than a full nonlinear formulation
(no hyperelastic material tangent needed) but does not capture large-strain
effects (nonlinear material response).

Element internal force
----------------------
1. F = I + du/dX  →  R, U = polar(F)
2. u_local_a = R^T · (X_a + u_a) - X_a   (rotation-free displacement)
3. f_local = K_linear · u_local            (linear force in rotated frame)
4. f = T_R · f_local                       (rotate back to global frame)

Element tangent stiffness
-------------------------
K_cr = T_R · K_linear · T_R^T  +  K_σ

where K_σ is the geometric (initial-stress) stiffness that accounts for
the change of rotation R with displacement.
"""

import numpy as np
from scipy.linalg import polar as scipy_polar

from femlab.core.kinematics import deformation_gradient_tet4
from femlab.core.element import tet4_element_stiffness, tet4_B_matrix


def polar_decomposition_tet4(
    X_ref: np.ndarray,
    u_e: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute polar decomposition F = R·U for a Tet4 element.

    Args:
        X_ref: (4, 3) reference node coordinates.
        u_e: (12,) element displacement vector.

    Returns:
        R: (3, 3) rotation matrix from polar decomposition.
        U: (3, 3) right stretch tensor (symmetric positive definite).
        dN_dX: (3, 4) shape function gradients w.r.t. reference coords.
        V_ref: Reference element volume.
    """
    F, dN_dX, V_ref = deformation_gradient_tet4(X_ref, u_e)
    R, U = scipy_polar(F, side='right')  # F = R @ U
    return R, U, dN_dX, V_ref


def _block_rotation(R: np.ndarray) -> np.ndarray:
    """Build 12×12 block-diagonal rotation T_R = blkdiag(R, R, R, R).

    Transforms element force / displacement vectors between global and
    corotational (local) frames:  f_global = T_R @ f_local.
    """
    T = np.zeros((12, 12))
    for a in range(4):
        T[3 * a:3 * a + 3, 3 * a:3 * a + 3] = R
    return T


def _corotational_local_displacement(
    X_ref: np.ndarray,
    u_e: np.ndarray,
    R: np.ndarray,
) -> np.ndarray:
    """Compute rotation-free (local) displacement for each node.

    u_local_a = R^T @ (X_a + u_a) - X_a

    This removes the rigid body rotation from the displacement,
    leaving only the deformation that the linear element "sees".
    """
    u_nodes = u_e.reshape(4, 3)
    x_nodes = X_ref + u_nodes  # current positions
    u_local = np.zeros(12)
    for a in range(4):
        u_local[3 * a:3 * a + 3] = R.T @ x_nodes[a] - X_ref[a]
    return u_local


def tet4_internal_force_cr(
    X_ref: np.ndarray,
    u_e: np.ndarray,
    D: np.ndarray,
) -> np.ndarray:
    """Compute corotational internal force for a Tet4 element.

    Args:
        X_ref: (4, 3) reference node coordinates.
        u_e: (12,) element displacement vector.
        D: (6, 6) linear isotropic constitutive matrix.

    Returns:
        f_int: (12,) element internal force vector in global frame.
    """
    F, dN_dX, V_ref = deformation_gradient_tet4(X_ref, u_e)
    R, _U = scipy_polar(F, side='right')

    T_R = _block_rotation(R)
    K_lin = tet4_element_stiffness(X_ref, D)
    u_local = _corotational_local_displacement(X_ref, u_e, R)

    f_local = K_lin @ u_local
    return T_R @ f_local


def _geometric_stiffness(
    dN_dX: np.ndarray,
    V_ref: float,
    sigma_voigt: np.ndarray,
) -> np.ndarray:
    """Compute the geometric (initial-stress) stiffness for a Tet4.

    K_σ[3a+i, 3b+j] = δ_{ij} · (Σ_{M,N} σ_{MN} · dN_a/dX_M · dN_b/dX_N) · V_ref

    This term accounts for the change in rotation R with displacement
    and is essential for quadratic convergence in Newton iteration.

    Args:
        dN_dX: (3, 4) shape function gradients w.r.t. reference coords.
        V_ref: Reference element volume.
        sigma_voigt: (6,) Voigt stress [σxx, σyy, σzz, τxy, τyz, τxz].

    Returns:
        K_sigma: (12, 12) geometric stiffness matrix.
    """
    # Unpack Voigt to 3×3 symmetric stress tensor.
    sigma_mat = np.array([
        [sigma_voigt[0], sigma_voigt[3], sigma_voigt[5]],
        [sigma_voigt[3], sigma_voigt[1], sigma_voigt[4]],
        [sigma_voigt[5], sigma_voigt[4], sigma_voigt[2]],
    ])

    # S_scalar[a, b] = Σ_{M,N} σ_{MN} · dN_a/dX_M · dN_b/dX_N · V_ref
    S_scalar = (dN_dX.T @ sigma_mat @ dN_dX) * V_ref  # (4, 4)

    # K_σ = S_scalar ⊗ I_3  (Kronecker product).
    return np.kron(S_scalar, np.eye(3))


def tet4_tangent_stiffness_cr(
    X_ref: np.ndarray,
    u_e: np.ndarray,
    D: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute corotational tangent stiffness and internal force for a Tet4.

    K_cr = T_R · K_linear · T_R^T  +  K_σ

    The tangent has two parts:
    1. Rotated linear stiffness  T_R K_lin T_R^T :  material response in
       the current rotated frame.
    2. Geometric stiffness  K_σ :  accounts for the change of rotation R
       with displacement (initial-stress / stress-stiffening effect).

    Args:
        X_ref: (4, 3) reference node coordinates.
        u_e: (12,) element displacement vector.
        D: (6, 6) linear isotropic constitutive matrix.

    Returns:
        K_cr: (12, 12) corotational tangent stiffness matrix.
        f_int: (12,) corotational internal force vector.
    """
    F, dN_dX, V_ref = deformation_gradient_tet4(X_ref, u_e)
    R, _U = scipy_polar(F, side='right')

    T_R = _block_rotation(R)
    K_lin = tet4_element_stiffness(X_ref, D)
    u_local = _corotational_local_displacement(X_ref, u_e, R)

    # Internal force: linear force in local frame, rotated to global.
    f_local = K_lin @ u_local
    f_int = T_R @ f_local

    # Rotated material stiffness.
    K_rot = T_R @ K_lin @ T_R.T

    # Geometric stiffness from local stress state.
    B = tet4_B_matrix(dN_dX)
    strain_local = B @ u_local
    sigma_local = D @ strain_local
    K_sigma = _geometric_stiffness(dN_dX, V_ref, sigma_local)

    return K_rot + K_sigma, f_int
