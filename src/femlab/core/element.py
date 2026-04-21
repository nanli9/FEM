"""Element routines: B-matrix, stiffness, residual, and stress recovery.

T3: 2D, 3-node linear triangle (plane stress / plane strain).
Tet4: 3D, 4-node linear tetrahedron.
"""

import numpy as np

from femlab.core.basis import t3_grad_phys, tet4_grad_phys
from femlab.core.quadrature import triangle_quadrature, tetrahedron_quadrature


def t3_B_matrix(grad_phys: np.ndarray) -> np.ndarray:
    """Build the 3×6 strain-displacement (B) matrix for a T3 element.

    B maps the element DOF vector u_e = [u0,v0, u1,v1, u2,v2]
    to the Voigt strain vector ε = [εxx, εyy, γxy].

    Args:
        grad_phys: (2, 3) array of shape function gradients in physical coords.
                   Row 0 = dN/dx, row 1 = dN/dy.

    Returns:
        (3, 6) B matrix.
    """
    dNdx = grad_phys[0]  # (3,)
    dNdy = grad_phys[1]  # (3,)
    B = np.zeros((3, 6))
    for i in range(3):
        B[0, 2 * i] = dNdx[i]       # εxx = du/dx
        B[1, 2 * i + 1] = dNdy[i]   # εyy = dv/dy
        B[2, 2 * i] = dNdy[i]       # γxy = du/dy + dv/dx
        B[2, 2 * i + 1] = dNdx[i]
    return B


def t3_element_stiffness(
    coords: np.ndarray,
    D: np.ndarray,
    thickness: float = 1.0,
    quad_order: int = 1,
) -> np.ndarray:
    """Compute the 6×6 element stiffness matrix for a T3 element.

    k_e = thickness * ∫ B^T D B dA
        = thickness * Σ_q (B^T D B * det(J) * w_q)

    For the linear T3 element B is constant, so a 1-point rule is exact.

    Args:
        coords: (3, 2) element vertex coordinates (counter-clockwise).
        D: (3, 3) constitutive matrix.
        thickness: Out-of-plane thickness (default 1.0).
        quad_order: Quadrature order (1 or 2).

    Returns:
        (6, 6) symmetric element stiffness matrix.
    """
    qpts, qwts = triangle_quadrature(quad_order)
    ke = np.zeros((6, 6))
    for qp, qw in zip(qpts, qwts):
        grad_phys, det_J = t3_grad_phys(coords)
        B = t3_B_matrix(grad_phys)
        ke += (B.T @ D @ B) * det_J * qw * thickness
    return ke


def t3_element_residual(
    coords: np.ndarray,
    D: np.ndarray,
    u_e: np.ndarray,
    thickness: float = 1.0,
    quad_order: int = 1,
) -> np.ndarray:
    """Compute the 6-vector internal force (residual) for a T3 element.

    f_int = thickness * ∫ B^T σ dA  where σ = D ε = D B u_e

    Args:
        coords: (3, 2) element vertex coordinates.
        D: (3, 3) constitutive matrix.
        u_e: (6,) element displacement vector [u0,v0,u1,v1,u2,v2].
        thickness: Out-of-plane thickness.
        quad_order: Quadrature order.

    Returns:
        (6,) element internal force vector.
    """
    qpts, qwts = triangle_quadrature(quad_order)
    fe = np.zeros(6)
    for qp, qw in zip(qpts, qwts):
        grad_phys, det_J = t3_grad_phys(coords)
        B = t3_B_matrix(grad_phys)
        strain = B @ u_e
        stress = D @ strain
        fe += (B.T @ stress) * det_J * qw * thickness
    return fe


def t3_stress(
    coords: np.ndarray,
    D: np.ndarray,
    u_e: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute constant strain and stress for a T3 element.

    Args:
        coords: (3, 2) element vertex coordinates.
        D: (3, 3) constitutive matrix.
        u_e: (6,) element displacement vector.

    Returns:
        strain: (3,) Voigt strain [εxx, εyy, γxy].
        stress: (3,) Voigt stress [σxx, σyy, τxy].
    """
    grad_phys, _ = t3_grad_phys(coords)
    B = t3_B_matrix(grad_phys)
    strain = B @ u_e
    stress = D @ strain
    return strain, stress


# ---------------------------------------------------------------------------
# Tet4 (4-node tetrahedron, 3D)
# ---------------------------------------------------------------------------

def tet4_B_matrix(grad_phys: np.ndarray) -> np.ndarray:
    """Build the 6×12 strain-displacement (B) matrix for a Tet4 element.

    DOF ordering: u_e = [u0,v0,w0, u1,v1,w1, u2,v2,w2, u3,v3,w3].
    Voigt strain: ε = [εxx, εyy, εzz, γxy, γyz, γxz].

    Args:
        grad_phys: (3, 4) — rows are dN/dx, dN/dy, dN/dz.

    Returns:
        (6, 12) B matrix.
    """
    dNdx = grad_phys[0]  # (4,)
    dNdy = grad_phys[1]
    dNdz = grad_phys[2]
    B = np.zeros((6, 12))
    for i in range(4):
        c = 3 * i
        B[0, c]     = dNdx[i]             # εxx
        B[1, c + 1] = dNdy[i]             # εyy
        B[2, c + 2] = dNdz[i]             # εzz
        B[3, c]     = dNdy[i]             # γxy
        B[3, c + 1] = dNdx[i]
        B[4, c + 1] = dNdz[i]             # γyz
        B[4, c + 2] = dNdy[i]
        B[5, c]     = dNdz[i]             # γxz
        B[5, c + 2] = dNdx[i]
    return B


def tet4_element_stiffness(
    coords: np.ndarray,
    D: np.ndarray,
    quad_order: int = 1,
) -> np.ndarray:
    """Compute the 12×12 element stiffness matrix for a Tet4 element.

    k_e = ∫ B^T D B dV = Σ_q (B^T D B * det(J) * w_q)

    For Tet4, B is constant ⇒ 1-point rule is exact.

    Args:
        coords: (4, 3) element vertex coordinates.
        D: (6, 6) constitutive matrix.
        quad_order: Quadrature order (1 or 2).

    Returns:
        (12, 12) symmetric element stiffness matrix.
    """
    qpts, qwts = tetrahedron_quadrature(quad_order)
    ke = np.zeros((12, 12))
    for qp, qw in zip(qpts, qwts):
        grad_phys, det_J = tet4_grad_phys(coords)
        B = tet4_B_matrix(grad_phys)
        ke += (B.T @ D @ B) * det_J * qw
    return ke


def tet4_element_residual(
    coords: np.ndarray,
    D: np.ndarray,
    u_e: np.ndarray,
    quad_order: int = 1,
) -> np.ndarray:
    """Compute the 12-vector internal force for a Tet4 element.

    Args:
        coords: (4, 3) element vertex coordinates.
        D: (6, 6) constitutive matrix.
        u_e: (12,) element DOF vector.
        quad_order: Quadrature order.

    Returns:
        (12,) element internal force vector.
    """
    qpts, qwts = tetrahedron_quadrature(quad_order)
    fe = np.zeros(12)
    for qp, qw in zip(qpts, qwts):
        grad_phys, det_J = tet4_grad_phys(coords)
        B = tet4_B_matrix(grad_phys)
        stress = D @ (B @ u_e)
        fe += (B.T @ stress) * det_J * qw
    return fe


def tet4_stress(
    coords: np.ndarray,
    D: np.ndarray,
    u_e: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute constant strain and stress for a Tet4 element.

    Args:
        coords: (4, 3) element vertex coordinates.
        D: (6, 6) constitutive matrix.
        u_e: (12,) element DOF vector.

    Returns:
        strain: (6,) Voigt strain.
        stress: (6,) Voigt stress.
    """
    grad_phys, _ = tet4_grad_phys(coords)
    B = tet4_B_matrix(grad_phys)
    strain = B @ u_e
    stress = D @ strain
    return strain, stress
