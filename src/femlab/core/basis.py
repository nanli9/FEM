"""Shape functions and gradients for linear simplex elements.

T3 (3-node triangle, 2D):
    Reference triangle: (0,0), (1,0), (0,1)
    N0 = 1 - xi - eta,  N1 = xi,  N2 = eta

Tet4 (4-node tetrahedron, 3D):
    Reference tetrahedron: (0,0,0), (1,0,0), (0,1,0), (0,0,1)
    N0 = 1 - xi - eta - zeta,  N1 = xi,  N2 = eta,  N3 = zeta
"""

import numpy as np


def t3_shape(xi: float, eta: float) -> np.ndarray:
    """Evaluate T3 shape functions at a reference point.

    Args:
        xi: First barycentric coordinate.
        eta: Second barycentric coordinate.

    Returns:
        (3,) array of shape function values [N0, N1, N2].
    """
    return np.array([1.0 - xi - eta, xi, eta])


def t3_grad_ref() -> np.ndarray:
    """Gradients of T3 shape functions in reference coordinates.

    These are constant for the linear triangle.

    Returns:
        (2, 3) array where row 0 = dN/dxi, row 1 = dN/deta.
        Column j corresponds to node j.
    """
    # dN/dxi:  [-1, 1, 0]
    # dN/deta: [-1, 0, 1]
    return np.array([
        [-1.0, 1.0, 0.0],
        [-1.0, 0.0, 1.0],
    ])


def t3_jacobian(coords: np.ndarray) -> np.ndarray:
    """Compute the 2x2 Jacobian mapping reference → physical coordinates.

    J = dN_ref @ coords
      = [[x1-x0, y1-y0],
         [x2-x0, y2-y0]]

    Args:
        coords: (3, 2) array of element vertex coordinates [[x0,y0],[x1,y1],[x2,y2]].

    Returns:
        (2, 2) Jacobian matrix.
    """
    dN_ref = t3_grad_ref()  # (2, 3)
    return dN_ref @ coords  # (2, 2)


def t3_grad_phys(coords: np.ndarray) -> tuple[np.ndarray, float]:
    """Gradients of T3 shape functions in physical coordinates.

    Args:
        coords: (3, 2) array of element vertex coordinates.

    Returns:
        grad_phys: (2, 3) array where row 0 = dN/dx, row 1 = dN/dy.
        det_J: Determinant of the Jacobian (2 * element area).
    """
    J = t3_jacobian(coords)
    det_J = float(np.linalg.det(J))
    if det_J <= 0.0:
        raise ValueError(
            f"Non-positive Jacobian determinant ({det_J:.6e}). "
            "Check element orientation (must be counter-clockwise)."
        )
    J_inv = np.linalg.inv(J)  # (2, 2)
    dN_ref = t3_grad_ref()    # (2, 3)
    # dN_ref = J @ dN_phys  (chain rule), so dN_phys = J^{-1} @ dN_ref
    grad_phys = J_inv @ dN_ref  # (2, 3)
    return grad_phys, det_J


# ---------------------------------------------------------------------------
# Tet4 (4-node tetrahedron, 3D)
# ---------------------------------------------------------------------------

def tet4_shape(xi: float, eta: float, zeta: float) -> np.ndarray:
    """Evaluate Tet4 shape functions at a reference point.

    Args:
        xi, eta, zeta: Barycentric coordinates in the reference tet.

    Returns:
        (4,) array of shape function values [N0, N1, N2, N3].
    """
    return np.array([1.0 - xi - eta - zeta, xi, eta, zeta])


def tet4_grad_ref() -> np.ndarray:
    """Gradients of Tet4 shape functions in reference coordinates.

    Constant for the linear tetrahedron.

    Returns:
        (3, 4) array where row 0 = dN/dxi, row 1 = dN/deta, row 2 = dN/dzeta.
    """
    return np.array([
        [-1.0, 1.0, 0.0, 0.0],
        [-1.0, 0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0, 1.0],
    ])


def tet4_jacobian(coords: np.ndarray) -> np.ndarray:
    """Compute the 3x3 Jacobian mapping reference → physical coordinates.

    J = dN_ref @ coords  (3,4) @ (4,3) = (3,3)

    Args:
        coords: (4, 3) array of element vertex coordinates.

    Returns:
        (3, 3) Jacobian matrix.
    """
    dN_ref = tet4_grad_ref()
    return dN_ref @ coords


def tet4_grad_phys(coords: np.ndarray) -> tuple[np.ndarray, float]:
    """Gradients of Tet4 shape functions in physical coordinates.

    Args:
        coords: (4, 3) array of element vertex coordinates.

    Returns:
        grad_phys: (3, 4) array — rows are dN/dx, dN/dy, dN/dz.
        det_J: Determinant of the Jacobian (6 * element volume).
    """
    J = tet4_jacobian(coords)
    det_J = float(np.linalg.det(J))
    if det_J <= 0.0:
        raise ValueError(
            f"Non-positive Jacobian determinant ({det_J:.6e}). "
            "Check element orientation."
        )
    J_inv = np.linalg.inv(J)
    dN_ref = tet4_grad_ref()
    grad_phys = J_inv @ dN_ref  # (3, 4)
    return grad_phys, det_J
