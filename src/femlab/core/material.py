"""Constitutive (material stiffness) matrices for linear isotropic elasticity.

2D: plane stress and plane strain (3×3).
3D: full isotropic elasticity (6×6).
"""

import numpy as np


def plane_stress_D(E: float, nu: float) -> np.ndarray:
    """Plane stress constitutive matrix (3x3).

    Relates stress {σxx, σyy, τxy} to strain {εxx, εyy, γxy}.

    Args:
        E: Young's modulus.
        nu: Poisson's ratio.

    Returns:
        (3, 3) symmetric positive-definite matrix.
    """
    c = E / (1.0 - nu * nu)
    return c * np.array([
        [1.0, nu, 0.0],
        [nu, 1.0, 0.0],
        [0.0, 0.0, (1.0 - nu) / 2.0],
    ])


def plane_strain_D(E: float, nu: float) -> np.ndarray:
    """Plane strain constitutive matrix (3x3).

    Relates stress {σxx, σyy, τxy} to strain {εxx, εyy, γxy}.

    Args:
        E: Young's modulus.
        nu: Poisson's ratio.

    Returns:
        (3, 3) symmetric positive-definite matrix.
    """
    c = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return c * np.array([
        [1.0 - nu, nu, 0.0],
        [nu, 1.0 - nu, 0.0],
        [0.0, 0.0, (1.0 - 2.0 * nu) / 2.0],
    ])


def isotropic_3d_D(E: float, nu: float) -> np.ndarray:
    """3D isotropic constitutive matrix (6×6).

    Voigt ordering: {σxx, σyy, σzz, τxy, τyz, τxz}.

    Args:
        E: Young's modulus.
        nu: Poisson's ratio.

    Returns:
        (6, 6) symmetric positive-definite matrix.
    """
    c = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    G = (1.0 - 2.0 * nu) / 2.0
    return c * np.array([
        [1.0 - nu, nu,       nu,       0.0, 0.0, 0.0],
        [nu,       1.0 - nu, nu,       0.0, 0.0, 0.0],
        [nu,       nu,       1.0 - nu, 0.0, 0.0, 0.0],
        [0.0,      0.0,      0.0,      G,   0.0, 0.0],
        [0.0,      0.0,      0.0,      0.0, G,   0.0],
        [0.0,      0.0,      0.0,      0.0, 0.0, G  ],
    ])
