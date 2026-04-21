"""2D constitutive (material stiffness) matrices.

Supports plane stress and plane strain for linear isotropic elasticity.
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
