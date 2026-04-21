"""Gauss quadrature rules for the reference triangle.

Reference triangle vertices: (0,0), (1,0), (0,1).
Area of reference triangle = 1/2.

Weights are scaled so that: sum(weights) = area = 1/2.
"""

import numpy as np


def triangle_quadrature(order: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Return quadrature points and weights for the reference triangle.

    Args:
        order: Quadrature order.
            1 — 1-point rule, exact for polynomials up to degree 1.
            2 — 3-point rule, exact for polynomials up to degree 2.

    Returns:
        points: (nq, 2) array of quadrature points (xi, eta).
        weights: (nq,) array of quadrature weights.
    """
    if order == 1:
        # Centroid rule: exact for linear polynomials.
        points = np.array([[1.0 / 3.0, 1.0 / 3.0]])
        weights = np.array([0.5])
    elif order == 2:
        # 3-point midpoint rule: exact for quadratic polynomials.
        points = np.array([
            [1.0 / 6.0, 1.0 / 6.0],
            [2.0 / 3.0, 1.0 / 6.0],
            [1.0 / 6.0, 2.0 / 3.0],
        ])
        weights = np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    else:
        raise ValueError(f"Unsupported quadrature order {order}. Use 1 or 2.")
    return points, weights
