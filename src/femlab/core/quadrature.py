"""Gauss quadrature rules for reference simplex elements.

Reference triangle: vertices (0,0), (1,0), (0,1).  Area = 1/2.
Reference tetrahedron: vertices (0,0,0), (1,0,0), (0,1,0), (0,0,1).  Volume = 1/6.

Weights are scaled so that sum(weights) = reference domain measure.
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


def tetrahedron_quadrature(order: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Return quadrature points and weights for the reference tetrahedron.

    Args:
        order: Quadrature order.
            1 — 1-point rule, exact for polynomials up to degree 1.
            2 — 4-point rule, exact for polynomials up to degree 2.

    Returns:
        points: (nq, 3) array of quadrature points (xi, eta, zeta).
        weights: (nq,) array of quadrature weights.  sum = 1/6.
    """
    if order == 1:
        # Centroid rule.
        points = np.array([[0.25, 0.25, 0.25]])
        weights = np.array([1.0 / 6.0])
    elif order == 2:
        # 4-point rule (Hammer-Stroud).
        a = 0.1381966011250105  # (5 - sqrt(5)) / 20
        b = 0.5854101966249685  # (5 + 3*sqrt(5)) / 20
        points = np.array([
            [a, a, a],
            [b, a, a],
            [a, b, a],
            [a, a, b],
        ])
        weights = np.full(4, 1.0 / 24.0)
    else:
        raise ValueError(f"Unsupported quadrature order {order}. Use 1 or 2.")
    return points, weights
