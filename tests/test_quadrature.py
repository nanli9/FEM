"""Unit tests for triangle quadrature rules."""

import numpy as np
import pytest

from femlab.core.quadrature import triangle_quadrature


class TestTriangleQuadrature:
    """Tests for Gauss quadrature on the reference triangle."""

    def test_order1_weight_sum(self):
        """1-point rule weights must sum to reference triangle area = 1/2.

        Regression tolerance: exact (1e-15).
        """
        pts, wts = triangle_quadrature(1)
        np.testing.assert_allclose(wts.sum(), 0.5, atol=1e-15)

    def test_order2_weight_sum(self):
        """3-point rule weights must sum to reference triangle area = 1/2.

        Regression tolerance: exact (1e-15).
        """
        pts, wts = triangle_quadrature(2)
        np.testing.assert_allclose(wts.sum(), 0.5, atol=1e-15)

    def test_order1_constant(self):
        """1-point rule must integrate f=1 over reference triangle exactly."""
        pts, wts = triangle_quadrature(1)
        integral = sum(1.0 * w for w in wts)
        np.testing.assert_allclose(integral, 0.5, atol=1e-15)

    def test_order1_linear(self):
        """1-point rule must integrate f=xi exactly (∫xi dA = 1/6).

        Regression tolerance: 1e-15.
        """
        pts, wts = triangle_quadrature(1)
        integral = sum(p[0] * w for p, w in zip(pts, wts))
        np.testing.assert_allclose(integral, 1.0 / 6.0, atol=1e-15)

    def test_order2_quadratic(self):
        """3-point rule must integrate f=xi^2 exactly (∫xi^2 dA = 1/12).

        Regression tolerance: 1e-15.
        """
        pts, wts = triangle_quadrature(2)
        integral = sum(p[0] ** 2 * w for p, w in zip(pts, wts))
        np.testing.assert_allclose(integral, 1.0 / 12.0, atol=1e-15)

    def test_order2_mixed(self):
        """3-point rule must integrate f=xi*eta exactly (∫xi*eta dA = 1/24).

        Regression tolerance: 1e-15.
        """
        pts, wts = triangle_quadrature(2)
        integral = sum(p[0] * p[1] * w for p, w in zip(pts, wts))
        np.testing.assert_allclose(integral, 1.0 / 24.0, atol=1e-15)

    def test_points_inside_triangle(self):
        """All quadrature points must lie inside the reference triangle."""
        for order in [1, 2]:
            pts, _ = triangle_quadrature(order)
            for p in pts:
                xi, eta = p
                assert xi >= 0.0 and eta >= 0.0 and xi + eta <= 1.0 + 1e-15

    def test_unsupported_order(self):
        with pytest.raises(ValueError, match="Unsupported"):
            triangle_quadrature(5)
