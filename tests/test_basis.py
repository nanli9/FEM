"""Unit tests for T3 shape functions and gradients."""

import numpy as np
import pytest

from femlab.core.basis import t3_shape, t3_grad_ref, t3_grad_phys, t3_jacobian


class TestT3Shape:
    """Tests for T3 shape function evaluation."""

    def test_partition_of_unity(self):
        """Shape functions must sum to 1 at any point in the reference triangle."""
        # Regression tolerance: 1e-15 (machine precision for linear functions).
        points = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0),
                  (1.0 / 3.0, 1.0 / 3.0), (0.25, 0.25)]
        for xi, eta in points:
            N = t3_shape(xi, eta)
            assert N.shape == (3,)
            np.testing.assert_allclose(N.sum(), 1.0, atol=1e-15)

    def test_kronecker_property(self):
        """N_i(node_j) = delta_{ij}."""
        ref_nodes = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
        for j, (xi, eta) in enumerate(ref_nodes):
            N = t3_shape(xi, eta)
            for i in range(3):
                expected = 1.0 if i == j else 0.0
                assert abs(N[i] - expected) < 1e-15

    def test_centroid(self):
        """At centroid (1/3, 1/3), all shape functions equal 1/3."""
        N = t3_shape(1.0 / 3.0, 1.0 / 3.0)
        np.testing.assert_allclose(N, [1.0 / 3.0] * 3, atol=1e-15)


class TestT3GradRef:
    """Tests for reference-coordinate gradients."""

    def test_shape(self):
        grad = t3_grad_ref()
        assert grad.shape == (2, 3)

    def test_values(self):
        grad = t3_grad_ref()
        expected = np.array([[-1.0, 1.0, 0.0],
                             [-1.0, 0.0, 1.0]])
        np.testing.assert_array_equal(grad, expected)


class TestT3Jacobian:
    """Tests for Jacobian computation."""

    def test_unit_reference_triangle(self):
        """Jacobian of the reference triangle itself should be identity."""
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        J = t3_jacobian(coords)
        np.testing.assert_allclose(J, np.eye(2), atol=1e-15)

    def test_scaled_triangle(self):
        """A triangle scaled by factor s should have J = s*I, det = s^2."""
        s = 3.0
        coords = np.array([[0.0, 0.0], [s, 0.0], [0.0, s]])
        J = t3_jacobian(coords)
        np.testing.assert_allclose(J, s * np.eye(2), atol=1e-14)
        np.testing.assert_allclose(np.linalg.det(J), s * s, atol=1e-14)

    def test_translated_triangle(self):
        """Translation should not affect the Jacobian (it depends on differences)."""
        offset = np.array([5.0, 7.0])
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]) + offset
        J = t3_jacobian(coords)
        np.testing.assert_allclose(J, np.eye(2), atol=1e-14)


class TestT3GradPhys:
    """Tests for physical-coordinate gradients."""

    def test_reference_triangle(self):
        """On the reference triangle, physical gradients equal reference gradients."""
        # Regression tolerance: 1e-14.
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        grad, det_J = t3_grad_phys(coords)
        expected = t3_grad_ref()
        np.testing.assert_allclose(grad, expected, atol=1e-14)
        np.testing.assert_allclose(det_J, 1.0, atol=1e-14)

    def test_scaled_triangle(self):
        """Gradients should scale inversely with element size."""
        s = 2.0
        coords = np.array([[0.0, 0.0], [s, 0.0], [0.0, s]])
        grad, det_J = t3_grad_phys(coords)
        np.testing.assert_allclose(det_J, s * s, atol=1e-14)
        expected = t3_grad_ref() / s
        np.testing.assert_allclose(grad, expected, atol=1e-14)

    def test_clockwise_raises(self):
        """Clockwise-oriented element should raise ValueError."""
        coords = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        with pytest.raises(ValueError, match="Non-positive Jacobian"):
            t3_grad_phys(coords)

    def test_linear_field_gradient(self):
        """Gradient of a linear field f = 2x + 3y should be recovered exactly.

        f at nodes → shape-function interpolation → gradient check.
        Regression tolerance: 1e-14.
        """
        coords = np.array([[1.0, 0.0], [3.0, 0.0], [2.0, 2.0]])
        grad, _ = t3_grad_phys(coords)
        # f = 2x + 3y → f_nodes = [2, 6, 10]
        f_nodes = 2.0 * coords[:, 0] + 3.0 * coords[:, 1]
        # grad_f = sum_i (grad_Ni * fi)
        grad_f = grad @ f_nodes  # (2,) → [df/dx, df/dy]
        np.testing.assert_allclose(grad_f, [2.0, 3.0], atol=1e-14)
