"""Unit tests for Tet4 basis, quadrature, and element routines."""

import numpy as np
import pytest

from femlab.core.basis import (
    tet4_shape, tet4_grad_ref, tet4_grad_phys, tet4_jacobian,
)
from femlab.core.quadrature import tetrahedron_quadrature
from femlab.core.element import tet4_B_matrix, tet4_element_stiffness, tet4_stress
from femlab.core.material import isotropic_3d_D


# ---------------------------------------------------------------------------
# Basis function tests
# ---------------------------------------------------------------------------

class TestTet4Shape:
    def test_partition_of_unity(self):
        """Shape functions must sum to 1 everywhere.

        Regression tolerance: 1e-15.
        """
        points = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
                  (0.25, 0.25, 0.25), (0.1, 0.2, 0.3)]
        for xi, eta, zeta in points:
            N = tet4_shape(xi, eta, zeta)
            np.testing.assert_allclose(N.sum(), 1.0, atol=1e-15)

    def test_kronecker_property(self):
        """N_i(node_j) = δ_{ij}."""
        ref_nodes = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
        for j, (xi, eta, zeta) in enumerate(ref_nodes):
            N = tet4_shape(xi, eta, zeta)
            for i in range(4):
                expected = 1.0 if i == j else 0.0
                assert abs(N[i] - expected) < 1e-15


class TestTet4Jacobian:
    def test_reference_tet(self):
        """Jacobian of the reference tet is the identity."""
        coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        J = tet4_jacobian(coords)
        np.testing.assert_allclose(J, np.eye(3), atol=1e-15)

    def test_scaled_tet(self):
        s = 2.0
        coords = np.array([[0, 0, 0], [s, 0, 0], [0, s, 0], [0, 0, s]])
        J = tet4_jacobian(coords)
        np.testing.assert_allclose(J, s * np.eye(3), atol=1e-14)

    def test_translated_tet(self):
        offset = np.array([3.0, 5.0, 7.0])
        coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float) + offset
        J = tet4_jacobian(coords)
        np.testing.assert_allclose(J, np.eye(3), atol=1e-14)


class TestTet4GradPhys:
    def test_reference_tet(self):
        """On the reference tet, physical gradients equal reference gradients.

        Regression tolerance: 1e-14.
        """
        coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        grad, det_J = tet4_grad_phys(coords)
        expected = tet4_grad_ref()
        np.testing.assert_allclose(grad, expected, atol=1e-14)
        np.testing.assert_allclose(det_J, 1.0, atol=1e-14)

    def test_linear_field_gradient(self):
        """Gradient of f = 2x + 3y + 5z must be recovered exactly.

        Regression tolerance: 1e-13.
        """
        coords = np.array([[1, 0, 0], [3, 0, 0], [2, 2, 0], [2, 1, 3]], dtype=float)
        grad, _ = tet4_grad_phys(coords)
        f_nodes = 2.0 * coords[:, 0] + 3.0 * coords[:, 1] + 5.0 * coords[:, 2]
        grad_f = grad @ f_nodes  # (3,)
        np.testing.assert_allclose(grad_f, [2.0, 3.0, 5.0], atol=1e-13)

    def test_negative_jacobian_raises(self):
        """Inverted tet should raise."""
        # Swap two nodes to flip orientation.
        coords = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        with pytest.raises(ValueError, match="Non-positive"):
            tet4_grad_phys(coords)


# ---------------------------------------------------------------------------
# Quadrature tests
# ---------------------------------------------------------------------------

class TestTetrahedronQuadrature:
    def test_order1_weight_sum(self):
        """Weights must sum to reference tet volume = 1/6.

        Regression tolerance: 1e-15.
        """
        _, wts = tetrahedron_quadrature(1)
        np.testing.assert_allclose(wts.sum(), 1.0 / 6.0, atol=1e-15)

    def test_order2_weight_sum(self):
        _, wts = tetrahedron_quadrature(2)
        np.testing.assert_allclose(wts.sum(), 1.0 / 6.0, atol=1e-15)

    def test_order1_linear(self):
        """∫ xi dV over ref tet = 1/24.

        Regression tolerance: 1e-15.
        """
        pts, wts = tetrahedron_quadrature(1)
        integral = sum(p[0] * w for p, w in zip(pts, wts))
        np.testing.assert_allclose(integral, 1.0 / 24.0, atol=1e-15)

    def test_order2_quadratic(self):
        """∫ xi^2 dV over ref tet = 1/60.

        Regression tolerance: 1e-14.
        """
        pts, wts = tetrahedron_quadrature(2)
        integral = sum(p[0] ** 2 * w for p, w in zip(pts, wts))
        np.testing.assert_allclose(integral, 1.0 / 60.0, atol=1e-14)

    def test_points_inside(self):
        for order in [1, 2]:
            pts, _ = tetrahedron_quadrature(order)
            for p in pts:
                assert p[0] >= 0 and p[1] >= 0 and p[2] >= 0
                assert p[0] + p[1] + p[2] <= 1.0 + 1e-15


# ---------------------------------------------------------------------------
# Element tests
# ---------------------------------------------------------------------------

class TestTet4Element:
    def test_stiffness_symmetry(self):
        """Regression tolerance: 1e-13."""
        coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        D = isotropic_3d_D(1.0, 0.3)
        ke = tet4_element_stiffness(coords, D)
        np.testing.assert_allclose(ke, ke.T, atol=1e-13)

    def test_stiffness_positive_semidefinite(self):
        """6 rigid body modes → 6 zero eigenvalues, rest positive.

        Regression tolerance: eigenvalues >= -1e-12.
        """
        coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        D = isotropic_3d_D(1.0, 0.3)
        ke = tet4_element_stiffness(coords, D)
        eigvals = np.linalg.eigvalsh(ke)
        assert np.all(eigvals >= -1e-12)

    def test_rigid_body_null_space(self):
        """Translations must be in the null space.

        Regression tolerance: 1e-12.
        """
        coords = np.array([[0, 0, 0], [1, 0, 0], [0.5, 0.8, 0], [0.3, 0.4, 0.7]])
        D = isotropic_3d_D(200.0, 0.3)
        ke = tet4_element_stiffness(coords, D)

        for axis in range(3):
            t = np.zeros(12)
            for i in range(4):
                t[3 * i + axis] = 1.0
            np.testing.assert_allclose(ke @ t, 0.0, atol=1e-12)

    def test_rigid_body_rotation_null_space(self):
        """Infinitesimal rotations must be in the null space.

        Rotation about z: u = -y, v = x, w = 0.
        Regression tolerance: 1e-11.
        """
        coords = np.array([[0, 0, 0], [1, 0, 0], [0.5, 0.8, 0], [0.3, 0.4, 0.7]])
        D = isotropic_3d_D(200.0, 0.3)
        ke = tet4_element_stiffness(coords, D)

        # Rotation about z-axis
        rot_z = np.zeros(12)
        for i in range(4):
            rot_z[3 * i]     = -coords[i, 1]
            rot_z[3 * i + 1] =  coords[i, 0]
        np.testing.assert_allclose(ke @ rot_z, 0.0, atol=1e-11)

        # Rotation about x-axis
        rot_x = np.zeros(12)
        for i in range(4):
            rot_x[3 * i + 1] = -coords[i, 2]
            rot_x[3 * i + 2] =  coords[i, 1]
        np.testing.assert_allclose(ke @ rot_x, 0.0, atol=1e-11)

    def test_uniaxial_stress(self):
        """εxx = 0.001 → check σxx = E(1-ν)/((1+ν)(1-2ν)) * εxx.

        Regression tolerance: 1e-10.
        """
        E, nu = 200e3, 0.3
        D = isotropic_3d_D(E, nu)
        coords = np.array([[0, 0, 0], [2, 0, 0], [1, 1, 0], [1, 0.5, 1.5]])
        eps = 0.001
        u_e = np.zeros(12)
        for i in range(4):
            u_e[3 * i] = eps * coords[i, 0]
        strain, stress = tet4_stress(coords, D, u_e)
        np.testing.assert_allclose(strain[0], eps, atol=1e-14)
        np.testing.assert_allclose(strain[1], 0.0, atol=1e-14)
        np.testing.assert_allclose(strain[2], 0.0, atol=1e-14)
        expected_sxx = E * (1 - nu) / ((1 + nu) * (1 - 2 * nu)) * eps
        np.testing.assert_allclose(stress[0], expected_sxx, rtol=1e-10)
