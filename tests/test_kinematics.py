"""Tests for deformation gradient and G matrix computation."""

import numpy as np
import pytest

from femlab.core.kinematics import deformation_gradient_tet4, tet4_G_matrix


class TestDeformationGradient:
    """Tests for deformation_gradient_tet4."""

    @pytest.fixture
    def ref_tet(self):
        """Unit reference tetrahedron."""
        return np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])

    def test_zero_displacement(self, ref_tet):
        """F = I for zero displacement."""
        u_e = np.zeros(12)
        F, dN_dX, V = deformation_gradient_tet4(ref_tet, u_e)
        np.testing.assert_allclose(F, np.eye(3), atol=1e-15)
        assert V > 0

    def test_rigid_translation(self, ref_tet):
        """F = I for uniform translation (no deformation)."""
        u_e = np.tile([1.0, 2.0, 3.0], 4)
        F, _, _ = deformation_gradient_tet4(ref_tet, u_e)
        np.testing.assert_allclose(F, np.eye(3), atol=1e-14)

    def test_uniform_stretch_x(self, ref_tet):
        """u_x = eps * X gives F = diag(1+eps, 1, 1)."""
        eps = 0.1
        u_e = np.zeros(12)
        for a in range(4):
            u_e[3 * a] = eps * ref_tet[a, 0]
        F, _, _ = deformation_gradient_tet4(ref_tet, u_e)
        expected = np.diag([1.0 + eps, 1.0, 1.0])
        np.testing.assert_allclose(F, expected, atol=1e-14)

    def test_uniform_stretch_y(self, ref_tet):
        """u_y = eps * Y gives F = diag(1, 1+eps, 1)."""
        eps = 0.2
        u_e = np.zeros(12)
        for a in range(4):
            u_e[3 * a + 1] = eps * ref_tet[a, 1]
        F, _, _ = deformation_gradient_tet4(ref_tet, u_e)
        expected = np.diag([1.0, 1.0 + eps, 1.0])
        np.testing.assert_allclose(F, expected, atol=1e-14)

    def test_simple_shear(self, ref_tet):
        """u_x = gamma * Y gives F_{01} = gamma."""
        gamma = 0.15
        u_e = np.zeros(12)
        for a in range(4):
            u_e[3 * a] = gamma * ref_tet[a, 1]
        F, _, _ = deformation_gradient_tet4(ref_tet, u_e)
        expected = np.eye(3)
        expected[0, 1] = gamma
        np.testing.assert_allclose(F, expected, atol=1e-14)

    def test_volume_unit_tet(self, ref_tet):
        """Reference volume of unit tet = 1/6."""
        _, _, V = deformation_gradient_tet4(ref_tet, np.zeros(12))
        np.testing.assert_allclose(V, 1.0 / 6.0, atol=1e-15)

    def test_volume_scaled_tet(self):
        """Reference volume scales with element size."""
        coords = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 4.0],
        ])
        _, _, V = deformation_gradient_tet4(coords, np.zeros(12))
        # Volume = |det([2,0,0; 0,3,0; 0,0,4])| / 6 = 24/6 = 4
        np.testing.assert_allclose(V, 4.0, atol=1e-13)

    def test_stretch_on_scaled_tet(self):
        """F correct for uniform stretch on a non-reference tet."""
        coords = np.array([
            [1.0, 2.0, 3.0],
            [3.0, 2.0, 3.0],
            [1.0, 5.0, 3.0],
            [1.0, 2.0, 7.0],
        ])
        eps = 0.05
        u_e = np.zeros(12)
        for a in range(4):
            u_e[3 * a + 2] = eps * coords[a, 2]  # u_z = eps * Z
        F, _, _ = deformation_gradient_tet4(coords, u_e)
        expected = np.eye(3)
        expected[2, 2] = 1.0 + eps
        np.testing.assert_allclose(F, expected, atol=1e-13)


class TestGMatrix:
    """Tests for tet4_G_matrix."""

    @pytest.fixture
    def ref_tet(self):
        return np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])

    def test_shape(self, ref_tet):
        """G must be (9, 12)."""
        _, dN_dX, _ = deformation_gradient_tet4(ref_tet, np.zeros(12))
        G = tet4_G_matrix(dN_dX)
        assert G.shape == (9, 12)

    def test_consistency_with_F(self, ref_tet):
        """G @ u_e must equal vec(F - I) for random displacement."""
        rng = np.random.default_rng(42)
        u_e = rng.standard_normal(12) * 0.1
        F, dN_dX, _ = deformation_gradient_tet4(ref_tet, u_e)
        G = tet4_G_matrix(dN_dX)
        np.testing.assert_allclose(G @ u_e, (F - np.eye(3)).ravel(), atol=1e-14)

    def test_consistency_scaled_tet(self):
        """G @ u_e = vec(F - I) on a non-reference tet."""
        coords = np.array([
            [1.0, 0.0, 0.0],
            [3.0, 1.0, 0.5],
            [1.5, 2.0, 0.0],
            [2.0, 0.5, 2.0],
        ])
        rng = np.random.default_rng(123)
        u_e = rng.standard_normal(12) * 0.05
        F, dN_dX, _ = deformation_gradient_tet4(coords, u_e)
        G = tet4_G_matrix(dN_dX)
        np.testing.assert_allclose(G @ u_e, (F - np.eye(3)).ravel(), atol=1e-13)

    def test_zero_for_translation(self, ref_tet):
        """G @ u_translation = 0 (translation has no gradient)."""
        _, dN_dX, _ = deformation_gradient_tet4(ref_tet, np.zeros(12))
        G = tet4_G_matrix(dN_dX)
        u_trans = np.tile([5.0, -3.0, 1.0], 4)
        np.testing.assert_allclose(G @ u_trans, 0.0, atol=1e-14)
