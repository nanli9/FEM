"""Unit tests for T3 element routines (B-matrix, stiffness, stress)."""

import numpy as np

from femlab.core.basis import t3_grad_phys
from femlab.core.element import t3_B_matrix, t3_element_stiffness, t3_stress
from femlab.core.material import plane_stress_D


class TestBMatrix:
    """Tests for the strain-displacement B matrix."""

    def test_shape(self):
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        grad, _ = t3_grad_phys(coords)
        B = t3_B_matrix(grad)
        assert B.shape == (3, 6)

    def test_rigid_body_translation(self):
        """Rigid translation should produce zero strain.

        u = [c, 0, c, 0, c, 0]  → εxx = εyy = γxy = 0.
        Regression tolerance: 1e-15.
        """
        coords = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 1.5]])
        grad, _ = t3_grad_phys(coords)
        B = t3_B_matrix(grad)
        # Uniform x-translation
        u_x = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        np.testing.assert_allclose(B @ u_x, 0.0, atol=1e-15)
        # Uniform y-translation
        u_y = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        np.testing.assert_allclose(B @ u_y, 0.0, atol=1e-15)

    def test_uniform_stretch(self):
        """Uniform x-stretch: u_i = ε * x_i, v_i = 0.

        Should give εxx = ε, εyy = 0, γxy = 0.
        Regression tolerance: 1e-14.
        """
        coords = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 1.0]])
        grad, _ = t3_grad_phys(coords)
        B = t3_B_matrix(grad)
        eps = 0.01
        u_e = np.array([eps * 0.0, 0.0, eps * 2.0, 0.0, eps * 1.0, 0.0])
        strain = B @ u_e
        np.testing.assert_allclose(strain, [eps, 0.0, 0.0], atol=1e-14)


class TestElementStiffness:
    """Tests for the T3 element stiffness matrix."""

    def test_symmetry(self):
        """Element stiffness must be symmetric.

        Regression tolerance: 1e-14.
        """
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.8]])
        D = plane_stress_D(1.0, 0.3)
        ke = t3_element_stiffness(coords, D)
        np.testing.assert_allclose(ke, ke.T, atol=1e-14)

    def test_positive_semidefinite(self):
        """Stiffness eigenvalues must be non-negative (3 rigid-body modes → 3 zeros).

        Regression tolerance: eigenvalues >= -1e-12.
        """
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        D = plane_stress_D(1.0, 0.3)
        ke = t3_element_stiffness(coords, D)
        eigvals = np.linalg.eigvalsh(ke)
        assert np.all(eigvals >= -1e-12), f"Negative eigenvalue: {eigvals.min()}"

    def test_rigid_body_null_space(self):
        """Rigid body modes must be in the null space of element stiffness.

        For 2D: two translations + one rotation (linearized).
        Regression tolerance: 1e-12.
        """
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.8]])
        D = plane_stress_D(200.0, 0.3)
        ke = t3_element_stiffness(coords, D)

        # Translation in x
        tx = np.array([1, 0, 1, 0, 1, 0], dtype=float)
        np.testing.assert_allclose(ke @ tx, 0.0, atol=1e-12)

        # Translation in y
        ty = np.array([0, 1, 0, 1, 0, 1], dtype=float)
        np.testing.assert_allclose(ke @ ty, 0.0, atol=1e-12)

        # Infinitesimal rotation about origin: u = -y, v = x
        rot = np.zeros(6)
        for i in range(3):
            rot[2 * i] = -coords[i, 1]
            rot[2 * i + 1] = coords[i, 0]
        np.testing.assert_allclose(ke @ rot, 0.0, atol=1e-12)

    def test_area_scaling(self):
        """Stiffness should scale inversely with element area.

        k ∝ 1/A  because B ∝ 1/A^{1/2} and ∫dA ∝ A → B^T D B * A ∝ 1/A.
        Regression tolerance: 1e-12.
        """
        D = plane_stress_D(1.0, 0.3)
        coords1 = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        coords2 = coords1 * 2.0
        ke1 = t3_element_stiffness(coords1, D)
        ke2 = t3_element_stiffness(coords2, D)
        # Area ratio = 4, so ke2 = ke1 / 4 * 4 ... actually:
        # B scales as 1/L, B^T D B scales as 1/L^2, area scales as L^2 → product is O(1).
        # For similar triangles scaled by s: ke2 = ke1 (stiffness is scale-invariant for T3).
        np.testing.assert_allclose(ke2, ke1, atol=1e-12)


class TestT3Stress:
    """Tests for stress recovery."""

    def test_uniaxial_stress(self):
        """Apply uniaxial x-strain εxx=0.001, expect σxx = E*εxx/(1-ν²) for plane stress.

        Regression tolerance: 1e-10.
        """
        E, nu = 200e3, 0.3
        D = plane_stress_D(E, nu)
        coords = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 1.0]])
        eps = 0.001
        # u_i = eps * x_i, v_i = 0
        u_e = np.array([eps * coords[0, 0], 0.0,
                        eps * coords[1, 0], 0.0,
                        eps * coords[2, 0], 0.0])
        strain, stress = t3_stress(coords, D, u_e)
        np.testing.assert_allclose(strain[0], eps, atol=1e-14)
        np.testing.assert_allclose(strain[1], 0.0, atol=1e-14)
        np.testing.assert_allclose(strain[2], 0.0, atol=1e-14)
        # σxx = E/(1-ν²) * (εxx + ν*εyy) = E/(1-ν²) * εxx
        expected_sxx = E / (1.0 - nu ** 2) * eps
        np.testing.assert_allclose(stress[0], expected_sxx, rtol=1e-10)
