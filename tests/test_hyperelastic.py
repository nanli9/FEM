"""Tests for hyperelastic material models."""

import numpy as np
import pytest

from femlab.core.hyperelastic import (
    lame_parameters,
    neo_hookean_energy,
    neo_hookean_pk1,
    neo_hookean_tangent,
)


class TestLameParameters:
    def test_known_values(self):
        E, nu = 100.0, 0.25
        mu, lam = lame_parameters(E, nu)
        # mu = E / (2(1+nu)) = 100 / 2.5 = 40
        np.testing.assert_allclose(mu, 40.0, atol=1e-14)
        # lam = E*nu / ((1+nu)(1-2nu)) = 25 / (1.25*0.5) = 40
        np.testing.assert_allclose(lam, 40.0, atol=1e-14)

    def test_incompressible_limit(self):
        """Lambda diverges as nu -> 0.5."""
        _, lam = lame_parameters(100.0, 0.499)
        assert lam > 1e4


class TestNeoHookeanEnergy:
    @pytest.fixture
    def material(self):
        return lame_parameters(1000.0, 0.3)

    def test_zero_at_identity(self, material):
        """W(I) = 0 (stress-free reference)."""
        mu, lam = material
        W = neo_hookean_energy(np.eye(3), mu, lam)
        np.testing.assert_allclose(W, 0.0, atol=1e-15)

    def test_positive_under_stretch(self, material):
        mu, lam = material
        F = np.diag([1.1, 0.95, 1.02])
        assert neo_hookean_energy(F, mu, lam) > 0

    def test_positive_under_shear(self, material):
        mu, lam = material
        F = np.eye(3)
        F[0, 1] = 0.1
        assert neo_hookean_energy(F, mu, lam) > 0

    def test_frame_indifference(self, material):
        """W(QF) = W(F) for rotation Q."""
        mu, lam = material
        F = np.array([[1.1, 0.02, 0.0],
                       [0.01, 0.98, 0.03],
                       [0.0, -0.01, 1.05]])
        # Rotation about z by 30 degrees.
        theta = np.pi / 6
        Q = np.array([[np.cos(theta), -np.sin(theta), 0],
                       [np.sin(theta), np.cos(theta), 0],
                       [0, 0, 1]])
        W_F = neo_hookean_energy(F, mu, lam)
        W_QF = neo_hookean_energy(Q @ F, mu, lam)
        np.testing.assert_allclose(W_QF, W_F, rtol=1e-12)


class TestNeoHookeanPK1:
    @pytest.fixture
    def material(self):
        return lame_parameters(1000.0, 0.3)

    def test_stress_free_reference(self, material):
        """P(I) = 0."""
        mu, lam = material
        P = neo_hookean_pk1(np.eye(3), mu, lam)
        np.testing.assert_allclose(P, 0.0, atol=1e-13)

    def test_numerical_gradient_of_energy(self, material):
        """P_{iJ} = dW/dF_{iJ} verified by central differences."""
        mu, lam = material
        F = np.array([[1.05, 0.02, 0.0],
                       [0.01, 0.98, 0.03],
                       [0.0, -0.01, 1.01]])
        P_analytical = neo_hookean_pk1(F, mu, lam)

        eps = 1e-7
        P_numerical = np.zeros((3, 3))
        for i in range(3):
            for J in range(3):
                F_p = F.copy(); F_p[i, J] += eps
                F_m = F.copy(); F_m[i, J] -= eps
                P_numerical[i, J] = (
                    neo_hookean_energy(F_p, mu, lam) -
                    neo_hookean_energy(F_m, mu, lam)
                ) / (2.0 * eps)

        np.testing.assert_allclose(P_analytical, P_numerical, atol=1e-5)

    def test_uniaxial_tension_sign(self, material):
        """Positive P_xx for uniaxial stretch."""
        mu, lam = material
        F = np.diag([1.1, 1.0, 1.0])
        P = neo_hookean_pk1(F, mu, lam)
        assert P[0, 0] > 0

    def test_uniaxial_compression_sign(self, material):
        """Negative P_xx for uniaxial compression."""
        mu, lam = material
        F = np.diag([0.9, 1.0, 1.0])
        P = neo_hookean_pk1(F, mu, lam)
        assert P[0, 0] < 0


class TestNeoHookeanTangent:
    @pytest.fixture
    def material(self):
        return lame_parameters(1000.0, 0.3)

    def test_tangent_at_identity_matches_linear(self, material):
        """At F=I, the tangent must equal the linear elastic tensor.

        A_{iJkL} = mu(d_ik d_JL + d_iL d_Jk) + lam d_iJ d_kL
        """
        mu, lam = material
        A = neo_hookean_tangent(np.eye(3), mu, lam)

        # Build linear elastic tensor in 9x9 form.
        C = np.zeros((9, 9))
        I3 = np.eye(3)
        for i in range(3):
            for J in range(3):
                for k in range(3):
                    for L in range(3):
                        C[3 * i + J, 3 * k + L] = (
                            mu * I3[i, k] * I3[J, L] +
                            mu * I3[i, L] * I3[J, k] +
                            lam * I3[i, J] * I3[k, L]
                        )

        np.testing.assert_allclose(A, C, atol=1e-12)

    def test_numerical_tangent(self, material):
        """A_{:, 3k+L} = dP/dF_{kL} verified by central differences."""
        mu, lam = material
        F = np.array([[1.05, 0.02, 0.0],
                       [0.01, 0.98, 0.03],
                       [0.0, -0.01, 1.01]])
        A_analytical = neo_hookean_tangent(F, mu, lam)

        eps = 1e-7
        A_numerical = np.zeros((9, 9))
        for k in range(3):
            for L in range(3):
                F_p = F.copy(); F_p[k, L] += eps
                F_m = F.copy(); F_m[k, L] -= eps
                dP = (neo_hookean_pk1(F_p, mu, lam) -
                      neo_hookean_pk1(F_m, mu, lam)) / (2.0 * eps)
                A_numerical[:, 3 * k + L] = dP.ravel()

        np.testing.assert_allclose(A_analytical, A_numerical, atol=1e-4)

    def test_major_symmetry(self, material):
        """A_{iJkL} = A_{kLiJ} implies A.T = A (9x9 is symmetric)."""
        mu, lam = material
        F = np.array([[1.1, 0.05, -0.02],
                       [0.03, 0.95, 0.01],
                       [-0.01, 0.02, 1.08]])
        A = neo_hookean_tangent(F, mu, lam)
        np.testing.assert_allclose(A, A.T, atol=1e-12)

    def test_tangent_shape(self, material):
        mu, lam = material
        A = neo_hookean_tangent(np.eye(3), mu, lam)
        assert A.shape == (9, 9)
