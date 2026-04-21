"""Hyperelastic material models for finite deformation analysis.

Provides strain energy density W, first Piola-Kirchhoff stress P = dW/dF,
and material tangent A = dP/dF for compressible hyperelastic materials.

Compressible Neo-Hookean model:
    W(F) = mu/2 (I1 - 3) - mu ln(J) + lam/2 (ln J)^2

where I1 = tr(F^T F), J = det(F), and mu, lam are Lame parameters.
"""

import numpy as np


def lame_parameters(E: float, nu: float) -> tuple[float, float]:
    """Convert Young's modulus and Poisson's ratio to Lame parameters.

    Args:
        E: Young's modulus.
        nu: Poisson's ratio.

    Returns:
        mu: Shear modulus (second Lame parameter).
        lam: First Lame parameter.
    """
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return mu, lam


def neo_hookean_energy(F: np.ndarray, mu: float, lam: float) -> float:
    """Compressible Neo-Hookean strain energy density.

    W = mu/2 (tr(F^T F) - 3) - mu ln(J) + lam/2 (ln J)^2

    Args:
        F: (3, 3) deformation gradient.
        mu: Shear modulus.
        lam: First Lame parameter.

    Returns:
        Strain energy density (scalar).
    """
    J = np.linalg.det(F)
    I1 = np.trace(F.T @ F)
    ln_J = np.log(J)
    return 0.5 * mu * (I1 - 3.0) - mu * ln_J + 0.5 * lam * ln_J ** 2


def neo_hookean_pk1(F: np.ndarray, mu: float, lam: float) -> np.ndarray:
    """First Piola-Kirchhoff stress for compressible Neo-Hookean material.

    P = mu * F + (lam * ln(J) - mu) * F^{-T}

    Args:
        F: (3, 3) deformation gradient.
        mu: Shear modulus.
        lam: First Lame parameter.

    Returns:
        P: (3, 3) first Piola-Kirchhoff stress tensor.
    """
    J = np.linalg.det(F)
    F_inv_T = np.linalg.inv(F).T
    ln_J = np.log(J)
    return mu * F + (lam * ln_J - mu) * F_inv_T


def neo_hookean_tangent(F: np.ndarray, mu: float, lam: float) -> np.ndarray:
    """Material tangent (4th-order tensor as 9x9 matrix) for Neo-Hookean.

    A_{iJkL} = mu * d_{ik} d_{JL}
             - (lam ln J - mu) * F^{-1}_{Jk} F^{-1}_{Li}
             + lam * F^{-1}_{Ji} F^{-1}_{Lk}

    Vectorized as A[3*i+J, 3*k+L] = A_{iJkL}.

    At F = I this recovers the linear elastic tensor:
        A_{iJkL} = mu (d_{ik} d_{JL} + d_{iL} d_{Jk}) + lam d_{iJ} d_{kL}

    Args:
        F: (3, 3) deformation gradient.
        mu: Shear modulus.
        lam: First Lame parameter.

    Returns:
        A: (9, 9) material tangent matrix.
    """
    J = np.linalg.det(F)
    F_inv = np.linalg.inv(F)
    ln_J = np.log(J)

    I3 = np.eye(3)
    c = lam * ln_J - mu

    # Build A as (3,3,3,3) then reshape to (9,9).
    # Term 1: mu * d_{ik} * d_{JL}
    # Term 2: -c * F_inv_{Jk} * F_inv_{Li}
    # Term 3: lam * F_inv_{Ji} * F_inv_{Lk}
    A4 = (mu * np.einsum('ik,JL->iJkL', I3, I3)
          - c * np.einsum('Jk,Li->iJkL', F_inv, F_inv)
          + lam * np.einsum('Ji,Lk->iJkL', F_inv, F_inv))

    # Reshape: A[3*i+J, 3*k+L] = A4[i,J,k,L] (C-order).
    return A4.reshape(9, 9)
