"""Unit tests for mass matrix computation.

Tests cover:
- Total mass conservation (sum of diagonal for lumped, row sums for consistent)
- Symmetry and positive-definiteness
- Lumped vs consistent row-sum equivalence
"""

import numpy as np

from femlab.core.mass import (
    t3_element_mass_consistent,
    t3_element_mass_lumped,
    tet4_element_mass_consistent,
    tet4_element_mass_lumped,
    assemble_global_mass_t3,
    assemble_global_mass_tet4,
)
from femlab.mesh.rectangle import rectangle_mesh
from femlab.mesh.box import box_mesh


class TestT3Mass:
    """Tests for T3 element mass matrices."""

    def test_consistent_symmetry(self):
        """Regression tolerance: 1e-15."""
        coords = np.array([[0, 0], [1, 0], [0.5, 0.8]])
        me = t3_element_mass_consistent(coords, rho=1.0, thickness=1.0)
        np.testing.assert_allclose(me, me.T, atol=1e-15)

    def test_consistent_total_mass(self):
        """Sum of all entries = ndof_per_node × total element mass.

        For a triangle: mass = ρ * t * A.
        Regression tolerance: 1e-14.
        """
        coords = np.array([[0, 0], [2, 0], [0, 1]], dtype=float)
        rho, t = 7.0, 0.5
        area = 0.5 * abs(2.0 * 1.0)  # = 1.0
        expected_mass = rho * t * area  # = 3.5
        me = t3_element_mass_consistent(coords, rho, t)
        # Sum of all entries = ndof_per_node * total_mass.
        np.testing.assert_allclose(me.sum(), expected_mass * 2, atol=1e-14)

    def test_lumped_total_mass(self):
        """Diagonal lumped mass must sum to total mass × ndof_per_node.

        Regression tolerance: 1e-14.
        """
        coords = np.array([[0, 0], [2, 0], [0, 1]], dtype=float)
        rho, t = 7.0, 0.5
        area = 1.0
        expected_mass = rho * t * area
        me = t3_element_mass_lumped(coords, rho, t)
        np.testing.assert_allclose(me.trace(), expected_mass * 2, atol=1e-14)

    def test_row_sum_equivalence(self):
        """Row sums of consistent mass must equal lumped diagonal.

        This ensures both mass formulations conserve total mass.
        Regression tolerance: 1e-14.
        """
        coords = np.array([[0, 0], [1, 0], [0.5, 0.8]])
        rho = 3.0
        mc = t3_element_mass_consistent(coords, rho)
        ml = t3_element_mass_lumped(coords, rho)
        np.testing.assert_allclose(mc.sum(axis=1), np.diag(ml), atol=1e-14)


class TestTet4Mass:
    """Tests for Tet4 element mass matrices."""

    def test_consistent_symmetry(self):
        coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        me = tet4_element_mass_consistent(coords, rho=1.0)
        np.testing.assert_allclose(me, me.T, atol=1e-15)

    def test_consistent_total_mass(self):
        """Sum of all entries = ndof_per_node × total element mass.

        Regression tolerance: 1e-14.
        """
        coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        rho = 5.0
        vol = 1.0 / 6.0
        expected_mass = rho * vol
        me = tet4_element_mass_consistent(coords, rho)
        np.testing.assert_allclose(me.sum(), expected_mass * 3, atol=1e-14)

    def test_row_sum_equivalence(self):
        """Regression tolerance: 1e-14."""
        coords = np.array([[0, 0, 0], [2, 0, 0], [1, 1.5, 0], [0.5, 0.5, 2.0]])
        rho = 3.0
        mc = tet4_element_mass_consistent(coords, rho)
        ml = tet4_element_mass_lumped(coords, rho)
        np.testing.assert_allclose(mc.sum(axis=1), np.diag(ml), atol=1e-14)


class TestGlobalMassAssembly:
    """Tests for assembled global mass matrices."""

    def test_t3_total_mass(self):
        """Total mass of a 1×1 rectangle with ρ=2, t=0.5: mass = 2*0.5*1 = 1.

        Regression tolerance: 1e-12.
        """
        nodes, elems = rectangle_mesh(Lx=1.0, Ly=1.0, nx=3, ny=3)
        rho, t = 2.0, 0.5
        M = assemble_global_mass_t3(nodes, elems, rho, t, lumped=True)
        total_mass = M.diagonal().sum() / 2.0  # 2 DOFs per node
        np.testing.assert_allclose(total_mass, rho * t * 1.0, atol=1e-12)

    def test_tet4_total_mass(self):
        """Total mass of a unit cube with ρ=3: mass = 3.

        Regression tolerance: 1e-11.
        """
        nodes, elems = box_mesh(1.0, 1.0, 1.0, nx=2, ny=2, nz=2)
        rho = 3.0
        M = assemble_global_mass_tet4(nodes, elems, rho, lumped=True)
        total_mass = M.diagonal().sum() / 3.0  # 3 DOFs per node
        np.testing.assert_allclose(total_mass, rho * 1.0, atol=1e-11)

    def test_consistent_positive_definite(self):
        """Global consistent mass must have all positive eigenvalues (no rigid body modes).

        Regression tolerance: eigenvalues > 0.
        """
        nodes, elems = rectangle_mesh(Lx=1.0, Ly=1.0, nx=2, ny=2)
        M = assemble_global_mass_t3(nodes, elems, rho=1.0, lumped=False)
        eigvals = np.linalg.eigvalsh(M.toarray())
        assert np.all(eigvals > -1e-14), f"Negative eigenvalue: {eigvals.min()}"
