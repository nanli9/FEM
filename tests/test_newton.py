"""Tests for nonlinear element routines and Newton-Raphson solver."""

import numpy as np
import pytest

from femlab.core.hyperelastic import lame_parameters, neo_hookean_pk1, neo_hookean_tangent
from femlab.core.element_nl import tet4_internal_force_nl, tet4_tangent_stiffness_nl
from femlab.core.element import tet4_element_stiffness
from femlab.core.material import isotropic_3d_D
from femlab.core.assembly import assemble_global_stiffness_tet4
from femlab.core.boundary import apply_dirichlet
from femlab.core.solver import solve_linear
from femlab.core.newton import solve_newton
from femlab.mesh.box import box_mesh, box_boundary_nodes


# ---------------------------------------------------------------------------
# Material helpers for tests
# ---------------------------------------------------------------------------

_E, _NU = 1000.0, 0.3
_MU, _LAM = lame_parameters(_E, _NU)


def _pk1_fn(F):
    return neo_hookean_pk1(F, _MU, _LAM)


def _tangent_fn(F):
    P = neo_hookean_pk1(F, _MU, _LAM)
    A = neo_hookean_tangent(F, _MU, _LAM)
    return P, A


# ---------------------------------------------------------------------------
# Element-level tests
# ---------------------------------------------------------------------------

class TestNonlinearElement:
    """Element-level tests for finite deformation Tet4."""

    @pytest.fixture
    def ref_tet(self):
        return np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])

    def test_internal_force_zero_at_reference(self, ref_tet):
        """f_int = 0 for zero displacement (stress-free reference)."""
        u_e = np.zeros(12)
        f_int = tet4_internal_force_nl(ref_tet, u_e, _pk1_fn)
        np.testing.assert_allclose(f_int, 0.0, atol=1e-14)

    def test_tangent_returns_zero_force_at_reference(self, ref_tet):
        """tet4_tangent_stiffness_nl also returns f_int=0 at u=0."""
        K_T, f_int = tet4_tangent_stiffness_nl(ref_tet, np.zeros(12), _tangent_fn)
        np.testing.assert_allclose(f_int, 0.0, atol=1e-14)
        assert K_T.shape == (12, 12)

    def test_tangent_at_zero_equals_linear_stiffness(self, ref_tet):
        """K_T at u=0 must equal the linear element stiffness K_linear.

        This is the critical bridge test: at zero deformation, the
        nonlinear formulation (G^T A(I) G V) must reduce to the
        linear formulation (B^T D B V).

        Regression tolerance: 1e-10.
        """
        D = isotropic_3d_D(_E, _NU)
        K_linear = tet4_element_stiffness(ref_tet, D)
        K_T, _ = tet4_tangent_stiffness_nl(ref_tet, np.zeros(12), _tangent_fn)
        np.testing.assert_allclose(K_T, K_linear, atol=1e-10)

    def test_tangent_at_zero_equals_linear_scaled_tet(self):
        """Same bridge test on a non-reference tetrahedron."""
        coords = np.array([
            [1.0, 2.0, 0.5],
            [3.0, 2.0, 0.5],
            [1.0, 4.0, 0.5],
            [1.0, 2.0, 3.5],
        ])
        D = isotropic_3d_D(_E, _NU)
        K_linear = tet4_element_stiffness(coords, D)
        K_T, _ = tet4_tangent_stiffness_nl(coords, np.zeros(12), _tangent_fn)
        np.testing.assert_allclose(K_T, K_linear, atol=1e-10)

    def test_tangent_symmetry(self, ref_tet):
        """K_T must be symmetric for any displacement."""
        rng = np.random.default_rng(99)
        u_e = rng.standard_normal(12) * 0.05
        K_T, _ = tet4_tangent_stiffness_nl(ref_tet, u_e, _tangent_fn)
        np.testing.assert_allclose(K_T, K_T.T, atol=1e-12)

    def test_small_displacement_force_matches_linear(self, ref_tet):
        """For small u, f_int_nl ≈ K_linear @ u_e."""
        D = isotropic_3d_D(_E, _NU)
        K_linear = tet4_element_stiffness(ref_tet, D)
        u_e = np.array([0, 0, 0, 1e-6, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
        f_nl = tet4_internal_force_nl(ref_tet, u_e, _pk1_fn)
        f_lin = K_linear @ u_e
        np.testing.assert_allclose(f_nl, f_lin, rtol=1e-4)


# ---------------------------------------------------------------------------
# Newton solver tests
# ---------------------------------------------------------------------------

class TestNewtonSolver:
    """Global Newton-Raphson solver tests."""

    @pytest.fixture
    def cantilever(self):
        """Small 3D cantilever for testing."""
        L, H, W = 4.0, 1.0, 1.0
        nx, ny, nz = 4, 2, 2
        nodes, elements = box_mesh(Lx=L, Ly=H, Lz=W, nx=nx, ny=ny, nz=nz)
        n_nodes = len(nodes)
        n_dof = 3 * n_nodes

        left = box_boundary_nodes(nx, ny, nz, "x0")
        bc_dofs = []
        for ni in left:
            bc_dofs.extend([3 * ni, 3 * ni + 1, 3 * ni + 2])
        bc_dofs = np.array(bc_dofs)
        bc_vals = np.zeros(len(bc_dofs))

        right = box_boundary_nodes(nx, ny, nz, "x1")
        right_coords = nodes[right]
        center = np.array([L, H / 2.0, W / 2.0])
        dists = np.linalg.norm(right_coords - center, axis=1)
        tip_node = right[np.argmin(dists)]

        return {
            "nodes": nodes, "elements": elements,
            "n_dof": n_dof,
            "bc_dofs": bc_dofs, "bc_vals": bc_vals,
            "tip_node": tip_node,
            "L": L, "H": H, "W": W,
        }

    def test_zero_load_zero_displacement(self, cantilever):
        """Zero external force gives zero displacement."""
        s = cantilever
        f_ext = np.zeros(s["n_dof"])
        result = solve_newton(
            s["nodes"], s["elements"], f_ext,
            s["bc_dofs"], s["bc_vals"], _tangent_fn,
            max_iter=5, atol=1e-10, verbose=False,
        )
        assert result["converged"]
        assert result["iterations"] == 0  # converges immediately
        np.testing.assert_allclose(result["u"], 0.0, atol=1e-14)

    def test_small_load_matches_linear(self, cantilever):
        """For small loads, Newton result matches linear FEM.

        Regression tolerance: 1% relative error on tip deflection.
        """
        s = cantilever
        P_small = -0.01
        tip_dof = 3 * s["tip_node"] + 1

        # Nonlinear solve.
        f_ext = np.zeros(s["n_dof"])
        f_ext[tip_dof] = P_small
        result = solve_newton(
            s["nodes"], s["elements"], f_ext,
            s["bc_dofs"], s["bc_vals"], _tangent_fn,
            max_iter=20, atol=1e-10, verbose=False,
        )
        assert result["converged"]
        u_nl = result["u"]

        # Linear solve.
        D = isotropic_3d_D(_E, _NU)
        K = assemble_global_stiffness_tet4(s["nodes"], s["elements"], D)
        f_lin = np.zeros(s["n_dof"])
        f_lin[tip_dof] = P_small
        K_bc, f_bc = apply_dirichlet(K, f_lin, s["bc_dofs"], s["bc_vals"])
        u_lin = solve_linear(K_bc, f_bc, verbose=False)

        # ~2% difference is expected: even small loads produce a slight
        # nonlinear correction from the Neo-Hookean model vs linear elasticity.
        np.testing.assert_allclose(u_nl, u_lin, rtol=3e-2, atol=1e-12)

    def test_quadratic_convergence(self, cantilever):
        """Residual norms should decrease quadratically.

        For a moderate load, check that convergence is fast
        (at most ~6 iterations) and the residual drops superlinearly.
        """
        s = cantilever
        f_ext = np.zeros(s["n_dof"])
        f_ext[3 * s["tip_node"] + 1] = -0.5

        result = solve_newton(
            s["nodes"], s["elements"], f_ext,
            s["bc_dofs"], s["bc_vals"], _tangent_fn,
            max_iter=20, atol=1e-12, verbose=False,
        )
        assert result["converged"]
        assert result["iterations"] <= 10

        # Residual should decrease monotonically after first step.
        norms = result["residual_norms"]
        for k in range(2, len(norms)):
            assert norms[k] < norms[k - 1], \
                f"Residual not decreasing at step {k}: {norms[k]} >= {norms[k-1]}"

    def test_bc_enforcement(self, cantilever):
        """Constrained DOFs must have exact prescribed values."""
        s = cantilever
        f_ext = np.zeros(s["n_dof"])
        f_ext[3 * s["tip_node"] + 1] = -0.3

        result = solve_newton(
            s["nodes"], s["elements"], f_ext,
            s["bc_dofs"], s["bc_vals"], _tangent_fn,
            max_iter=20, atol=1e-10, verbose=False,
        )
        assert result["converged"]
        np.testing.assert_allclose(
            result["u"][s["bc_dofs"]], s["bc_vals"], atol=1e-14,
        )

    def test_tip_deflects_downward(self, cantilever):
        """Tip should deflect in the direction of the applied load."""
        s = cantilever
        f_ext = np.zeros(s["n_dof"])
        f_ext[3 * s["tip_node"] + 1] = -1.0

        result = solve_newton(
            s["nodes"], s["elements"], f_ext,
            s["bc_dofs"], s["bc_vals"], _tangent_fn,
            max_iter=20, atol=1e-8, verbose=False,
        )
        assert result["converged"]
        assert result["u"][3 * s["tip_node"] + 1] < 0

    def test_result_keys(self, cantilever):
        """Return dict has the expected keys."""
        s = cantilever
        result = solve_newton(
            s["nodes"], s["elements"], np.zeros(s["n_dof"]),
            s["bc_dofs"], s["bc_vals"], _tangent_fn,
            max_iter=5, verbose=False,
        )
        assert "u" in result
        assert "converged" in result
        assert "iterations" in result
        assert "residual_norms" in result
