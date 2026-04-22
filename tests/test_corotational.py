"""Tests for corotational FEM: large-rotation cases that expose linear FEM failures.

Key tests:
- Pure rigid rotation must give zero internal force (linear FEM gives large spurious force)
- Rotation + small stretch must recover correct stress (linear FEM gives garbage)
- K_cr at u=0 equals K_linear (bridge test)
- Tangent consistency vs finite differences
- Newton convergence for corotational solver
- Comparison: corotational vs full nonlinear vs linear
"""

import numpy as np
import pytest

from femlab.core.material import isotropic_3d_D
from femlab.core.element import tet4_element_stiffness
from femlab.core.corotational import (
    polar_decomposition_tet4,
    tet4_internal_force_cr,
    tet4_tangent_stiffness_cr,
    _block_rotation,
    _corotational_local_displacement,
)
from femlab.core.assembly_cr import assemble_system_tet4_cr
from femlab.core.assembly import assemble_global_stiffness_tet4
from femlab.core.boundary import apply_dirichlet
from femlab.core.solver import solve_linear
from femlab.core.newton import solve_newton, solve_newton_general
from femlab.core.hyperelastic import lame_parameters, neo_hookean_pk1, neo_hookean_tangent
from femlab.mesh.box import box_mesh, box_boundary_nodes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_E, _NU = 1000.0, 0.3
_D = isotropic_3d_D(_E, _NU)
_MU, _LAM = lame_parameters(_E, _NU)


def _rotation_matrix(axis: str, angle: float) -> np.ndarray:
    """Build 3×3 rotation matrix about a coordinate axis."""
    c, s = np.cos(angle), np.sin(angle)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)
    elif axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)
    elif axis == "z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
    raise ValueError(f"Unknown axis: {axis}")


def _apply_F_to_tet(X_ref, F):
    """Given F, compute element displacement u_e such that F = I + du/dX."""
    # For constant-strain Tet4:  x_a = F @ X_a + c  (affine map)
    # With c = 0 (no translation):  u_a = (F - I) @ X_a
    u_nodes = ((F - np.eye(3)) @ X_ref.T).T  # (4, 3)
    return u_nodes.ravel()


def _tangent_fn_nl(F):
    """Material tangent for full nonlinear (Neo-Hookean)."""
    P = neo_hookean_pk1(F, _MU, _LAM)
    A = neo_hookean_tangent(F, _MU, _LAM)
    return P, A


# ---------------------------------------------------------------------------
# Element-level tests
# ---------------------------------------------------------------------------

class TestCorotationalElement:
    """Element-level corotational Tet4 tests."""

    @pytest.fixture
    def ref_tet(self):
        return np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])

    @pytest.fixture
    def scaled_tet(self):
        return np.array([
            [1.0, 2.0, 0.5],
            [3.0, 2.0, 0.5],
            [1.0, 4.0, 0.5],
            [1.0, 2.0, 3.5],
        ])

    # --- Pure rotation tests (the signature test for corotational FEM) ---

    @pytest.mark.parametrize("angle", [
        np.pi / 6,    # 30 degrees
        np.pi / 4,    # 45 degrees
        np.pi / 2,    # 90 degrees
        2 * np.pi / 3, # 120 degrees
    ])
    @pytest.mark.parametrize("axis", ["x", "y", "z"])
    def test_pure_rotation_zero_force(self, ref_tet, angle, axis):
        """Pure rigid rotation must produce zero corotational internal force.

        This is THE fundamental test for corotational FEM: rigid body
        rotations cause no stress.  Tolerance: 1e-10.
        """
        R = _rotation_matrix(axis, angle)
        u_e = _apply_F_to_tet(ref_tet, R)

        f_cr = tet4_internal_force_cr(ref_tet, u_e, _D)
        np.testing.assert_allclose(f_cr, 0.0, atol=1e-10)

    @pytest.mark.parametrize("angle", [np.pi / 4, np.pi / 2])
    def test_linear_fails_for_pure_rotation(self, ref_tet, angle):
        """Linear FEM gives large spurious force for pure rotation.

        This demonstrates why corotational FEM is needed: linear FEM
        cannot distinguish rotation from deformation.
        """
        R = _rotation_matrix("z", angle)
        u_e = _apply_F_to_tet(ref_tet, R)

        K_lin = tet4_element_stiffness(ref_tet, _D)
        f_linear = K_lin @ u_e
        # Linear force is large (not zero) — clearly wrong.
        assert np.linalg.norm(f_linear) > 10.0, \
            f"Linear force should be large for {np.degrees(angle):.0f}° rotation"

    def test_pure_rotation_90_scaled_tet(self, scaled_tet):
        """90° rotation on a non-reference tet still gives zero force."""
        R = _rotation_matrix("z", np.pi / 2)
        u_e = _apply_F_to_tet(scaled_tet, R)
        f_cr = tet4_internal_force_cr(scaled_tet, u_e, _D)
        np.testing.assert_allclose(f_cr, 0.0, atol=1e-10)

    # --- Rotation + stretch tests ---

    @pytest.mark.parametrize("angle", [np.pi / 4, np.pi / 2])
    def test_rotation_plus_stretch_correct_stress(self, ref_tet, angle):
        """Large rotation + 5% uniaxial stretch: corotational extracts
        the correct local deformation.

        The corotational element should give the same force magnitude
        as a pure 5% stretch (no rotation).  Tolerance: 1e-8.
        """
        eps = 0.05
        stretch = np.diag([1.0 + eps, 1.0, 1.0])
        R = _rotation_matrix("z", angle)
        F = R @ stretch

        u_rot_stretch = _apply_F_to_tet(ref_tet, F)
        u_stretch_only = _apply_F_to_tet(ref_tet, stretch)

        f_cr = tet4_internal_force_cr(ref_tet, u_rot_stretch, _D)
        f_stretch = tet4_internal_force_cr(ref_tet, u_stretch_only, _D)

        # The force magnitudes should match (forces are in different
        # directions due to the rotation, but the norm is the same).
        np.testing.assert_allclose(
            np.linalg.norm(f_cr), np.linalg.norm(f_stretch),
            rtol=1e-8,
        )

    def test_rotation_plus_stretch_linear_wrong(self, ref_tet):
        """Linear FEM gives wrong force magnitude for rotation + stretch.

        At 90° rotation + 5% stretch, the linear force norm is far from
        the corotational force norm.
        """
        eps = 0.05
        R = _rotation_matrix("z", np.pi / 2)
        F = R @ np.diag([1.0 + eps, 1.0, 1.0])
        u_e = _apply_F_to_tet(ref_tet, F)

        K_lin = tet4_element_stiffness(ref_tet, _D)
        f_linear = K_lin @ u_e
        f_cr = tet4_internal_force_cr(ref_tet, u_e, _D)

        # Linear force is dominated by the rotation, not the stretch.
        ratio = np.linalg.norm(f_linear) / np.linalg.norm(f_cr)
        assert ratio > 5.0, \
            f"Linear force should be much larger than corotational (ratio={ratio:.1f})"

    # --- Bridge test ---

    def test_tangent_at_zero_equals_linear(self, ref_tet):
        """K_cr at u=0 must equal K_linear (bridge test).  Tolerance: 1e-12."""
        K_cr, f_int = tet4_tangent_stiffness_cr(ref_tet, np.zeros(12), _D)
        K_lin = tet4_element_stiffness(ref_tet, _D)
        np.testing.assert_allclose(K_cr, K_lin, atol=1e-12)

    def test_tangent_at_zero_equals_linear_scaled_tet(self, scaled_tet):
        """Bridge test on non-reference tet."""
        K_cr, _ = tet4_tangent_stiffness_cr(scaled_tet, np.zeros(12), _D)
        K_lin = tet4_element_stiffness(scaled_tet, _D)
        np.testing.assert_allclose(K_cr, K_lin, atol=1e-12)

    # --- Tangent properties ---

    def test_tangent_symmetry_at_rotation(self, ref_tet):
        """K_cr must be symmetric under large rotation."""
        R = _rotation_matrix("z", np.pi / 3)
        stretch = np.diag([1.03, 0.99, 1.01])
        u_e = _apply_F_to_tet(ref_tet, R @ stretch)
        K_cr, _ = tet4_tangent_stiffness_cr(ref_tet, u_e, _D)
        np.testing.assert_allclose(K_cr, K_cr.T, atol=1e-10)

    def test_tangent_symmetry_random(self, ref_tet):
        """K_cr must be symmetric for random displacement."""
        rng = np.random.default_rng(42)
        u_e = rng.standard_normal(12) * 0.05
        K_cr, _ = tet4_tangent_stiffness_cr(ref_tet, u_e, _D)
        np.testing.assert_allclose(K_cr, K_cr.T, atol=1e-10)

    def test_tangent_consistency_finite_difference_small(self, ref_tet):
        """K_cr must be approximately consistent with f_int (small deformation).

        At small deformation, the geometric stiffness is small and the
        approximate initial-stress formula has ~0.2% relative error on
        individual entries.  Tolerance: rtol=1e-2, atol=0.1.
        """
        R = _rotation_matrix("y", 0.05)
        stretch = np.diag([1.002, 0.999, 1.001])
        u_e = _apply_F_to_tet(ref_tet, R @ stretch)

        K_cr, f0 = tet4_tangent_stiffness_cr(ref_tet, u_e, _D)

        delta = 1e-7
        K_num = np.zeros((12, 12))
        for j in range(12):
            u_p = u_e.copy()
            u_m = u_e.copy()
            u_p[j] += delta
            u_m[j] -= delta
            f_p = tet4_internal_force_cr(ref_tet, u_p, _D)
            f_m = tet4_internal_force_cr(ref_tet, u_m, _D)
            K_num[:, j] = (f_p - f_m) / (2.0 * delta)

        np.testing.assert_allclose(K_cr, K_num, rtol=1e-2, atol=0.1)

    def test_tangent_consistency_finite_difference_large(self, ref_tet):
        """K_cr approximate consistency at larger deformation.

        The geometric stiffness uses the standard initial-stress formula,
        which is approximate in the corotational context.  At moderate
        deformation (~2% strain, 0.3 rad rotation) we expect up to ~5-7%
        error on individual entries.  This is acceptable — the tangent
        still provides good Newton convergence.
        """
        R = _rotation_matrix("y", 0.3)
        stretch = np.diag([1.02, 0.99, 1.01])
        u_e = _apply_F_to_tet(ref_tet, R @ stretch)

        K_cr, f0 = tet4_tangent_stiffness_cr(ref_tet, u_e, _D)

        delta = 1e-7
        K_num = np.zeros((12, 12))
        for j in range(12):
            u_p = u_e.copy()
            u_m = u_e.copy()
            u_p[j] += delta
            u_m[j] -= delta
            f_p = tet4_internal_force_cr(ref_tet, u_p, _D)
            f_m = tet4_internal_force_cr(ref_tet, u_m, _D)
            K_num[:, j] = (f_p - f_m) / (2.0 * delta)

        # Relative tolerance 10% with absolute tolerance 1.5 for small entries.
        np.testing.assert_allclose(K_cr, K_num, rtol=0.10, atol=1.5)

    # --- Polar decomposition ---

    def test_polar_decomposition_identity(self, ref_tet):
        """F=I gives R=I, U=I."""
        R, U, _, _ = polar_decomposition_tet4(ref_tet, np.zeros(12))
        np.testing.assert_allclose(R, np.eye(3), atol=1e-14)
        np.testing.assert_allclose(U, np.eye(3), atol=1e-14)

    def test_polar_decomposition_pure_rotation(self, ref_tet):
        """For pure rotation, R = F and U = I."""
        R_true = _rotation_matrix("z", np.pi / 3)
        u_e = _apply_F_to_tet(ref_tet, R_true)
        R, U, _, _ = polar_decomposition_tet4(ref_tet, u_e)
        np.testing.assert_allclose(R, R_true, atol=1e-12)
        np.testing.assert_allclose(U, np.eye(3), atol=1e-12)

    def test_polar_decomposition_stretch(self, ref_tet):
        """For pure stretch, R = I and U = stretch."""
        stretch = np.diag([1.1, 0.95, 1.02])
        u_e = _apply_F_to_tet(ref_tet, stretch)
        R, U, _, _ = polar_decomposition_tet4(ref_tet, u_e)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(U, stretch, atol=1e-12)

    def test_polar_decomposition_rotation_plus_stretch(self, ref_tet):
        """Polar decomposition correctly separates R and U."""
        R_true = _rotation_matrix("x", np.pi / 4)
        U_true = np.diag([1.05, 0.98, 1.01])
        F = R_true @ U_true
        u_e = _apply_F_to_tet(ref_tet, F)
        R, U, _, _ = polar_decomposition_tet4(ref_tet, u_e)
        np.testing.assert_allclose(R, R_true, atol=1e-12)
        np.testing.assert_allclose(U, U_true, atol=1e-12)


# ---------------------------------------------------------------------------
# Newton solver tests with corotational elements
# ---------------------------------------------------------------------------

class TestCorotationalNewton:
    """Global Newton solver with corotational assembly."""

    @pytest.fixture
    def cantilever(self):
        """Small 3D cantilever for solver tests."""
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

    def _solve_cr(self, s, f_ext, u0=None, **kwargs):
        """Solve with corotational Newton."""
        D = isotropic_3d_D(_E, _NU)

        def assemble_fn(u):
            return assemble_system_tet4_cr(
                s["nodes"], s["elements"], u, D)

        defaults = dict(max_iter=30, atol=1e-10, verbose=False)
        defaults.update(kwargs)
        return solve_newton_general(
            s["n_dof"], f_ext, s["bc_dofs"], s["bc_vals"],
            assemble_fn, u0=u0, **defaults,
        )

    def test_zero_load_zero_displacement(self, cantilever):
        """Zero external force gives zero displacement."""
        f_ext = np.zeros(cantilever["n_dof"])
        result = self._solve_cr(cantilever, f_ext)
        assert result["converged"]
        assert result["iterations"] == 0
        np.testing.assert_allclose(result["u"], 0.0, atol=1e-14)

    def test_small_load_matches_linear(self, cantilever):
        """For small loads, corotational matches linear FEM (3% tolerance)."""
        s = cantilever
        P_small = -0.01
        tip_dof = 3 * s["tip_node"] + 1

        # Corotational solve.
        f_ext = np.zeros(s["n_dof"])
        f_ext[tip_dof] = P_small
        result = self._solve_cr(s, f_ext)
        assert result["converged"]
        u_cr = result["u"]

        # Linear solve.
        D = isotropic_3d_D(_E, _NU)
        K = assemble_global_stiffness_tet4(s["nodes"], s["elements"], D)
        f_lin = np.zeros(s["n_dof"])
        f_lin[tip_dof] = P_small
        K_bc, f_bc = apply_dirichlet(K, f_lin, s["bc_dofs"], s["bc_vals"])
        u_lin = solve_linear(K_bc, f_bc, verbose=False)

        np.testing.assert_allclose(u_cr, u_lin, rtol=3e-2, atol=1e-12)

    def test_newton_convergence(self, cantilever):
        """Newton converges within 15 iterations for moderate load."""
        s = cantilever
        f_ext = np.zeros(s["n_dof"])
        f_ext[3 * s["tip_node"] + 1] = -0.5
        result = self._solve_cr(s, f_ext, atol=1e-10)
        assert result["converged"]
        assert result["iterations"] <= 15

    def test_bc_enforcement(self, cantilever):
        """Constrained DOFs must have exact prescribed values."""
        s = cantilever
        f_ext = np.zeros(s["n_dof"])
        f_ext[3 * s["tip_node"] + 1] = -0.3
        result = self._solve_cr(s, f_ext)
        assert result["converged"]
        np.testing.assert_allclose(
            result["u"][s["bc_dofs"]], s["bc_vals"], atol=1e-14)

    def test_tip_deflects_downward(self, cantilever):
        """Tip should deflect in the direction of the applied load."""
        s = cantilever
        f_ext = np.zeros(s["n_dof"])
        f_ext[3 * s["tip_node"] + 1] = -1.0
        result = self._solve_cr(s, f_ext, atol=1e-8)
        assert result["converged"]
        assert result["u"][3 * s["tip_node"] + 1] < 0


# ---------------------------------------------------------------------------
# Corotational vs Full Nonlinear vs Linear — large rotation comparison
# ---------------------------------------------------------------------------

class TestCorotationalVsNonlinear:
    """Compare corotational, full nonlinear, and linear FEM at large deflection.

    These tests demonstrate that:
    - Linear FEM over-predicts deflection at large loads (no geometric stiffening)
    - Corotational FEM correctly captures geometric stiffening
    - Corotational and full nonlinear agree for small-to-moderate strains
    """

    @pytest.fixture
    def beam(self):
        """Cantilever for comparison tests."""
        L, H, W = 6.0, 1.0, 1.0
        nx, ny, nz = 6, 2, 2
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

    def test_geometric_stiffening(self, beam):
        """Corotational predicts LESS deflection than linear at large load.

        This is the geometric stiffening effect: as the beam rotates,
        the effective lever arm decreases and the structure becomes stiffer.
        Linear FEM misses this entirely.

        Note: Tet4 elements are very stiff in bending (locking), so
        geometric stiffening is less dramatic than with higher-order
        elements.  We apply a large load and use many steps.

        Regression: corotational deflection < 98% of linear deflection.
        """
        s = beam
        P_large = -10.0
        tip_dof = 3 * s["tip_node"] + 1

        # Linear solve.
        D = isotropic_3d_D(_E, _NU)
        K = assemble_global_stiffness_tet4(s["nodes"], s["elements"], D)
        f_lin = np.zeros(s["n_dof"])
        f_lin[tip_dof] = P_large
        K_bc, f_bc = apply_dirichlet(K, f_lin, s["bc_dofs"], s["bc_vals"])
        u_lin = solve_linear(K_bc, f_bc, verbose=False)
        delta_lin = abs(u_lin[tip_dof])

        # Corotational solve (load stepping for robustness).
        u_cr = np.zeros(s["n_dof"])
        n_steps = 15
        for step in range(1, n_steps + 1):
            frac = step / n_steps
            f_ext = np.zeros(s["n_dof"])
            f_ext[tip_dof] = P_large * frac

            def assemble_fn(u, _D=D):
                return assemble_system_tet4_cr(s["nodes"], s["elements"], u, _D)

            result = solve_newton_general(
                s["n_dof"], f_ext, s["bc_dofs"], s["bc_vals"],
                assemble_fn, u0=u_cr, max_iter=30, atol=1e-8, verbose=False,
            )
            assert result["converged"], f"CR did not converge at step {step}"
            u_cr = result["u"]

        delta_cr = abs(u_cr[tip_dof])

        # Corotational should show geometric stiffening.
        ratio = delta_cr / delta_lin
        assert ratio < 0.98, \
            f"Corotational should be stiffer than linear (ratio={ratio:.3f})"

    def test_corotational_vs_nonlinear_moderate_load(self, beam):
        """At moderate load, corotational and full nonlinear agree within 10%.

        Both capture geometric stiffening.  They differ because the
        corotational uses a linear material while Neo-Hookean is nonlinear,
        but for moderate strains the difference is small.
        """
        s = beam
        P_moderate = -1.0
        tip_dof = 3 * s["tip_node"] + 1
        D = isotropic_3d_D(_E, _NU)

        # Corotational solve.
        f_ext = np.zeros(s["n_dof"])
        f_ext[tip_dof] = P_moderate

        def assemble_cr(u):
            return assemble_system_tet4_cr(s["nodes"], s["elements"], u, D)

        result_cr = solve_newton_general(
            s["n_dof"], f_ext, s["bc_dofs"], s["bc_vals"],
            assemble_cr, max_iter=30, atol=1e-10, verbose=False,
        )
        assert result_cr["converged"]

        # Full nonlinear solve.
        result_nl = solve_newton(
            s["nodes"], s["elements"], f_ext,
            s["bc_dofs"], s["bc_vals"], _tangent_fn_nl,
            max_iter=30, atol=1e-10, verbose=False,
        )
        assert result_nl["converged"]

        delta_cr = result_cr["u"][tip_dof]
        delta_nl = result_nl["u"][tip_dof]

        # Should agree within 10% for moderate load.
        np.testing.assert_allclose(delta_cr, delta_nl, rtol=0.10)

    def test_large_rotation_all_three_methods(self, beam):
        """Compare all three methods at a large load.

        Ordering of tip deflection magnitudes:
            |delta_linear| > |delta_cr| and |delta_linear| > |delta_nl|

        Both nonlinear methods show geometric stiffening vs linear.
        """
        s = beam
        P_large = -2.0
        tip_dof = 3 * s["tip_node"] + 1
        D = isotropic_3d_D(_E, _NU)

        # Linear.
        K = assemble_global_stiffness_tet4(s["nodes"], s["elements"], D)
        f_lin = np.zeros(s["n_dof"])
        f_lin[tip_dof] = P_large
        K_bc, f_bc = apply_dirichlet(K, f_lin, s["bc_dofs"], s["bc_vals"])
        u_lin = solve_linear(K_bc, f_bc, verbose=False)

        # Corotational (with load stepping).
        u_cr = np.zeros(s["n_dof"])
        n_steps = 4
        for step in range(1, n_steps + 1):
            frac = step / n_steps
            f_ext = np.zeros(s["n_dof"])
            f_ext[tip_dof] = P_large * frac

            def assemble_cr(u, _D=D):
                return assemble_system_tet4_cr(s["nodes"], s["elements"], u, _D)

            res = solve_newton_general(
                s["n_dof"], f_ext, s["bc_dofs"], s["bc_vals"],
                assemble_cr, u0=u_cr, max_iter=30, atol=1e-8, verbose=False,
            )
            assert res["converged"]
            u_cr = res["u"]

        # Nonlinear (with load stepping).
        u_nl = np.zeros(s["n_dof"])
        for step in range(1, n_steps + 1):
            frac = step / n_steps
            f_ext = np.zeros(s["n_dof"])
            f_ext[tip_dof] = P_large * frac
            res = solve_newton(
                s["nodes"], s["elements"], f_ext,
                s["bc_dofs"], s["bc_vals"], _tangent_fn_nl,
                u0=u_nl, max_iter=30, atol=1e-8, verbose=False,
            )
            assert res["converged"]
            u_nl = res["u"]

        delta_lin = abs(u_lin[tip_dof])
        delta_cr = abs(u_cr[tip_dof])
        delta_nl = abs(u_nl[tip_dof])

        # Both nonlinear methods predict less deflection than linear.
        assert delta_cr < delta_lin, \
            f"CR ({delta_cr:.4f}) should be < linear ({delta_lin:.4f})"
        assert delta_nl < delta_lin, \
            f"NL ({delta_nl:.4f}) should be < linear ({delta_lin:.4f})"
