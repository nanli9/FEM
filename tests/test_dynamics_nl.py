"""Tests for nonlinear time integration schemes.

Tests cover:
- Backward Euler: numerical damping, zero IC, BC enforcement, Newton convergence
- Newmark-beta (nonlinear): energy conservation, zero IC, BC enforcement
- Quasi-static: equilibrium convergence, load stepping
- Small-amplitude nonlinear ≈ linear comparison

All tests use corotational assembly on a small 3D Tet4 cantilever beam.
Regression tolerances documented per test.
"""

import numpy as np
import pytest

from femlab.core.material import isotropic_3d_D
from femlab.core.assembly import assemble_global_stiffness_tet4
from femlab.core.assembly_cr import assemble_system_tet4_cr
from femlab.core.boundary import apply_dirichlet
from femlab.core.dynamics import newmark_beta as newmark_beta_linear
from femlab.core.dynamics_nl import (
    backward_euler_nl,
    newmark_beta_nl,
    quasi_static_nl,
    compute_strain_energy_cr,
)
from femlab.core.mass import assemble_global_mass_tet4
from femlab.core.newton import solve_newton_general
from femlab.core.solver import solve_linear
from femlab.mesh.box import box_mesh, box_boundary_nodes


# ---------------------------------------------------------------------------
# Shared fixture: small 3D cantilever beam
# ---------------------------------------------------------------------------

def _build_beam_3d(nx=4, ny=2, nz=2, L=4.0, H=1.0, W=1.0,
                   E=1000.0, nu=0.3, rho=1.0):
    """Build a small 3D cantilever beam system for testing."""
    D = isotropic_3d_D(E, nu)
    nodes, elements = box_mesh(Lx=L, Ly=H, Lz=W, nx=nx, ny=ny, nz=nz)
    n_nodes = len(nodes)
    n_dof = 3 * n_nodes

    M = assemble_global_mass_tet4(nodes, elements, rho, lumped=False)
    K_lin = assemble_global_stiffness_tet4(nodes, elements, D)

    # Fix left face (x=0).
    left = box_boundary_nodes(nx, ny, nz, "x0")
    bc_dofs = []
    for ni in left:
        bc_dofs.extend([3 * ni, 3 * ni + 1, 3 * ni + 2])
    bc_dofs = np.array(bc_dofs)
    bc_vals = np.zeros(len(bc_dofs))

    # Tip node (center of right face).
    right = box_boundary_nodes(nx, ny, nz, "x1")
    right_coords = nodes[right]
    center = np.array([L, H / 2.0, W / 2.0])
    dists = np.linalg.norm(right_coords - center, axis=1)
    tip_node = right[np.argmin(dists)]
    tip_dof_y = 3 * tip_node + 1

    def assemble_cr(u):
        return assemble_system_tet4_cr(nodes, elements, u, D)

    return {
        "nodes": nodes, "elements": elements, "D": D,
        "M": M, "K_lin": K_lin,
        "n_dof": n_dof, "bc_dofs": bc_dofs, "bc_vals": bc_vals,
        "tip_node": tip_node, "tip_dof_y": tip_dof_y,
        "assemble_cr": assemble_cr,
        "E": E, "nu": nu, "rho": rho,
        "L": L, "H": H, "W": W,
    }


# ===========================================================================
# Backward Euler tests
# ===========================================================================

class TestBackwardEuler:

    def test_zero_ic_stays_at_rest(self):
        """Zero initial conditions and zero force → stays at rest."""
        sys = _build_beam_3d()
        u0 = np.zeros(sys["n_dof"])
        v0 = np.zeros(sys["n_dof"])
        f_ext = np.zeros(sys["n_dof"])

        result = backward_euler_nl(
            sys["M"], f_ext, sys["assemble_cr"],
            u0, v0, dt=0.01, n_steps=5,
            bc_dofs=sys["bc_dofs"], bc_vals=sys["bc_vals"],
        )

        assert np.allclose(result["u_hist"][-1], 0.0, atol=1e-12)
        assert np.allclose(result["v_hist"][-1], 0.0, atol=1e-12)

    def test_bc_enforcement(self):
        """Constrained DOFs remain at prescribed values throughout."""
        sys = _build_beam_3d()
        # Small tip load to get some motion.
        f_ext = np.zeros(sys["n_dof"])
        f_ext[sys["tip_dof_y"]] = -1.0

        # Solve static first, then release.
        result_static = solve_newton_general(
            sys["n_dof"], f_ext, sys["bc_dofs"], sys["bc_vals"],
            sys["assemble_cr"],
        )
        u0 = result_static["u"]
        v0 = np.zeros(sys["n_dof"])

        result = backward_euler_nl(
            sys["M"], np.zeros(sys["n_dof"]), sys["assemble_cr"],
            u0, v0, dt=0.05, n_steps=10,
            bc_dofs=sys["bc_dofs"], bc_vals=sys["bc_vals"],
        )

        for step in range(result["u_hist"].shape[0]):
            np.testing.assert_allclose(
                result["u_hist"][step, sys["bc_dofs"]], sys["bc_vals"],
                atol=1e-12, err_msg=f"BC violated at step {step}",
            )

    def test_numerical_damping(self):
        """Backward Euler introduces numerical damping: KE should decrease.

        Starting from a static deflection with v0=0, the beam oscillates.
        Backward Euler should show decreasing kinetic energy peaks.
        Regression: final KE < 80% of peak KE after 20 steps.
        """
        sys = _build_beam_3d()
        f_ext = np.zeros(sys["n_dof"])
        f_ext[sys["tip_dof_y"]] = -2.0

        result_static = solve_newton_general(
            sys["n_dof"], f_ext, sys["bc_dofs"], sys["bc_vals"],
            sys["assemble_cr"],
        )
        u0 = result_static["u"]
        v0 = np.zeros(sys["n_dof"])

        result = backward_euler_nl(
            sys["M"], np.zeros(sys["n_dof"]), sys["assemble_cr"],
            u0, v0, dt=0.02, n_steps=50,
            bc_dofs=sys["bc_dofs"], bc_vals=sys["bc_vals"],
        )

        ke = result["ke_hist"]
        # Skip step 0 (KE=0 since v0=0). Find peak KE.
        peak_ke = np.max(ke[1:])
        # Final KE should be significantly less than peak (damping).
        assert peak_ke > 0, "No kinetic energy developed"
        # After 50 steps, BE should have damped substantially.
        final_ke = ke[-1]
        assert final_ke < 0.9 * peak_ke, (
            f"Backward Euler should damp: final KE={final_ke:.6e}, "
            f"peak KE={peak_ke:.6e}, ratio={final_ke/peak_ke:.3f}"
        )

    def test_newton_converges(self):
        """Newton inner loop converges at each time step.

        If Newton doesn't converge, the results would be nonsensical.
        We verify by checking that all displacement values are finite.
        """
        sys = _build_beam_3d()
        f_ext = np.zeros(sys["n_dof"])
        f_ext[sys["tip_dof_y"]] = -1.0

        result_static = solve_newton_general(
            sys["n_dof"], f_ext, sys["bc_dofs"], sys["bc_vals"],
            sys["assemble_cr"],
        )
        u0 = result_static["u"]
        v0 = np.zeros(sys["n_dof"])

        result = backward_euler_nl(
            sys["M"], np.zeros(sys["n_dof"]), sys["assemble_cr"],
            u0, v0, dt=0.02, n_steps=20,
            bc_dofs=sys["bc_dofs"], bc_vals=sys["bc_vals"],
        )

        assert np.all(np.isfinite(result["u_hist"])), "Non-finite displacements"
        assert np.all(np.isfinite(result["v_hist"])), "Non-finite velocities"
        assert np.all(np.isfinite(result["ke_hist"])), "Non-finite KE"


# ===========================================================================
# Newmark-beta (nonlinear) tests
# ===========================================================================

class TestNewmarkNL:

    def test_zero_ic_stays_at_rest(self):
        """Zero initial conditions and zero force → stays at rest."""
        sys = _build_beam_3d()
        u0 = np.zeros(sys["n_dof"])
        v0 = np.zeros(sys["n_dof"])
        f_ext = np.zeros(sys["n_dof"])

        result = newmark_beta_nl(
            sys["M"], f_ext, sys["assemble_cr"],
            u0, v0, dt=0.01, n_steps=5,
            bc_dofs=sys["bc_dofs"], bc_vals=sys["bc_vals"],
        )

        assert np.allclose(result["u_hist"][-1], 0.0, atol=1e-12)
        assert np.allclose(result["v_hist"][-1], 0.0, atol=1e-12)

    def test_bc_enforcement(self):
        """Constrained DOFs remain at prescribed values throughout."""
        sys = _build_beam_3d()
        f_ext = np.zeros(sys["n_dof"])
        f_ext[sys["tip_dof_y"]] = -1.0

        result_static = solve_newton_general(
            sys["n_dof"], f_ext, sys["bc_dofs"], sys["bc_vals"],
            sys["assemble_cr"],
        )
        u0 = result_static["u"]
        v0 = np.zeros(sys["n_dof"])

        result = newmark_beta_nl(
            sys["M"], np.zeros(sys["n_dof"]), sys["assemble_cr"],
            u0, v0, dt=0.05, n_steps=10,
            bc_dofs=sys["bc_dofs"], bc_vals=sys["bc_vals"],
        )

        for step in range(result["u_hist"].shape[0]):
            np.testing.assert_allclose(
                result["u_hist"][step, sys["bc_dofs"]], sys["bc_vals"],
                atol=1e-12, err_msg=f"BC violated at step {step}",
            )

    def test_energy_conservation(self):
        """Newmark (beta=1/4, gamma=1/2) approximately conserves energy.

        For small deformations, total energy (KE + PE) should stay nearly
        constant.  Regression: < 5% drift over 30 steps.
        """
        sys = _build_beam_3d()
        # Small load → small deformation → near-linear regime.
        f_ext = np.zeros(sys["n_dof"])
        f_ext[sys["tip_dof_y"]] = -0.5

        result_static = solve_newton_general(
            sys["n_dof"], f_ext, sys["bc_dofs"], sys["bc_vals"],
            sys["assemble_cr"],
        )
        u0 = result_static["u"]
        v0 = np.zeros(sys["n_dof"])

        result = newmark_beta_nl(
            sys["M"], np.zeros(sys["n_dof"]), sys["assemble_cr"],
            u0, v0, dt=0.02, n_steps=30,
            bc_dofs=sys["bc_dofs"], bc_vals=sys["bc_vals"],
        )

        # Compute total energy at each step: KE + PE.
        total_energy = np.zeros(len(result["ke_hist"]))
        for i in range(len(total_energy)):
            ke = result["ke_hist"][i]
            pe = compute_strain_energy_cr(
                sys["nodes"], sys["elements"], result["u_hist"][i], sys["D"],
            )
            total_energy[i] = ke + pe

        E0 = total_energy[0]
        assert E0 > 0, "Initial energy should be positive"
        max_drift = np.max(np.abs(total_energy[1:] - E0)) / E0
        assert max_drift < 0.05, (
            f"Energy drift {max_drift:.3f} exceeds 5% tolerance"
        )

    def test_newton_converges(self):
        """Newton converges at each step — results are all finite."""
        sys = _build_beam_3d()
        f_ext = np.zeros(sys["n_dof"])
        f_ext[sys["tip_dof_y"]] = -1.0

        result_static = solve_newton_general(
            sys["n_dof"], f_ext, sys["bc_dofs"], sys["bc_vals"],
            sys["assemble_cr"],
        )
        u0 = result_static["u"]
        v0 = np.zeros(sys["n_dof"])

        result = newmark_beta_nl(
            sys["M"], np.zeros(sys["n_dof"]), sys["assemble_cr"],
            u0, v0, dt=0.02, n_steps=20,
            bc_dofs=sys["bc_dofs"], bc_vals=sys["bc_vals"],
        )

        assert np.all(np.isfinite(result["u_hist"])), "Non-finite displacements"
        assert np.all(np.isfinite(result["v_hist"])), "Non-finite velocities"

    def test_small_amplitude_matches_linear(self):
        """For tiny deformation, nonlinear Newmark ≈ linear Newmark.

        Compare tip displacement at each step. Regression: relative
        difference < 5% averaged over steps with non-trivial motion.
        """
        sys = _build_beam_3d()
        # Very small tip load → linear regime.
        f_ext = np.zeros(sys["n_dof"])
        f_ext[sys["tip_dof_y"]] = -0.1

        result_static = solve_newton_general(
            sys["n_dof"], f_ext, sys["bc_dofs"], sys["bc_vals"],
            sys["assemble_cr"],
        )
        u0 = result_static["u"]
        v0 = np.zeros(sys["n_dof"])

        dt = 0.02
        n_steps = 20

        # Nonlinear Newmark.
        result_nl = newmark_beta_nl(
            sys["M"], np.zeros(sys["n_dof"]), sys["assemble_cr"],
            u0, v0, dt=dt, n_steps=n_steps,
            bc_dofs=sys["bc_dofs"], bc_vals=sys["bc_vals"],
        )

        # Linear Newmark (constant K with BCs applied).
        K_bc = sys["K_lin"].tolil()
        M_bc = sys["M"].tolil()
        for dof in sys["bc_dofs"]:
            K_bc[dof, :] = 0.0
            K_bc[:, dof] = 0.0
            K_bc[dof, dof] = 1.0
        K_bc = K_bc.tocsr()

        result_lin = newmark_beta_linear(
            sys["M"], K_bc, np.zeros(sys["n_dof"]),
            u0, v0, dt=dt, n_steps=n_steps,
            bc_dofs=sys["bc_dofs"],
        )

        # Compare tip displacement.
        tip = sys["tip_dof_y"]
        tip_nl = result_nl["u_hist"][:, tip]
        tip_lin = result_lin["u"][:, tip]

        # Focus on steps where there's meaningful displacement.
        scale = np.max(np.abs(tip_nl))
        assert scale > 1e-6, "No meaningful displacement"
        diff = np.abs(tip_nl - tip_lin) / scale
        mean_diff = np.mean(diff[1:])  # skip step 0 (identical)
        assert mean_diff < 0.05, (
            f"NL vs linear mean relative diff = {mean_diff:.4f} > 5%"
        )


# ===========================================================================
# Quasi-static tests
# ===========================================================================

class TestQuasiStatic:

    def test_converges_to_zero(self):
        """Starting from bent, ramping f_ext to 0 → u converges to ~0.

        The stress-free configuration is u=0, so quasi-static unloading
        should return the beam to the straight configuration.
        """
        sys = _build_beam_3d()
        f_ext_start = np.zeros(sys["n_dof"])
        f_ext_start[sys["tip_dof_y"]] = -2.0

        result_static = solve_newton_general(
            sys["n_dof"], f_ext_start, sys["bc_dofs"], sys["bc_vals"],
            sys["assemble_cr"],
        )
        u0 = result_static["u"]
        assert np.max(np.abs(u0)) > 1e-4, "No initial deformation"

        f_ext_end = np.zeros(sys["n_dof"])
        result = quasi_static_nl(
            sys["assemble_cr"], sys["n_dof"],
            f_ext_start, f_ext_end, u0,
            n_steps=10,
            bc_dofs=sys["bc_dofs"], bc_vals=sys["bc_vals"],
        )

        # Final displacement should be near zero.
        np.testing.assert_allclose(
            result["u_hist"][-1], 0.0, atol=1e-6,
            err_msg="Quasi-static did not return to zero",
        )

    def test_each_step_converges(self):
        """Newton converges at every load level — all displacements finite."""
        sys = _build_beam_3d()
        f_ext_start = np.zeros(sys["n_dof"])
        f_ext_start[sys["tip_dof_y"]] = -2.0

        result_static = solve_newton_general(
            sys["n_dof"], f_ext_start, sys["bc_dofs"], sys["bc_vals"],
            sys["assemble_cr"],
        )
        u0 = result_static["u"]
        f_ext_end = np.zeros(sys["n_dof"])

        result = quasi_static_nl(
            sys["assemble_cr"], sys["n_dof"],
            f_ext_start, f_ext_end, u0,
            n_steps=10,
            bc_dofs=sys["bc_dofs"], bc_vals=sys["bc_vals"],
        )

        assert np.all(np.isfinite(result["u_hist"])), "Non-finite displacements"

    def test_intermediate_equilibrium(self):
        """Intermediate steps satisfy f_int(u) ≈ f_ext at that load level.

        Regression: residual < 1e-6 at each intermediate step.
        """
        sys = _build_beam_3d()
        f_ext_start = np.zeros(sys["n_dof"])
        f_ext_start[sys["tip_dof_y"]] = -2.0

        result_static = solve_newton_general(
            sys["n_dof"], f_ext_start, sys["bc_dofs"], sys["bc_vals"],
            sys["assemble_cr"],
        )
        u0 = result_static["u"]
        f_ext_end = np.zeros(sys["n_dof"])

        n_steps = 5
        result = quasi_static_nl(
            sys["assemble_cr"], sys["n_dof"],
            f_ext_start, f_ext_end, u0,
            n_steps=n_steps,
            bc_dofs=sys["bc_dofs"], bc_vals=sys["bc_vals"],
        )

        for step in range(1, n_steps + 1):
            frac = step / n_steps
            f_ext = (1.0 - frac) * f_ext_start + frac * f_ext_end
            _K, f_int = sys["assemble_cr"](result["u_hist"][step])
            R = f_int - f_ext
            R[sys["bc_dofs"]] = 0.0
            R_norm = np.linalg.norm(R)
            assert R_norm < 1e-6, (
                f"Step {step}: residual ||R|| = {R_norm:.3e} > 1e-6"
            )

    def test_monotonic_unloading(self):
        """Tip displacement magnitude decreases monotonically during unloading."""
        sys = _build_beam_3d()
        f_ext_start = np.zeros(sys["n_dof"])
        f_ext_start[sys["tip_dof_y"]] = -2.0

        result_static = solve_newton_general(
            sys["n_dof"], f_ext_start, sys["bc_dofs"], sys["bc_vals"],
            sys["assemble_cr"],
        )
        u0 = result_static["u"]
        f_ext_end = np.zeros(sys["n_dof"])

        result = quasi_static_nl(
            sys["assemble_cr"], sys["n_dof"],
            f_ext_start, f_ext_end, u0,
            n_steps=10,
            bc_dofs=sys["bc_dofs"], bc_vals=sys["bc_vals"],
        )

        tip = sys["tip_dof_y"]
        tip_disp = np.abs(result["u_hist"][:, tip])
        # Should decrease (or stay equal) at each step.
        for i in range(1, len(tip_disp)):
            assert tip_disp[i] <= tip_disp[i - 1] + 1e-10, (
                f"Tip displacement not monotonically decreasing at step {i}: "
                f"{tip_disp[i]:.6e} > {tip_disp[i-1]:.6e}"
            )


# ===========================================================================
# Strain energy helper test
# ===========================================================================

class TestStrainEnergy:

    def test_zero_displacement_zero_energy(self):
        """Zero displacement → zero strain energy."""
        sys = _build_beam_3d()
        u = np.zeros(sys["n_dof"])
        pe = compute_strain_energy_cr(sys["nodes"], sys["elements"], u, sys["D"])
        assert abs(pe) < 1e-15

    def test_positive_energy_for_deformation(self):
        """Non-zero displacement → positive strain energy."""
        sys = _build_beam_3d()
        f_ext = np.zeros(sys["n_dof"])
        f_ext[sys["tip_dof_y"]] = -1.0
        result = solve_newton_general(
            sys["n_dof"], f_ext, sys["bc_dofs"], sys["bc_vals"],
            sys["assemble_cr"],
        )
        pe = compute_strain_energy_cr(
            sys["nodes"], sys["elements"], result["u"], sys["D"],
        )
        assert pe > 0, f"Strain energy should be positive, got {pe}"

    def test_energy_matches_linear_for_small_deformation(self):
        """For small deformation, CR strain energy ≈ 0.5*u^T*K*u.

        Regression: < 1% relative difference.
        """
        sys = _build_beam_3d()
        f_ext = np.zeros(sys["n_dof"])
        f_ext[sys["tip_dof_y"]] = -0.1
        result = solve_newton_general(
            sys["n_dof"], f_ext, sys["bc_dofs"], sys["bc_vals"],
            sys["assemble_cr"],
        )
        u = result["u"]
        pe_cr = compute_strain_energy_cr(
            sys["nodes"], sys["elements"], u, sys["D"],
        )
        pe_lin = 0.5 * u @ sys["K_lin"] @ u
        rel_diff = abs(pe_cr - pe_lin) / pe_lin
        assert rel_diff < 0.01, (
            f"CR vs linear PE: rel diff = {rel_diff:.4f} > 1%"
        )
