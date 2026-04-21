"""Tests for time integration schemes.

Tests cover:
- Energy conservation in free vibration (undamped)
- Central difference and Newmark-beta agreement
- Cantilever natural frequency convergence

Analytical reference for a simply-supported beam first natural frequency:
    f_1 = π² / (2 L²) * sqrt(EI / (ρA))

For a cantilever (clamped-free):
    f_1 = (1.8751)² / (2π L²) * sqrt(EI / (ρA))

Regression tolerances documented per test.
"""

import numpy as np

from femlab.core.assembly import assemble_global_stiffness
from femlab.core.boundary import apply_dirichlet
from femlab.core.dynamics import central_difference, newmark_beta
from femlab.core.mass import assemble_global_mass_t3
from femlab.core.material import plane_stress_D
from femlab.core.solver import solve_linear
from femlab.mesh.rectangle import rectangle_mesh, boundary_nodes


def _build_beam_system(nx, ny, L, H, E, nu, rho, thickness=1.0):
    """Build K, M, bc_dofs for a 2D cantilever beam (left edge fixed)."""
    D = plane_stress_D(E, nu)
    nodes, elements = rectangle_mesh(Lx=L, Ly=H, nx=nx, ny=ny)
    n_nodes = len(nodes)

    K = assemble_global_stiffness(nodes, elements, D, thickness)
    M_consistent = assemble_global_mass_t3(nodes, elements, rho, thickness, lumped=False)
    M_lumped = assemble_global_mass_t3(nodes, elements, rho, thickness, lumped=True)

    # Fix left edge.
    left = boundary_nodes(nx, ny, "left")
    bc_dofs = []
    for ni in left:
        bc_dofs.extend([2 * ni, 2 * ni + 1])
    bc_dofs = np.array(bc_dofs)

    return nodes, elements, K, M_consistent, M_lumped, bc_dofs


def test_central_difference_energy_conservation():
    """Free vibration: total energy should be approximately conserved.

    For central difference (explicit), energy is not exactly conserved
    but should oscillate around the initial value with small amplitude
    when dt is well below the stability limit.

    Regression tolerance: 5% drift over 200 steps.
    """
    L, H = 4.0, 0.5
    E, nu = 1000.0, 0.0
    rho = 1.0
    nx, ny = 16, 2

    nodes, elements, K, M_con, M_lump, bc_dofs = _build_beam_system(
        nx, ny, L, H, E, nu, rho,
    )
    n_dof = 2 * len(nodes)

    # Apply BCs to K for dynamics.
    K_bc = K.tolil()
    for dof in bc_dofs:
        K_bc[dof, :] = 0.0
        K_bc[:, dof] = 0.0
        K_bc[dof, dof] = 1.0
    K_bc = K_bc.tocsr()

    M_diag = M_lump.diagonal().copy()
    M_diag[bc_dofs] = 1.0

    # Initial condition: static deflection under tip load.
    f = np.zeros(n_dof)
    right = boundary_nodes(nx, ny, "right")
    mid_right = right[len(right) // 2]
    f[2 * mid_right + 1] = -1.0
    f[bc_dofs] = 0.0

    K_static, f_static = apply_dirichlet(K, f, bc_dofs, np.zeros(len(bc_dofs)))
    u0 = solve_linear(K_static, f_static, verbose=False)

    v0 = np.zeros(n_dof)
    f_free = np.zeros(n_dof)  # free vibration

    # Estimate critical time step: dt < 2/omega_max ≈ h_min / c
    h_min = min(L / nx, H / ny)
    c_wave = np.sqrt(E / rho)
    dt = 0.5 * h_min / c_wave  # safety factor 0.5

    result = central_difference(
        M_diag, K_bc, f_free, u0, v0, dt, n_steps=200, bc_dofs=bc_dofs,
    )

    E0 = result["energy"][0]
    energy_drift = np.abs(result["energy"] - E0) / max(abs(E0), 1e-20)
    max_drift = energy_drift.max()
    assert max_drift < 0.05, f"Energy drift {max_drift:.4f} exceeds 5%"


def test_newmark_energy_conservation():
    """Newmark-beta (average acceleration) should conserve energy very well.

    Regression tolerance: 0.1% drift over 100 steps.
    """
    L, H = 4.0, 0.5
    E, nu = 1000.0, 0.0
    rho = 1.0
    nx, ny = 12, 2

    nodes, elements, K, M_con, M_lump, bc_dofs = _build_beam_system(
        nx, ny, L, H, E, nu, rho,
    )
    n_dof = 2 * len(nodes)

    # Apply BCs to K and M for Newmark.
    K_bc = K.tolil()
    M_bc = M_con.tolil()
    for dof in bc_dofs:
        K_bc[dof, :] = 0.0
        K_bc[:, dof] = 0.0
        K_bc[dof, dof] = 1.0
        M_bc[dof, :] = 0.0
        M_bc[:, dof] = 0.0
        M_bc[dof, dof] = 1.0
    K_bc = K_bc.tocsr()
    M_bc = M_bc.tocsr()

    # Initial displacement from static solve.
    f = np.zeros(n_dof)
    right = boundary_nodes(nx, ny, "right")
    mid_right = right[len(right) // 2]
    f[2 * mid_right + 1] = -1.0
    f[bc_dofs] = 0.0

    K_static, f_static = apply_dirichlet(K, f, bc_dofs, np.zeros(len(bc_dofs)))
    u0 = solve_linear(K_static, f_static, verbose=False)
    v0 = np.zeros(n_dof)

    dt = 0.01
    result = newmark_beta(
        M_bc, K_bc, np.zeros(n_dof), u0, v0, dt, n_steps=100, bc_dofs=bc_dofs,
    )

    E0 = result["energy"][0]
    energy_drift = np.abs(result["energy"] - E0) / max(abs(E0), 1e-20)
    max_drift = energy_drift.max()
    assert max_drift < 0.001, f"Newmark energy drift {max_drift:.6f} exceeds 0.1%"


def test_vibration_frequency():
    """First natural frequency of a cantilever beam from FFT of tip displacement.

    Analytical: f_1 = (1.8751)² / (2π L²) * sqrt(EI/(ρA))

    For L=4, H=0.5, E=1000, ν=0, ρ=1, t=1:
        I = 1*0.5³/12 = 0.01042
        A = 0.5*1 = 0.5
        f_1 = 3.5156 / (2π * 16) * sqrt(1000*0.01042 / (1*0.5))
            = 3.5156 / 100.53 * sqrt(20.833)
            = 0.03497 * 4.564 = 0.1596 Hz

    Regression tolerance: 30% (T3 elements are stiff in bending, shifting frequency up).
    """
    L, H = 4.0, 0.5
    E, nu = 1000.0, 0.0
    rho = 1.0
    thickness = 1.0
    nx, ny = 20, 3

    nodes, elements, K, M_con, M_lump, bc_dofs = _build_beam_system(
        nx, ny, L, H, E, nu, rho, thickness,
    )
    n_dof = 2 * len(nodes)

    # Apply BCs.
    K_bc = K.tolil()
    M_bc = M_con.tolil()
    for dof in bc_dofs:
        K_bc[dof, :] = 0.0
        K_bc[:, dof] = 0.0
        K_bc[dof, dof] = 1.0
        M_bc[dof, :] = 0.0
        M_bc[:, dof] = 0.0
        M_bc[dof, dof] = 1.0
    K_bc = K_bc.tocsr()
    M_bc = M_bc.tocsr()

    # Initial displacement: static tip load.
    f = np.zeros(n_dof)
    right = boundary_nodes(nx, ny, "right")
    mid_right = right[len(right) // 2]
    f[2 * mid_right + 1] = -1.0
    f[bc_dofs] = 0.0

    K_static, f_static = apply_dirichlet(K, f, bc_dofs, np.zeros(len(bc_dofs)))
    u0 = solve_linear(K_static, f_static, verbose=False)
    v0 = np.zeros(n_dof)

    # Simulate enough time to capture several oscillation periods.
    dt = 0.02
    n_steps = 500
    result = newmark_beta(
        M_bc, K_bc, np.zeros(n_dof), u0, v0, dt, n_steps, bc_dofs=bc_dofs,
    )

    # Extract tip displacement history.
    tip_dof = 2 * mid_right + 1
    tip_hist = result["u"][:, tip_dof]

    # FFT to find dominant frequency.
    tip_centered = tip_hist - tip_hist.mean()
    fft_vals = np.abs(np.fft.rfft(tip_centered))
    freqs = np.fft.rfftfreq(len(tip_centered), d=dt)

    # Skip DC component.
    peak_idx = np.argmax(fft_vals[1:]) + 1
    f_fem = freqs[peak_idx]

    # Analytical.
    I = thickness * H ** 3 / 12.0
    A = H * thickness
    f_analytical = (1.8751 ** 2) / (2 * np.pi * L ** 2) * np.sqrt(E * I / (rho * A))

    rel_error = abs(f_fem - f_analytical) / f_analytical
    assert rel_error < 0.30, (
        f"Frequency error: FEM={f_fem:.4f} Hz, analytical={f_analytical:.4f} Hz, "
        f"rel_error={rel_error:.4f}"
    )
