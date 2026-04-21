#!/usr/bin/env python3
"""Milestone 2 demo: free vibration of a 3D cantilever beam (Tet4).

Displaces the beam statically under a tip load, then releases it
and simulates free vibration using both central difference (explicit)
and Newmark-beta (implicit) integrators.  Shows an animated 3D mesh
and plots tip displacement / energy history.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from femlab.core.assembly import assemble_global_stiffness_tet4
from femlab.core.boundary import apply_dirichlet
from femlab.core.dynamics import central_difference, newmark_beta
from femlab.core.mass import assemble_global_mass_tet4
from femlab.core.material import isotropic_3d_D
from femlab.core.solver import solve_linear
from femlab.mesh.box import box_mesh, box_boundary_nodes


def main():
    print("=" * 60)
    print("FEMLab Milestone 2 — 3D Beam Free Vibration")
    print("=" * 60)

    # --- Problem parameters ---
    L, H, W = 10.0, 1.0, 1.0
    E, nu = 1000.0, 0.3
    rho = 1.0
    nx, ny, nz = 16, 3, 3

    I = W * H ** 3 / 12.0
    A = H * W
    f1_analytical = (1.8751 ** 2) / (2 * np.pi * L ** 2) * np.sqrt(E * I / (rho * A))
    T1 = 1.0 / f1_analytical

    print(f"\nBeam: L={L}, H={H}, W={W}")
    print(f"Material: E={E}, ν={nu}, ρ={rho}")
    print(f"Mesh: {nx}×{ny}×{nz} = {6*nx*ny*nz} Tet4 elements")
    print(f"Analytical 1st natural frequency: {f1_analytical:.4f} Hz")
    print(f"Analytical 1st period: {T1:.4f} s")

    # --- Build system ---
    D = isotropic_3d_D(E, nu)
    nodes, elements = box_mesh(Lx=L, Ly=H, Lz=W, nx=nx, ny=ny, nz=nz)
    n_nodes = len(nodes)
    n_dof = 3 * n_nodes

    print(f"Nodes: {n_nodes}, Elements: {len(elements)}, DOFs: {n_dof}")

    print("\nAssembling stiffness...")
    K = assemble_global_stiffness_tet4(nodes, elements, D)
    print("Assembling mass (consistent)...")
    M_con = assemble_global_mass_tet4(nodes, elements, rho, lumped=False)
    print("Assembling mass (lumped)...")
    M_lump = assemble_global_mass_tet4(nodes, elements, rho, lumped=True)

    # Fix x=0 face.
    left = box_boundary_nodes(nx, ny, nz, "x0")
    bc_dofs = []
    for ni in left:
        bc_dofs.extend([3 * ni, 3 * ni + 1, 3 * ni + 2])
    bc_dofs = np.array(bc_dofs)

    # --- Static initial condition: tip load at center of x=L face ---
    right = box_boundary_nodes(nx, ny, nz, "x1")
    right_coords = nodes[right]
    center = np.array([L, H / 2.0, W / 2.0])
    dists = np.linalg.norm(right_coords - center, axis=1)
    tip_node = right[np.argmin(dists)]
    tip_dof = 3 * tip_node + 1  # y-direction

    f_static = np.zeros(n_dof)
    f_static[tip_dof] = -1.0
    K_bc, f_bc = apply_dirichlet(K, f_static, bc_dofs, np.zeros(len(bc_dofs)))
    print("\nSolving static initial condition...")
    u0 = solve_linear(K_bc, f_bc, verbose=False)
    v0 = np.zeros(n_dof)

    print(f"Static tip deflection (y): {u0[tip_dof]:.6f}")

    # --- Apply BCs for dynamics ---
    K_dyn = K.tolil()
    M_con_dyn = M_con.tolil()
    for dof in bc_dofs:
        K_dyn[dof, :] = 0.0
        K_dyn[:, dof] = 0.0
        K_dyn[dof, dof] = 1.0
        M_con_dyn[dof, :] = 0.0
        M_con_dyn[:, dof] = 0.0
        M_con_dyn[dof, dof] = 1.0
    K_dyn = K_dyn.tocsr()
    M_con_dyn = M_con_dyn.tocsr()

    M_diag = M_lump.diagonal().copy()
    M_diag[bc_dofs] = 1.0

    f_free = np.zeros(n_dof)

    # --- Central difference (explicit) ---
    h_min = min(L / nx, H / ny, W / nz)
    c_wave = np.sqrt(E / rho)
    dt_explicit = 0.5 * h_min / c_wave
    n_periods = 3
    n_steps_exp = int(n_periods * T1 / dt_explicit)

    print(f"\n--- Central Difference (explicit) ---")
    print(f"dt = {dt_explicit:.6f}, steps = {n_steps_exp}")
    res_cd = central_difference(
        M_diag, K_dyn, f_free, u0, v0, dt_explicit, n_steps_exp,
        bc_dofs=bc_dofs, verbose=True,
    )
    E0_cd = res_cd["energy"][0]
    drift_cd = np.abs(res_cd["energy"] - E0_cd) / abs(E0_cd)
    print(f"Max energy drift: {drift_cd.max():.6e}")

    # --- Newmark-beta (implicit) ---
    dt_implicit = T1 / 50  # 50 steps per period
    n_steps_nm = int(n_periods * T1 / dt_implicit)

    print(f"\n--- Newmark-beta (implicit, β=1/4, γ=1/2) ---")
    print(f"dt = {dt_implicit:.6f}, steps = {n_steps_nm}")
    res_nm = newmark_beta(
        M_con_dyn, K_dyn, f_free, u0, v0, dt_implicit, n_steps_nm,
        bc_dofs=bc_dofs, verbose=True,
    )
    E0_nm = res_nm["energy"][0]
    drift_nm = np.abs(res_nm["energy"] - E0_nm) / abs(E0_nm)
    print(f"Max energy drift: {drift_nm.max():.6e}")

    # --- FFT for frequency ---
    tip_cd = res_cd["u"][:, tip_dof]
    tip_nm = res_nm["u"][:, tip_dof]

    fft_cd = np.abs(np.fft.rfft(tip_cd - tip_cd.mean()))
    freqs_cd = np.fft.rfftfreq(len(tip_cd), d=dt_explicit)
    peak_cd = np.argmax(fft_cd[1:]) + 1
    f_cd = freqs_cd[peak_cd]

    fft_nm = np.abs(np.fft.rfft(tip_nm - tip_nm.mean()))
    freqs_nm = np.fft.rfftfreq(len(tip_nm), d=dt_implicit)
    peak_nm = np.argmax(fft_nm[1:]) + 1
    f_nm = freqs_nm[peak_nm]

    print(f"\n--- Frequency comparison ---")
    print(f"Analytical f1:          {f1_analytical:.4f} Hz")
    print(f"Central difference f1:  {f_cd:.4f} Hz  "
          f"(error: {abs(f_cd - f1_analytical)/f1_analytical*100:.1f}%)")
    print(f"Newmark-beta f1:        {f_nm:.4f} Hz  "
          f"(error: {abs(f_nm - f1_analytical)/f1_analytical*100:.1f}%)")

    # --- Animated 3D mesh visualization ---
    import pyvista as pv

    print("\nRendering 3D vibration animation (close window to continue to plots)...")

    # Use central difference results (finer time resolution).
    u_hist = res_cd["u"]
    t_hist = res_cd["t"]

    # Subsample to ~300 frames for smooth playback.
    n_total = len(t_hist)
    n_frames = min(300, n_total)
    frame_idx = np.linspace(0, n_total - 1, n_frames, dtype=int)

    # Displacement scale so the motion is clearly visible.
    max_tip = np.max(np.abs(u_hist[:, tip_dof]))
    vis_scale = 0.2 * L / max(max_tip, 1e-12)

    # Compute global color limits across all frames.
    all_ux = u_hist[frame_idx, 0::3]
    all_uy = u_hist[frame_idx, 1::3]
    all_uz = u_hist[frame_idx, 2::3]
    all_mag = np.sqrt(all_ux ** 2 + all_uy ** 2 + all_uz ** 2)
    clim = [0.0, all_mag.max()]

    # Build PyVista mesh (Tet4).
    n_cells = len(elements)
    k = elements.shape[1]  # 4
    vtk_cells = np.column_stack(
        [np.full(n_cells, k, dtype=elements.dtype), elements]
    ).ravel()
    celltypes = np.full(n_cells, pv.CellType.TETRA, dtype=np.uint8)
    grid = pv.UnstructuredGrid(vtk_cells, celltypes, nodes.copy())
    grid.point_data["disp_mag"] = np.zeros(n_nodes)

    pl = pv.Plotter()
    pl.add_mesh(
        grid, scalars="disp_mag", cmap="viridis",
        show_edges=False, clim=clim, name="beam",
    )
    pl.add_scalar_bar("Displacement magnitude")
    time_text = pl.add_text(
        f"t = {t_hist[0]:.4f}", position="upper_right", font_size=10,
    )
    pl.add_title("3D Free Vibration (central difference)")

    def advance_frame(step):
        fi = step % n_frames
        i = frame_idx[fi]
        ux = u_hist[i, 0::3]
        uy = u_hist[i, 1::3]
        uz = u_hist[i, 2::3]
        deformed = nodes.copy()
        deformed[:, 0] += ux * vis_scale
        deformed[:, 1] += uy * vis_scale
        deformed[:, 2] += uz * vis_scale
        grid.points = deformed
        grid.point_data["disp_mag"] = np.sqrt(ux ** 2 + uy ** 2 + uz ** 2)
        time_text.SetText(0, f"t = {t_hist[i]:.4f}")

    # Loop many times (~100 loops at 30 fps).
    pl.iren.add_timer_event(
        max_steps=n_frames * 100, duration=33, callback=advance_frame,
    )
    pl.show()

    # --- Plots with PyVista Chart2D ---
    print("Showing tip displacement plot (close window to continue)...")

    # Chart 1: Tip displacement
    chart1 = pv.Chart2D()
    chart1.line(res_cd["t"], res_cd["u"][:, tip_dof], label="Central diff", color="tab:blue")
    chart1.line(res_nm["t"], res_nm["u"][:, tip_dof], label="Newmark", color="tab:orange")
    chart1.x_label = "Time"
    chart1.y_label = "Tip displacement (y)"
    chart1.title = "Free vibration — tip displacement"
    chart1.legend_visible = True

    pl1 = pv.Plotter()
    pl1.add_chart(chart1)
    pl1.show(title="Tip Displacement")

    # Chart 2: Energy conservation
    print("Showing energy conservation plot (close window to finish)...")
    chart2 = pv.Chart2D()
    chart2.line(res_cd["t"], res_cd["energy"] / E0_cd, label="Central diff", color="tab:blue")
    chart2.line(res_nm["t"], res_nm["energy"] / E0_nm, label="Newmark", color="tab:orange")
    chart2.x_label = "Time"
    chart2.y_label = "E / E₀"
    chart2.title = "Energy conservation"
    chart2.legend_visible = True

    pl2 = pv.Plotter()
    pl2.add_chart(chart2)
    pl2.show(title="Energy Conservation")

    print("\n" + "=" * 60)
    print("Milestone 2 — 3D vibration demo complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
