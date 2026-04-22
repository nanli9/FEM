#!/usr/bin/env python3
"""Nonlinear vibration demo: three time integration methods compared.

Pre-bends a 3D cantilever beam to a large deformation (L-shape) using
corotational FEM, then releases it and simulates the dynamic response
using three methods:

  1. Backward Euler + Newton  — 1st order, numerically dissipative
  2. Newmark-beta + Newton    — 2nd order (beta=1/4, gamma=1/2), energy-conserving
  3. Quasi-static Newton      — no inertia, smooth equilibrium path

Shows a 3-panel animation and tip displacement vs time chart to highlight
how integrator choice affects oscillation and damping behavior.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from femlab.core.material import isotropic_3d_D
from femlab.core.assembly_cr import assemble_system_tet4_cr
from femlab.core.dynamics_nl import (
    backward_euler_nl,
    newmark_beta_nl,
    quasi_static_nl,
)
from femlab.core.mass import assemble_global_mass_tet4
from femlab.core.newton import solve_newton_general
from femlab.mesh.box import box_mesh, box_boundary_nodes


def main():
    print("=" * 70)
    print("FEMLab — Nonlinear Vibration: Three Integration Methods")
    print("  Backward Euler  vs  Newmark-beta  vs  Quasi-static")
    print("=" * 70)

    # --- Problem parameters ---
    L, H, W = 8.0, 1.0, 1.0
    E, nu = 1000.0, 0.3
    rho = 1.0
    nx, ny, nz = 8, 2, 2

    D = isotropic_3d_D(E, nu)
    I_beam = W * H ** 3 / 12.0
    A_beam = H * W

    # --- Mesh ---
    nodes, elements = box_mesh(Lx=L, Ly=H, Lz=W, nx=nx, ny=ny, nz=nz)
    n_nodes = len(nodes)
    n_dof = 3 * n_nodes
    print(f"\nBeam: L={L}, H={H}, W={W}")
    print(f"Material: E={E}, nu={nu}, rho={rho}")
    print(f"Mesh: {nx}x{ny}x{nz} = {len(elements)} Tet4 elements")
    print(f"Nodes: {n_nodes}, DOFs: {n_dof}")

    # --- BCs: fix x=0 face ---
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
    tip_dof = 3 * tip_node + 1  # y-direction
    print(f"Tip node: {tip_node} at {nodes[tip_node]}")

    # Corotational assembly function.
    def assemble_cr(u):
        return assemble_system_tet4_cr(nodes, elements, u, D)

    # Mass matrix.
    M = assemble_global_mass_tet4(nodes, elements, rho, lumped=False)

    # ================================================================
    # Phase 1: Static pre-bend to L-shape
    # ================================================================
    print("\n" + "=" * 70)
    print("Phase 1: Static pre-bend using corotational Newton")
    print("=" * 70)

    # Choose tip force large enough to produce significant bending.
    P_max = -15.0
    n_load_steps = 15

    f_ext_bend = np.zeros(n_dof)
    f_ext_bend[tip_dof] = P_max

    u_bend = np.zeros(n_dof)
    for step in range(1, n_load_steps + 1):
        frac = step / n_load_steps
        f_step = np.zeros(n_dof)
        f_step[tip_dof] = P_max * frac

        result = solve_newton_general(
            n_dof, f_step, bc_dofs, bc_vals,
            assemble_cr, u0=u_bend,
            max_iter=30, atol=1e-8, verbose=False,
        )
        if not result["converged"]:
            print(f"  Load step {step}: Newton did not converge!")
            break
        u_bend = result["u"]
        print(f"  Load step {step}/{n_load_steps}: P={P_max*frac:.2f}, "
              f"tip_y={u_bend[tip_dof]:.4f}, iters={result['iterations']}")

    tip_deflection = u_bend[tip_dof]
    print(f"\nFinal pre-bend tip deflection (y): {tip_deflection:.4f}")
    print(f"Deflection / beam length: {abs(tip_deflection)/L:.2f}")

    # ================================================================
    # Phase 2: Dynamic release — run all three methods
    # ================================================================
    print("\n" + "=" * 70)
    print("Phase 2: Release from bent configuration")
    print("=" * 70)

    # Time stepping parameters.
    # Estimate first natural frequency (linear beam theory).
    f1 = (1.8751 ** 2) / (2 * np.pi * L ** 2) * np.sqrt(E * I_beam / (rho * A_beam))
    T1 = 1.0 / f1
    n_periods = 2
    dt = T1 / 70.0  # 70 steps per period — stable for large deformation
    n_steps = int(n_periods * T1 / dt)

    print(f"Estimated 1st natural period: T1 = {T1:.4f}")
    print(f"Time step: dt = {dt:.4f}")
    print(f"Total steps: {n_steps} ({n_periods} periods)")

    f_ext_zero = np.zeros(n_dof)
    v0 = np.zeros(n_dof)

    # 2a. Backward Euler + Newton.
    print(f"\n--- Backward Euler + Newton ({n_steps} steps) ---")
    res_be = backward_euler_nl(
        M, f_ext_zero, assemble_cr,
        u_bend.copy(), v0.copy(), dt, n_steps,
        bc_dofs=bc_dofs, bc_vals=bc_vals,
        newton_max_iter=30, newton_atol=1e-8,
        verbose=True,
    )

    # 2b. Newmark-beta + Newton.
    print(f"\n--- Newmark-beta + Newton ({n_steps} steps) ---")
    res_nm = newmark_beta_nl(
        M, f_ext_zero, assemble_cr,
        u_bend.copy(), v0.copy(), dt, n_steps,
        bc_dofs=bc_dofs, bc_vals=bc_vals,
        newton_max_iter=30, newton_atol=1e-8,
        verbose=True,
    )

    # 2c. Quasi-static Newton (ramp load from P_bend to 0).
    # Use fewer steps since each QS step is cheap (Newton converges fast).
    n_qs_steps = n_steps
    print(f"\n--- Quasi-static Newton ({n_qs_steps} steps) ---")
    res_qs = quasi_static_nl(
        assemble_cr, n_dof,
        f_ext_bend, f_ext_zero, u_bend.copy(),
        n_steps=n_qs_steps,
        bc_dofs=bc_dofs, bc_vals=bc_vals,
        newton_max_iter=30, newton_atol=1e-8,
        verbose=True,
    )

    # ================================================================
    # Phase 3: Summary
    # ================================================================
    print("\n" + "=" * 70)
    print("Summary: Tip displacement (y) at selected time steps")
    print("=" * 70)

    print(f"{'Step':>6}  {'BE tip_y':>12}  {'NM tip_y':>12}  {'QS tip_y':>12}")
    print("-" * 50)
    sample_steps = np.linspace(0, n_steps, min(11, n_steps + 1), dtype=int)
    for i in sample_steps:
        be_tip = res_be["u_hist"][i, tip_dof]
        nm_tip = res_nm["u_hist"][i, tip_dof]
        qs_i = min(i, len(res_qs["u_hist"]) - 1)
        qs_tip = res_qs["u_hist"][qs_i, tip_dof]
        print(f"{i:6d}  {be_tip:12.4f}  {nm_tip:12.4f}  {qs_tip:12.4f}")

    # ================================================================
    # Phase 4: Visualization
    # ================================================================
    import pyvista as pv

    # --- 3-panel animation ---
    print("\nRendering 3-panel animation (close window to continue)...")

    n_frames = min(200, n_steps + 1)
    frame_idx = np.linspace(0, n_steps, n_frames, dtype=int)

    # Displacement scale for visualization.
    max_tip = max(
        np.max(np.abs(res_be["u_hist"][:, tip_dof])),
        np.max(np.abs(res_nm["u_hist"][:, tip_dof])),
        1e-12,
    )
    vis_scale = 1.0  # 1:1 scale since deformation is already large

    # Global color limits — use same range across all three panels.
    def _max_disp_mag(u_hist, indices):
        ux = u_hist[indices, 0::3]
        uy = u_hist[indices, 1::3]
        uz = u_hist[indices, 2::3]
        return np.max(np.sqrt(ux ** 2 + uy ** 2 + uz ** 2))

    clim_max = max(
        _max_disp_mag(res_be["u_hist"], frame_idx),
        _max_disp_mag(res_nm["u_hist"], frame_idx),
        _max_disp_mag(res_qs["u_hist"],
                      np.clip(frame_idx, 0, len(res_qs["u_hist"]) - 1)),
    )
    clim = [0.0, clim_max]

    # Build VTK cell array for Tet4.
    n_cells = len(elements)
    vtk_cells = np.column_stack(
        [np.full(n_cells, 4, dtype=elements.dtype), elements]
    ).ravel()
    celltypes = np.full(n_cells, pv.CellType.TETRA, dtype=np.uint8)

    def _make_grid():
        g = pv.UnstructuredGrid(vtk_cells.copy(), celltypes.copy(), nodes.copy())
        g.point_data["disp_mag"] = np.zeros(n_nodes)
        return g

    grid_be = _make_grid()
    grid_nm = _make_grid()
    grid_qs = _make_grid()

    pl = pv.Plotter(shape=(1, 3))

    methods = [
        ("Backward Euler", grid_be, res_be),
        ("Newmark b=1/4", grid_nm, res_nm),
        ("Quasi-static", grid_qs, res_qs),
    ]

    texts = []
    for col, (name, grid, _res) in enumerate(methods):
        pl.subplot(0, col)
        # Suppress per-panel scalar bars; add one shared bar below.
        pl.add_mesh(
            grid, scalars="disp_mag", cmap="viridis",
            show_edges=False, clim=clim, name=f"beam_{col}",
            show_scalar_bar=False,
        )
        pl.add_title(name, font_size=10)
        txt = pl.add_text(
            "step 0", position="upper_right", font_size=8,
        )
        texts.append(txt)

    # Add a single shared scalar bar on the last subplot.
    pl.subplot(0, 2)
    pl.add_scalar_bar(
        "Displacement magnitude", vertical=True,
        n_labels=5, fmt="%.2f",
    )

    # Set consistent camera across subplots.
    for col in range(3):
        pl.subplot(0, col)
        pl.camera_position = [
            (L / 2, -L * 1.5, L * 0.8),
            (L / 2, 0.0, W / 2),
            (0, 0, 1),
        ]

    def _update_grid(grid, u_hist, frame_i):
        i = frame_idx[frame_i % n_frames]
        if i >= len(u_hist):
            i = len(u_hist) - 1
        u = u_hist[i]
        ux = u[0::3]
        uy = u[1::3]
        uz = u[2::3]
        deformed = nodes.copy()
        deformed[:, 0] += ux * vis_scale
        deformed[:, 1] += uy * vis_scale
        deformed[:, 2] += uz * vis_scale
        grid.points = deformed
        grid.point_data["disp_mag"] = np.sqrt(ux ** 2 + uy ** 2 + uz ** 2)

    def advance_frame(step):
        fi = step % n_frames
        for col, (name, grid, res) in enumerate(methods):
            _update_grid(grid, res["u_hist"], fi)
            texts[col].SetText(0, f"step {frame_idx[fi]}")

    pl.iren.add_timer_event(
        max_steps=n_frames * 50, duration=33, callback=advance_frame,
    )
    pl.show(title="Three Integration Methods — Corotational FEM")

    # --- Tip displacement vs time chart ---
    print("Showing tip displacement comparison (close window to finish)...")

    chart = pv.Chart2D()
    chart.line(
        res_be["t"], res_be["u_hist"][:, tip_dof],
        label="Backward Euler", color="tab:red", style="--",
    )
    chart.line(
        res_nm["t"], res_nm["u_hist"][:, tip_dof],
        label="Newmark β=1/4", color="tab:blue",
    )
    # Quasi-static: map pseudo-time [0,1] to real time axis.
    qs_t_mapped = res_qs["t"] * res_be["t"][-1]
    chart.line(
        qs_t_mapped, res_qs["u_hist"][:, tip_dof],
        label="Quasi-static", color="tab:green", style="-.",
    )
    chart.x_label = "Time"
    chart.y_label = "Tip displacement (y)"
    chart.title = "Backward Euler vs Newmark vs Quasi-static"
    chart.legend_visible = True

    pl2 = pv.Plotter()
    pl2.add_chart(chart)
    pl2.show(title="Tip Displacement Comparison")

    # --- Kinetic energy chart (dynamic methods only) ---
    print("Showing kinetic energy comparison (close window to finish)...")

    chart_ke = pv.Chart2D()
    chart_ke.line(
        res_be["t"], res_be["ke_hist"],
        label="Backward Euler", color="tab:red", style="--",
    )
    chart_ke.line(
        res_nm["t"], res_nm["ke_hist"],
        label="Newmark β=1/4", color="tab:blue",
    )
    chart_ke.x_label = "Time"
    chart_ke.y_label = "Kinetic Energy"
    chart_ke.title = "Kinetic Energy: Backward Euler vs Newmark"
    chart_ke.legend_visible = True

    pl3 = pv.Plotter()
    pl3.add_chart(chart_ke)
    pl3.show(title="Kinetic Energy Comparison")

    print("\n" + "=" * 70)
    print("Nonlinear vibration demo complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
