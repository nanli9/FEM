#!/usr/bin/env python3
"""Vertical beam hanging under gravity — dynamic simulation.

A 3D beam is fixed at its upper end and released from rest under gravity.
The beam oscillates about the static equilibrium position.

Uses Newmark-beta (average acceleration, unconditionally stable) for
time integration with the linear elastic stiffness and consistent mass.

Press [SPACE] to pause / resume the animation.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from femlab.core.material import isotropic_3d_D
from femlab.core.assembly import assemble_global_stiffness_tet4, assemble_global_force_tet4
from femlab.core.mass import assemble_global_mass_tet4
from femlab.core.dynamics import newmark_beta
from femlab.mesh.box import box_mesh, box_boundary_nodes


def main():
    print("=" * 60)
    print("Vertical Beam — Gravity Drop (Newmark-beta)")
    print("=" * 60)

    # --- Problem parameters ---
    # Beam dimensions: long axis along Y (vertical), cross-section in XZ.
    L = 6.0       # beam length (vertical, y-direction)
    H = 2.0       # cross-section width (x)
    W = 2.0       # cross-section depth (z)
    E = 200.0     # Young's modulus (soft rubber-like, same as stretch demo)
    nu = 0.3
    rho = 1.0     # mass density
    g = 9.81      # gravitational acceleration

    # Mesh divisions.
    # box_mesh builds along [0,Lx] x [0,Ly] x [0,Lz].
    # We want the long axis along y, so Ly = L, Lx = H, Lz = W.
    nx, ny, nz = 5, 16, 5   # 5x16x5 → 2400 tets
    nodes, elements = box_mesh(Lx=H, Ly=L, Lz=W, nx=nx, ny=ny, nz=nz)
    n_nodes = len(nodes)
    n_dof = 3 * n_nodes

    print(f"\nBeam: L={L} (vertical), cross-section {H}x{W}")
    print(f"Material: E={E}, nu={nu}, rho={rho}")
    print(f"Mesh: {nx}x{ny}x{nz} = {len(elements)} Tet4 elements, {n_nodes} nodes")

    # --- Constitutive matrix ---
    D = isotropic_3d_D(E, nu)

    # --- Assembly ---
    print("\nAssembling stiffness, mass, and gravity force...")
    K = assemble_global_stiffness_tet4(nodes, elements, D)
    M = assemble_global_mass_tet4(nodes, elements, rho, lumped=False)

    # Gravity body force: -g in the y-direction.
    # f_ext = integral(rho * g_vec * N dV) assembled consistently.
    body_force = np.array([0.0, -rho * g, 0.0])
    f_ext = assemble_global_force_tet4(nodes, elements, body_force)

    # --- BCs: fix the top face (y = L) ---
    top = box_boundary_nodes(nx, ny, nz, "y1")
    bc_dofs = np.array([d for n in top for d in (3 * n, 3 * n + 1, 3 * n + 2)])

    print(f"Fixed DOFs: {len(bc_dofs)} (top face, {len(top)} nodes)")
    total_mass = rho * H * L * W
    print(f"Total mass: {total_mass:.2f}, total gravity load: {total_mass * g:.2f}")

    # --- Time integration ---
    # Estimate natural period for time step selection.
    # For a cantilever-like beam, f1 ≈ (1.875)^2 / (2π L^2) * sqrt(E*I / (rho*A))
    I_beam = H * W**3 / 12.0
    A_beam = H * W
    omega1_est = (1.875**2) / (L**2) * np.sqrt(E * I_beam / (rho * A_beam))
    T1_est = 2.0 * np.pi / omega1_est
    print(f"\nEstimated first natural period: T1 ≈ {T1_est:.4f} s")

    # Simulate several periods to see oscillation.
    n_periods = 6
    T_total = n_periods * T1_est
    dt = T1_est / 40.0    # ~40 steps per period
    n_steps = int(T_total / dt)
    print(f"Simulation: T_total={T_total:.4f} s, dt={dt:.6f} s, {n_steps} steps")

    # Initial conditions: at rest, undeformed.
    u0 = np.zeros(n_dof)
    v0 = np.zeros(n_dof)

    print("\nRunning Newmark-beta integration...")
    result = newmark_beta(
        M, K, f_ext, u0, v0, dt, n_steps,
        bc_dofs=bc_dofs, verbose=True,
    )

    # --- Pick a tip node for monitoring ---
    bottom = box_boundary_nodes(nx, ny, nz, "y0")
    bottom_coords = nodes[bottom]
    center_xz = np.array([H / 2.0, 0.0, W / 2.0])
    dists = np.linalg.norm(bottom_coords - center_xz, axis=1)
    tip_node = bottom[np.argmin(dists)]
    tip_dof_y = 3 * tip_node + 1
    print(f"\nTip node: {tip_node} at {nodes[tip_node]}")

    # Static deflection for reference.
    # Approximate: delta_static ≈ rho*g*A*L^4 / (8*E*I) for uniform load cantilever
    delta_static = rho * g * A_beam * L**4 / (8.0 * E * I_beam)
    print(f"Estimated static tip deflection: {delta_static:.6f}")

    tip_history = result["u"][:, tip_dof_y]
    print(f"Actual tip y-displacement range: [{tip_history.min():.6f}, {tip_history.max():.6f}]")

    # --- Sample frames for animation ---
    # Take every few steps to get ~80-120 frames.
    frame_stride = max(1, n_steps // 100)
    frame_indices = list(range(0, n_steps + 1, frame_stride))
    if frame_indices[-1] != n_steps:
        frame_indices.append(n_steps)

    print(f"\nAnimation: {len(frame_indices)} frames (stride={frame_stride})")

    # --- Animation ---
    print("\nLaunching animation (close window to exit)...")
    print("  [SPACE] pause / resume")

    import pyvista as pv

    def make_grid(pts, elems):
        cells = np.hstack([np.full((len(elems), 1), 4, dtype=np.int64), elems])
        celltypes = np.full(len(elems), pv.CellType.TETRA, dtype=np.uint8)
        return pv.UnstructuredGrid(cells.ravel(), celltypes, pts)

    # Pre-build grids for sampled frames.
    grids = []
    frame_data_list = []  # (deformed, dmag, t, tip_uy)
    for fi in frame_indices:
        u_snap = result["u"][fi]
        ux = u_snap[0::3]
        uy = u_snap[1::3]
        uz = u_snap[2::3]
        dmag = np.sqrt(ux**2 + uy**2 + uz**2)

        deformed = nodes.copy()
        deformed[:, 0] += ux
        deformed[:, 1] += uy
        deformed[:, 2] += uz

        grid = make_grid(deformed, elements)
        grid.point_data["Displacement"] = dmag
        grids.append(grid)
        frame_data_list.append((deformed, dmag, fi * dt, u_snap[tip_dof_y]))

    # Scalar range.
    vmin = 0.0
    vmax = max(fd[1].max() for fd in frame_data_list)

    pl = pv.Plotter()
    pl.set_background("white")

    mesh_block = grids[0]
    pl.add_mesh(
        mesh_block,
        scalars="Displacement",
        cmap="turbo",
        clim=[vmin, vmax],
        show_edges=True,
        edge_color="gray",
        scalar_bar_args={"title": "  |u|  ", "color": "black"},
    )

    # Undeformed wireframe reference.
    ref_grid = make_grid(nodes, elements)
    pl.add_mesh(ref_grid, style="wireframe", color="black", opacity=0.15)

    # Text annotations.
    TEXT_CORNER = 2  # upper_left
    text_actor = pl.add_text(
        "t=0.0000  tip_uy=0.000000",
        position="upper_left",
        font_size=10,
        color="black",
    )
    pause_actor = pl.add_text(
        "[SPACE] pause/resume",
        position="lower_left",
        font_size=8,
        color="gray",
    )

    # Camera: side view looking at the hanging beam.
    beam_center = [H / 2, L / 2, W / 2]
    pl.camera_position = [
        (beam_center[0] + 12.0, beam_center[1], beam_center[2] + 8.0),
        beam_center,
        (0, 1, 0),
    ]

    # --- Pause / resume ---
    anim_state = {"paused": False, "frame_idx": 0, "forward": True}

    def toggle_pause():
        anim_state["paused"] = not anim_state["paused"]
        status = "PAUSED" if anim_state["paused"] else "PLAYING"
        pause_actor.SetText(0, f"[SPACE] pause/resume  —  {status}")
        pl.render()

    pl.add_key_event("space", toggle_pause)

    # Animation: loop forward then backward (ping-pong), many cycles.
    n_frames = len(grids)
    total_timer_steps = 50 * (2 * (n_frames - 1))
    duration_ms = 50  # fast for smooth dynamic animation

    def update_frame(step):
        if anim_state["paused"]:
            return

        idx = anim_state["frame_idx"]
        _, dmag, t, tip_uy = frame_data_list[idx]

        mesh_block.points = grids[idx].points.copy()
        mesh_block.point_data["Displacement"] = dmag

        text_actor.SetText(
            TEXT_CORNER,
            f"t={t:.4f} s   tip_uy={tip_uy:.6f}   "
            f"static_ref={-delta_static:.6f}",
        )

        # Advance — play forward once, then loop.
        anim_state["frame_idx"] += 1
        if anim_state["frame_idx"] >= n_frames:
            anim_state["frame_idx"] = 0

    pl.add_timer_event(
        max_steps=total_timer_steps,
        duration=duration_ms,
        callback=update_frame,
    )
    pl.show(title="Vertical Beam — Gravity Drop")

    print("\nDone.")


if __name__ == "__main__":
    main()
