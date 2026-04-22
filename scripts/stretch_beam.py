#!/usr/bin/env python3
"""Neo-Hookean beam stretch demo: displacement-controlled tensile test.

A 3D beam is pulled symmetrically from both ends using prescribed
displacements. The deformation gradient, Neo-Hookean stress response,
and geometric nonlinearity are exercised as the stretch grows large.

The script produces an animated PyVista visualization that steps through
load increments, coloring the deformed mesh by displacement magnitude.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from femlab.core.hyperelastic import lame_parameters, neo_hookean_pk1, neo_hookean_tangent
from femlab.core.newton import solve_newton
from femlab.mesh.box import box_mesh, box_boundary_nodes


def main():
    print("=" * 60)
    print("Neo-Hookean Beam Stretch — Symmetric Tension Animation")
    print("=" * 60)

    # --- Problem parameters ---
    L, H, W = 6.0, 2.0, 2.0       # beam dimensions (thicker cross-section)
    E, nu = 200.0, 0.3             # soft rubber-like material
    nx, ny, nz = 16, 5, 5         # mesh divisions (finer mesh)
    max_stretch = 0.5 * L          # total extension per side (50% of half-length)
    n_steps = 40                   # load increments (smoother animation)

    mu, lam = lame_parameters(E, nu)

    def tangent_fn(F):
        P = neo_hookean_pk1(F, mu, lam)
        A = neo_hookean_tangent(F, mu, lam)
        return P, A

    # --- Mesh ---
    nodes, elements = box_mesh(Lx=L, Ly=H, Lz=W, nx=nx, ny=ny, nz=nz)
    n_nodes = len(nodes)
    n_dof = 3 * n_nodes
    print(f"\nBeam: L={L}, H={H}, W={W}")
    print(f"Material: E={E}, nu={nu} (Neo-Hookean)")
    print(f"Mesh: {nx}x{ny}x{nz} = {len(elements)} Tet4 elements, {n_nodes} nodes")

    # --- BCs: pull both ends symmetrically ---
    left = box_boundary_nodes(nx, ny, nz, "x0")
    right = box_boundary_nodes(nx, ny, nz, "x1")

    # Build DOF lists for left and right faces.
    # Left face: x-dof gets -delta, y/z fixed to 0.
    # Right face: x-dof gets +delta, y/z fixed to 0.
    left_x_dofs = np.array([3 * n for n in left])
    left_yz_dofs = np.array([d for n in left for d in (3 * n + 1, 3 * n + 2)])
    right_x_dofs = np.array([3 * n for n in right])
    right_yz_dofs = np.array([d for n in right for d in (3 * n + 1, 3 * n + 2)])

    fixed_dofs = np.concatenate([left_yz_dofs, right_yz_dofs])
    stretch_left_dofs = left_x_dofs
    stretch_right_dofs = right_x_dofs

    bc_dofs_base = np.concatenate([stretch_left_dofs, fixed_dofs, stretch_right_dofs])

    # --- Incremental loading ---
    print(f"\nStretching beam by +/-{max_stretch:.2f} in {n_steps} steps")
    print(f"Total engineering strain: {2 * max_stretch / L:.1%}\n")

    u = np.zeros(n_dof)
    f_ext = np.zeros(n_dof)  # purely displacement-driven

    # Store snapshots for animation.
    snapshots = []  # list of (deformed_nodes, disp_magnitude, delta, reaction)
    # Save undeformed as first frame.
    snapshots.append((nodes.copy(), np.zeros(n_nodes), 0.0, 0.0))

    for step in range(1, n_steps + 1):
        frac = step / n_steps
        delta = max_stretch * frac

        # Prescribed values: left pulls -x, right pulls +x, y/z = 0.
        bc_vals = np.concatenate([
            np.full(len(stretch_left_dofs), -delta),   # left x
            np.zeros(len(fixed_dofs)),                  # y/z on both faces
            np.full(len(stretch_right_dofs), +delta),   # right x
        ])

        print(f"--- Step {step:2d}/{n_steps}: delta = +/-{delta:.4f} "
              f"(eng. strain = {2 * delta / L:.1%}) ---")

        result = solve_newton(
            nodes, elements, f_ext, bc_dofs_base, bc_vals, tangent_fn,
            u0=u, max_iter=30, atol=1e-8, verbose=True,
        )

        if not result["converged"]:
            print("Newton did not converge — stopping load stepping.")
            break

        u = result["u"]

        # Compute deformed config and displacement magnitude.
        ux = u[0::3]
        uy = u[1::3]
        uz = u[2::3]
        disp_mag = np.sqrt(ux**2 + uy**2 + uz**2)

        deformed = nodes.copy()
        deformed[:, 0] += ux
        deformed[:, 1] += uy
        deformed[:, 2] += uz

        # Estimate reaction force on right face.
        from femlab.core.assembly_nl import assemble_internal_force_tet4_nl

        def pk1_fn(F):
            return neo_hookean_pk1(F, mu, lam)

        f_int = assemble_internal_force_tet4_nl(nodes, elements, u, pk1_fn)
        reaction = np.sum(f_int[stretch_right_dofs])

        snapshots.append((deformed.copy(), disp_mag.copy(), delta, reaction))
        print(f"  Max |u| = {disp_mag.max():.6f}, reaction Fx(right) = {reaction:.4f}\n")

    # --- Print summary ---
    print("=" * 60)
    print("Step  delta      strain     max|u|     Fx(right)")
    print("-" * 60)
    for i, (_, dmag, d, rxn) in enumerate(snapshots):
        strain = 2 * d / L
        print(f" {i:3d}   {d:8.4f}   {strain:8.1%}   {dmag.max():10.6f}   {rxn:10.4f}")

    # --- Animation ---
    print("\nLaunching animation (close window to exit)...")
    print("  [SPACE] pause / resume")

    import pyvista as pv

    # Build unstructured grid template.
    def make_grid(pts, elems):
        cells = np.hstack([np.full((len(elems), 1), 4, dtype=np.int64), elems])
        celltypes = np.full(len(elems), pv.CellType.TETRA, dtype=np.uint8)
        return pv.UnstructuredGrid(cells.ravel(), celltypes, pts)

    # Pre-build grids for each frame.
    grids = []
    for deformed, dmag, delta, _ in snapshots:
        grid = make_grid(deformed, elements)
        grid.point_data["Displacement"] = dmag
        grids.append(grid)

    # Global scalar range for consistent colorbar.
    vmin = 0.0
    vmax = max(g.point_data["Displacement"].max() for g in grids)

    pl = pv.Plotter()
    pl.set_background("white")

    # Add first frame.
    mesh_block = grids[0]
    actor = pl.add_mesh(
        mesh_block,
        scalars="Displacement",
        cmap="turbo",
        clim=[vmin, vmax],
        show_edges=True,
        edge_color="gray",
        scalar_bar_args={"title": "  |u|  ", "color": "black"},
    )

    # Add undeformed wireframe for reference.
    ref_grid = make_grid(nodes, elements)
    pl.add_mesh(ref_grid, style="wireframe", color="black", opacity=0.15)

    # Text annotations.
    # CornerAnnotation corners: 0=lower_left, 1=lower_right, 2=upper_left, 3=upper_right.
    TEXT_CORNER = 2
    text_actor = pl.add_text(
        "Step 0 / {} — delta=0.0000".format(n_steps),
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

    # Camera: look along z toward beam center.
    center = [L / 2, H / 2, W / 2]
    pl.camera_position = [
        (center[0], center[1] - 8.0, center[2] + 12.0),
        center,
        (0, 1, 0),
    ]

    # --- Pause / resume state ---
    anim_state = {"paused": False, "frame_idx": 0, "forward": True}

    def toggle_pause():
        anim_state["paused"] = not anim_state["paused"]
        status = "PAUSED" if anim_state["paused"] else "PLAYING"
        pause_actor.SetText(0, f"[SPACE] pause/resume  —  {status}")
        pl.render()

    pl.add_key_event("space", toggle_pause)

    # Animation: bounce back and forth through snapshots, loop forever.
    n_frames = len(snapshots)
    # Use a large max_steps so animation keeps going; user closes window to stop.
    total_steps = 50 * (2 * (n_frames - 1))
    duration_ms = 100  # faster frame rate for smoother animation

    def update_frame(step):
        if anim_state["paused"]:
            return

        idx = anim_state["frame_idx"]
        deformed, dmag, delta, rxn = snapshots[idx]

        # Update mesh geometry and scalars in-place.
        mesh_block.points = grids[idx].points.copy()
        mesh_block.point_data["Displacement"] = dmag

        strain = 2 * delta / L
        text_actor.SetText(
            TEXT_CORNER,
            f"Step {idx}/{n_steps}   delta={delta:.4f}   "
            f"strain={strain:.0%}   Fx={rxn:.2f}",
        )

        # Advance frame (ping-pong).
        if anim_state["forward"]:
            anim_state["frame_idx"] += 1
            if anim_state["frame_idx"] >= n_frames:
                anim_state["frame_idx"] = n_frames - 2
                anim_state["forward"] = False
        else:
            anim_state["frame_idx"] -= 1
            if anim_state["frame_idx"] < 0:
                anim_state["frame_idx"] = 1
                anim_state["forward"] = True

    pl.add_timer_event(
        max_steps=total_steps,
        duration=duration_ms,
        callback=update_frame,
    )
    pl.show(title="Neo-Hookean Beam Stretch — Symmetric Tension")

    print("\nDone.")


if __name__ == "__main__":
    main()
