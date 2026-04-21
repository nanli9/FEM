#!/usr/bin/env python3
"""Milestone 2 demo: 3D cantilever beam with Tet4 elements.

Solves a 3D cantilever beam under a tip point load and visualizes
the deformed mesh colored by displacement magnitude.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from femlab.core.assembly import assemble_global_stiffness_tet4
from femlab.core.boundary import apply_dirichlet
from femlab.core.material import isotropic_3d_D
from femlab.core.solver import solve_linear
from femlab.mesh.box import box_mesh, box_boundary_nodes
from femlab.viz import show_scalar_field


def main():
    print("=" * 60)
    print("FEMLab Milestone 2 — 3D Cantilever Beam")
    print("=" * 60)

    # --- Problem parameters ---
    L, H, W = 10.0, 1.0, 1.0
    E, nu = 1000.0, 0.3
    P = -1.0  # downward tip load
    nx, ny, nz = 20, 4, 4

    I = W * H ** 3 / 12.0
    delta_exact = P * L ** 3 / (3.0 * E * I)

    print(f"\nBeam: L={L}, H={H}, W={W}")
    print(f"Material: E={E}, ν={nu}")
    print(f"Tip load: P={P} (y-direction)")
    print(f"Mesh: {nx}×{ny}×{nz} = {6*nx*ny*nz} Tet4 elements")
    print(f"Euler-Bernoulli tip deflection: {delta_exact:.6f}")

    # --- Mesh ---
    D = isotropic_3d_D(E, nu)
    nodes, elements = box_mesh(Lx=L, Ly=H, Lz=W, nx=nx, ny=ny, nz=nz)
    n_nodes = len(nodes)
    print(f"Nodes: {n_nodes}, Elements: {len(elements)}")

    # --- Assemble ---
    print("\nAssembling global stiffness...")
    K = assemble_global_stiffness_tet4(nodes, elements, D)
    f = np.zeros(3 * n_nodes)

    # --- Load: point force at center of right face ---
    right = box_boundary_nodes(nx, ny, nz, "x1")
    right_coords = nodes[right]
    center = np.array([L, H / 2.0, W / 2.0])
    dists = np.linalg.norm(right_coords - center, axis=1)
    tip_node = right[np.argmin(dists)]
    f[3 * tip_node + 1] = P  # y-direction load
    print(f"Tip node: {tip_node} at ({nodes[tip_node, 0]:.1f}, "
          f"{nodes[tip_node, 1]:.1f}, {nodes[tip_node, 2]:.1f})")

    # --- BCs: fix x=0 face ---
    left = box_boundary_nodes(nx, ny, nz, "x0")
    bc_dofs, bc_vals = [], []
    for ni in left:
        bc_dofs.extend([3 * ni, 3 * ni + 1, 3 * ni + 2])
        bc_vals.extend([0.0, 0.0, 0.0])

    print("Applying Dirichlet BCs (x=0 face fixed)...")
    K_mod, f_mod = apply_dirichlet(K, f, np.array(bc_dofs), np.array(bc_vals))

    # --- Solve ---
    print("\nSolving...")
    u = solve_linear(K_mod, f_mod, verbose=True)

    delta_fem = u[3 * tip_node + 1]
    rel_error = abs((delta_fem - delta_exact) / delta_exact) * 100
    print(f"\nFEM tip deflection (y):   {delta_fem:.6f}")
    print(f"Analytical (E-B):         {delta_exact:.6f}")
    print(f"Relative error:           {rel_error:.2f}%")

    # --- Visualization ---
    ux = u[0::3]
    uy = u[1::3]
    uz = u[2::3]
    disp_mag = np.sqrt(ux ** 2 + uy ** 2 + uz ** 2)

    scale = 0.2 * L / max(abs(uy).max(), 1e-12)
    deformed = nodes.copy()
    deformed[:, 0] += ux * scale
    deformed[:, 1] += uy * scale
    deformed[:, 2] += uz * scale

    print("\nRendering deformed mesh (close window to finish)...")
    print("  Controls: w=cycle display, e=toggle internal edges, r=reset camera")
    show_scalar_field(
        deformed, elements, disp_mag,
        scalar_name="Displacement magnitude",
        title=f"3D Cantilever (scale={scale:.0f}x)",
        show_edges=False,
    )

    print("\n" + "=" * 60)
    print("Milestone 2 — 3D static demo complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
