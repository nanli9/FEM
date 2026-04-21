#!/usr/bin/env python3
"""Milestone 1 demo: 2D cantilever beam with T3 elements.

Solves a cantilever beam under a tip point load and visualizes:
    1. Deformed mesh with displacement magnitude
    2. Stress field (σxx)

Compares FEM tip deflection against Euler-Bernoulli beam theory.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from femlab.core.assembly import assemble_global_stiffness
from femlab.core.boundary import apply_dirichlet
from femlab.core.element import t3_stress
from femlab.core.material import plane_stress_D
from femlab.core.solver import solve_linear
from femlab.mesh.rectangle import rectangle_mesh, boundary_nodes
from femlab.viz import show_scalar_field


def main():
    print("=" * 60)
    print("FEMLab Milestone 1 — 2D Cantilever Beam")
    print("=" * 60)

    # --- Problem parameters ---
    L, H = 10.0, 1.0       # beam length and height
    E, nu = 1000.0, 0.3    # Young's modulus and Poisson's ratio
    P = -1.0                # tip load (downward)
    thickness = 1.0
    nx, ny = 40, 8          # mesh resolution

    I = thickness * H ** 3 / 12.0
    delta_exact = P * L ** 3 / (3.0 * E * I)

    print(f"\nBeam: L={L}, H={H}, E={E}, ν={nu}")
    print(f"Tip load: P={P}")
    print(f"Mesh: {nx}×{ny} = {2*nx*ny} T3 elements")
    print(f"Euler-Bernoulli tip deflection: {delta_exact:.6f}")

    # --- Mesh ---
    D = plane_stress_D(E, nu)
    nodes, elements = rectangle_mesh(Lx=L, Ly=H, nx=nx, ny=ny)
    n_nodes = len(nodes)
    print(f"Nodes: {n_nodes}, Elements: {len(elements)}")

    # --- Assemble ---
    print("\nAssembling global stiffness...")
    K = assemble_global_stiffness(nodes, elements, D, thickness)
    f = np.zeros(2 * n_nodes)

    # --- Load: point force at mid-height of right edge ---
    right = boundary_nodes(nx, ny, "right")
    right_coords = nodes[right]
    mid_idx = np.argmin(np.abs(right_coords[:, 1] - H / 2.0))
    tip_node = right[mid_idx]
    f[2 * tip_node + 1] = P
    print(f"Tip node: {tip_node} at ({nodes[tip_node, 0]:.1f}, {nodes[tip_node, 1]:.1f})")

    # --- BCs: fix left edge ---
    left = boundary_nodes(nx, ny, "left")
    bc_dofs, bc_vals = [], []
    for ni in left:
        bc_dofs.extend([2 * ni, 2 * ni + 1])
        bc_vals.extend([0.0, 0.0])

    print("Applying Dirichlet BCs (left edge fixed)...")
    K_mod, f_mod = apply_dirichlet(K, f, np.array(bc_dofs), np.array(bc_vals))

    # --- Solve ---
    print("\nSolving...")
    u = solve_linear(K_mod, f_mod, verbose=True)

    delta_fem = u[2 * tip_node + 1]
    rel_error = abs((delta_fem - delta_exact) / delta_exact) * 100
    print(f"\nFEM tip deflection:   {delta_fem:.6f}")
    print(f"Analytical (E-B):     {delta_exact:.6f}")
    print(f"Relative error:       {rel_error:.2f}%")

    # --- Post-process: displacement magnitude ---
    ux = u[0::2]
    uy = u[1::2]
    disp_mag = np.sqrt(ux ** 2 + uy ** 2)

    # Deformed coordinates (scaled for visibility).
    scale = 0.2 * L / max(abs(uy).max(), 1e-12)
    deformed = nodes.copy()
    deformed[:, 0] += ux * scale
    deformed[:, 1] += uy * scale

    # --- Post-process: stress σxx per element → averaged to nodes ---
    stress_xx_elem = np.zeros(len(elements))
    for e in range(len(elements)):
        n0, n1, n2 = elements[e]
        coords_e = nodes[[n0, n1, n2]]
        u_e = np.array([u[2*n0], u[2*n0+1], u[2*n1], u[2*n1+1], u[2*n2], u[2*n2+1]])
        _, stress = t3_stress(coords_e, D, u_e)
        stress_xx_elem[e] = stress[0]

    # Average element stresses to nodes.
    stress_xx_nodal = np.zeros(n_nodes)
    count = np.zeros(n_nodes)
    for e in range(len(elements)):
        for ni in elements[e]:
            stress_xx_nodal[ni] += stress_xx_elem[e]
            count[ni] += 1
    stress_xx_nodal /= np.maximum(count, 1)

    # --- Visualize ---
    print("\nRendering deformed mesh (close window to continue)...")
    show_scalar_field(
        deformed, elements, disp_mag,
        scalar_name="Displacement magnitude",
        title=f"Cantilever: Deformed shape (scale={scale:.0f}x)",
    )

    print("Rendering stress field (close window to finish)...")
    show_scalar_field(
        nodes, elements, stress_xx_nodal,
        scalar_name="σxx",
        title="Cantilever: Bending stress σxx",
    )

    print("\n" + "=" * 60)
    print("Milestone 1 demo complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
