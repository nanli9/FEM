#!/usr/bin/env python3
"""Milestone 4 demo: Corotational FEM vs Linear vs Full Nonlinear.

Solves a 3D cantilever beam under progressively increasing tip load
using three methods:
  1. Linear FEM (single solve per load level)
  2. Corotational FEM (Newton iteration with polar decomposition)
  3. Full nonlinear FEM (Newton iteration with Neo-Hookean material)

Shows load-displacement curves and deformed meshes to demonstrate:
  - Linear FEM over-predicts deflection (no geometric stiffening)
  - Corotational and full nonlinear capture geometric stiffening
  - Corotational is cheaper (reuses linear K) but agrees closely
    with full nonlinear for moderate strains
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from femlab.core.material import isotropic_3d_D
from femlab.core.assembly import assemble_global_stiffness_tet4
from femlab.core.assembly_cr import assemble_system_tet4_cr
from femlab.core.boundary import apply_dirichlet
from femlab.core.solver import solve_linear
from femlab.core.newton import solve_newton, solve_newton_general
from femlab.core.hyperelastic import lame_parameters, neo_hookean_pk1, neo_hookean_tangent
from femlab.mesh.box import box_mesh, box_boundary_nodes
from femlab.viz import show_scalar_field


def main():
    print("=" * 65)
    print("FEMLab Milestone 4 — Corotational FEM Comparison")
    print("  Linear  vs  Corotational  vs  Full Nonlinear (Neo-Hookean)")
    print("=" * 65)

    # --- Problem parameters ---
    L, H, W = 10.0, 1.0, 1.0
    E, nu = 1000.0, 0.3
    nx, ny, nz = 10, 2, 2
    P_max = -5.0
    n_load_steps = 10

    D = isotropic_3d_D(E, nu)
    mu, lam = lame_parameters(E, nu)
    I_beam = W * H ** 3 / 12.0

    def tangent_fn_nl(F):
        P = neo_hookean_pk1(F, mu, lam)
        A = neo_hookean_tangent(F, mu, lam)
        return P, A

    # --- Mesh ---
    nodes, elements = box_mesh(Lx=L, Ly=H, Lz=W, nx=nx, ny=ny, nz=nz)
    n_nodes = len(nodes)
    n_dof = 3 * n_nodes
    print(f"\nBeam: L={L}, H={H}, W={W}")
    print(f"Material: E={E}, nu={nu}")
    print(f"Mesh: {nx}x{ny}x{nz} = {len(elements)} Tet4 elements")
    print(f"Nodes: {n_nodes}, DOFs: {n_dof}")

    # --- BCs: fix x=0 face ---
    left = box_boundary_nodes(nx, ny, nz, "x0")
    bc_dofs = []
    for ni in left:
        bc_dofs.extend([3 * ni, 3 * ni + 1, 3 * ni + 2])
    bc_dofs = np.array(bc_dofs)
    bc_vals = np.zeros(len(bc_dofs))

    # Tip node.
    right = box_boundary_nodes(nx, ny, nz, "x1")
    right_coords = nodes[right]
    center = np.array([L, H / 2.0, W / 2.0])
    dists = np.linalg.norm(right_coords - center, axis=1)
    tip_node = right[np.argmin(dists)]
    tip_dof = 3 * tip_node + 1
    print(f"Tip node: {tip_node} at {nodes[tip_node]}")

    # Pre-assemble linear stiffness (only done once).
    K_lin_global = assemble_global_stiffness_tet4(nodes, elements, D)

    # --- Load stepping ---
    print(f"\nLoad stepping: {n_load_steps} increments, P_max = {P_max}")
    print(f"Linear tip deflection at P_max: {P_max * L**3 / (3*E*I_beam):.6f}")

    loads = []
    tip_linear = []
    tip_cr = []
    tip_nl = []

    u_cr = np.zeros(n_dof)
    u_nl = np.zeros(n_dof)

    for step in range(1, n_load_steps + 1):
        frac = step / n_load_steps
        P_current = P_max * frac

        f_ext = np.zeros(n_dof)
        f_ext[tip_dof] = P_current

        print(f"\n--- Load step {step}/{n_load_steps}: P = {P_current:.4f} ---")

        # 1. Linear (direct solve).
        K_bc, f_bc = apply_dirichlet(K_lin_global, f_ext.copy(), bc_dofs, bc_vals)
        u_lin = solve_linear(K_bc, f_bc, verbose=False)
        delta_lin = u_lin[tip_dof]

        # 2. Corotational Newton.
        def assemble_cr(u, _D=D):
            return assemble_system_tet4_cr(nodes, elements, u, _D)

        print("  [Corotational]")
        result_cr = solve_newton_general(
            n_dof, f_ext, bc_dofs, bc_vals,
            assemble_cr, u0=u_cr, max_iter=30, atol=1e-8, verbose=True,
        )
        if not result_cr["converged"]:
            print("  CR Newton did not converge! Stopping.")
            break
        u_cr = result_cr["u"]
        delta_cr = u_cr[tip_dof]

        # 3. Full nonlinear Newton.
        print("  [Full nonlinear (Neo-Hookean)]")
        result_nl = solve_newton(
            nodes, elements, f_ext, bc_dofs, bc_vals, tangent_fn_nl,
            u0=u_nl, max_iter=30, atol=1e-8, verbose=True,
        )
        if not result_nl["converged"]:
            print("  NL Newton did not converge! Stopping.")
            break
        u_nl = result_nl["u"]
        delta_nl = u_nl[tip_dof]

        loads.append(abs(P_current))
        tip_linear.append(abs(delta_lin))
        tip_cr.append(abs(delta_cr))
        tip_nl.append(abs(delta_nl))

        print(f"  Tip deflection (linear):       {delta_lin:.6f}")
        print(f"  Tip deflection (corotational):  {delta_cr:.6f}")
        print(f"  Tip deflection (nonlinear):     {delta_nl:.6f}")
        if abs(delta_lin) > 0:
            print(f"  Ratio CR/linear:  {abs(delta_cr)/abs(delta_lin):.4f}")
            print(f"  Ratio NL/linear:  {abs(delta_nl)/abs(delta_lin):.4f}")

    # --- Summary ---
    print("\n" + "=" * 65)
    print("Load-displacement summary:")
    print(f"  {'Load':>8}  {'Linear':>12}  {'Corot':>12}  {'NeoHookean':>12}"
          f"  {'CR/Lin':>8}  {'NL/Lin':>8}")
    print("  " + "-" * 68)
    for i in range(len(loads)):
        r_cr = tip_cr[i] / tip_linear[i] if tip_linear[i] > 0 else 0
        r_nl = tip_nl[i] / tip_linear[i] if tip_linear[i] > 0 else 0
        print(f"  {loads[i]:8.4f}  {tip_linear[i]:12.6f}  {tip_cr[i]:12.6f}"
              f"  {tip_nl[i]:12.6f}  {r_cr:8.4f}  {r_nl:8.4f}")

    # --- Visualization: deformed mesh (corotational) ---
    ux = u_cr[0::3]
    uy = u_cr[1::3]
    uz = u_cr[2::3]
    disp_mag = np.sqrt(ux ** 2 + uy ** 2 + uz ** 2)

    deformed = nodes.copy()
    deformed[:, 0] += ux
    deformed[:, 1] += uy
    deformed[:, 2] += uz

    print("\nRendering corotational deformed mesh (close window to continue)...")
    show_scalar_field(
        deformed, elements, disp_mag,
        scalar_name="Displacement magnitude",
        title=f"Corotational Cantilever (P={P_max})",
        show_edges=False,
    )

    # --- Load-displacement comparison chart ---
    import pyvista as pv

    print("Showing load-displacement comparison (close window to finish)...")
    chart = pv.Chart2D()
    chart.line(tip_linear, loads, label="Linear", color="tab:orange", style="--")
    chart.line(tip_cr, loads, label="Corotational", color="tab:blue")
    chart.line(tip_nl, loads, label="Neo-Hookean (NL)", color="tab:red", style="-.")
    chart.x_label = "|Tip deflection|"
    chart.y_label = "|Applied load|"
    chart.title = "Linear vs Corotational vs Full Nonlinear"
    chart.legend_visible = True

    pl = pv.Plotter()
    pl.add_chart(chart)
    pl.show(title="Load-Displacement Comparison")

    print("\n" + "=" * 65)
    print("Milestone 4 — corotational FEM demo complete")
    print("=" * 65)


if __name__ == "__main__":
    main()
