#!/usr/bin/env python3
"""Milestone 3 demo: Large-deformation 3D cantilever with Neo-Hookean material.

Solves a 3D cantilever beam under progressively increasing tip load
using Newton-Raphson iteration. Compares nonlinear deflection against
the linear solution and shows geometric stiffening at large loads.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from femlab.core.hyperelastic import lame_parameters, neo_hookean_pk1, neo_hookean_tangent
from femlab.core.newton import solve_newton
from femlab.mesh.box import box_mesh, box_boundary_nodes
from femlab.viz import show_scalar_field


def main():
    print("=" * 60)
    print("FEMLab Milestone 3 — Nonlinear 3D Cantilever (Neo-Hookean)")
    print("=" * 60)

    # --- Problem parameters ---
    L, H, W = 10.0, 1.0, 1.0
    E, nu = 1000.0, 0.3
    nx, ny, nz = 10, 2, 2
    P_max = -5.0       # total tip load (y-direction)
    n_load_steps = 10

    mu, lam = lame_parameters(E, nu)
    I_beam = W * H ** 3 / 12.0

    # Material callables.
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
    print(f"Mesh: {nx}x{ny}x{nz} = {len(elements)} Tet4 elements")
    print(f"Nodes: {n_nodes}, DOFs: {n_dof}")

    # --- BCs: fix x=0 face ---
    left = box_boundary_nodes(nx, ny, nz, "x0")
    bc_dofs = []
    for ni in left:
        bc_dofs.extend([3 * ni, 3 * ni + 1, 3 * ni + 2])
    bc_dofs = np.array(bc_dofs)
    bc_vals = np.zeros(len(bc_dofs))

    # Tip node at center of x=L face.
    right = box_boundary_nodes(nx, ny, nz, "x1")
    right_coords = nodes[right]
    center = np.array([L, H / 2.0, W / 2.0])
    dists = np.linalg.norm(right_coords - center, axis=1)
    tip_node = right[np.argmin(dists)]
    tip_dof = 3 * tip_node + 1
    print(f"Tip node: {tip_node} at {nodes[tip_node]}")

    # --- Load stepping ---
    print(f"\nLoad stepping: {n_load_steps} increments, P_max = {P_max}")
    print(f"Linear tip deflection at P_max: {P_max * L**3 / (3*E*I_beam):.6f}")

    u = np.zeros(n_dof)
    loads = []
    tip_nl = []
    tip_linear = []

    for step in range(1, n_load_steps + 1):
        frac = step / n_load_steps
        P_current = P_max * frac

        f_ext = np.zeros(n_dof)
        f_ext[tip_dof] = P_current

        print(f"\n--- Load step {step}/{n_load_steps}: P = {P_current:.4f} ---")
        result = solve_newton(
            nodes, elements, f_ext, bc_dofs, bc_vals, tangent_fn,
            u0=u, max_iter=30, atol=1e-8, verbose=True,
        )

        if not result["converged"]:
            print("Newton did not converge! Stopping.")
            break

        u = result["u"]
        delta_nl = u[tip_dof]
        delta_lin = P_current * L ** 3 / (3.0 * E * I_beam)

        loads.append(abs(P_current))
        tip_nl.append(abs(delta_nl))
        tip_linear.append(abs(delta_lin))

        print(f"  Tip deflection (NL):     {delta_nl:.6f}")
        print(f"  Tip deflection (linear): {delta_lin:.6f}")
        print(f"  Ratio NL/linear:         {delta_nl / delta_lin:.4f}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("Load-displacement summary:")
    print(f"  {'Load':>8}  {'NL defl':>12}  {'Linear defl':>12}  {'Ratio':>8}")
    print("  " + "-" * 46)
    for i in range(len(loads)):
        ratio = tip_nl[i] / tip_linear[i] if tip_linear[i] > 0 else 0
        print(f"  {loads[i]:8.4f}  {tip_nl[i]:12.6f}  {tip_linear[i]:12.6f}  {ratio:8.4f}")

    # --- Visualization ---
    ux = u[0::3]
    uy = u[1::3]
    uz = u[2::3]
    disp_mag = np.sqrt(ux ** 2 + uy ** 2 + uz ** 2)

    deformed = nodes.copy()
    deformed[:, 0] += ux
    deformed[:, 1] += uy
    deformed[:, 2] += uz

    print("\nRendering deformed mesh (close window to continue)...")
    show_scalar_field(
        deformed, elements, disp_mag,
        scalar_name="Displacement magnitude",
        title=f"Nonlinear Cantilever (P={P_max}, Neo-Hookean)",
        show_edges=False,
    )

    # --- Load-displacement curve ---
    import pyvista as pv

    print("Showing load-displacement curve (close window to finish)...")
    chart = pv.Chart2D()
    chart.line(tip_nl, loads, label="Nonlinear (Neo-Hookean)", color="tab:blue")
    chart.line(tip_linear, loads, label="Linear", color="tab:orange", style="--")
    chart.x_label = "|Tip deflection|"
    chart.y_label = "|Applied load|"
    chart.title = "Load-displacement curve"
    chart.legend_visible = True

    pl = pv.Plotter()
    pl.add_chart(chart)
    pl.show(title="Load-Displacement Curve")

    print("\n" + "=" * 60)
    print("Milestone 3 — nonlinear demo complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
