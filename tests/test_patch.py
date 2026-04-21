"""Patch test for 2D linear elasticity with T3 elements.

The patch test verifies that the FEM pipeline (assembly, BCs, solve)
can reproduce a known uniform strain field exactly.  This is the
fundamental correctness check for any displacement-based FEM.

Test setup:
    - Rectangular domain [0, 2] × [0, 1], meshed with T3 elements.
    - Plane stress, E = 1000, ν = 0.3.
    - Prescribe displacements on ALL boundary nodes corresponding to
      a uniform strain field εxx = 0.001, εyy = -0.0003, γxy = 0.
    - Solve for interior node displacements.
    - Check that interior displacements match the analytical field
      and that element strains are uniform.

Regression tolerance: 1e-10 for displacements, 1e-10 for strains.
"""

import numpy as np

from femlab.core.assembly import assemble_global_stiffness
from femlab.core.boundary import apply_dirichlet
from femlab.core.element import t3_stress
from femlab.core.material import plane_stress_D
from femlab.core.solver import solve_linear
from femlab.mesh.rectangle import rectangle_mesh, boundary_nodes


def test_patch_uniform_strain():
    """Patch test: uniform strain field must be reproduced exactly."""
    # --- Setup ---
    E, nu = 1000.0, 0.3
    D = plane_stress_D(E, nu)
    nx, ny = 4, 2
    nodes, elements = rectangle_mesh(Lx=2.0, Ly=1.0, nx=nx, ny=ny)
    n_nodes = len(nodes)

    # Prescribed uniform strain: εxx = 0.001, εyy = -ν*εxx (Poisson), γxy = 0
    # For simplicity use an arbitrary linear displacement field:
    #   u(x,y) = a*x + b*y
    #   v(x,y) = c*x + d*y
    # giving εxx=a, εyy=d, γxy=b+c.
    a, d = 0.001, -0.0003
    b, c = 0.0, 0.0  # no shear

    def exact_disp(x, y):
        return a * x + b * y, c * x + d * y

    # Identify boundary nodes.
    left = set(boundary_nodes(nx, ny, "left"))
    right = set(boundary_nodes(nx, ny, "right"))
    bottom = set(boundary_nodes(nx, ny, "bottom"))
    top = set(boundary_nodes(nx, ny, "top"))
    bc_node_set = left | right | bottom | top

    # Build BC arrays.
    bc_dofs = []
    bc_vals = []
    for ni in bc_node_set:
        ux, uy = exact_disp(nodes[ni, 0], nodes[ni, 1])
        bc_dofs.extend([2 * ni, 2 * ni + 1])
        bc_vals.extend([ux, uy])
    bc_dofs = np.array(bc_dofs)
    bc_vals = np.array(bc_vals)

    # --- Assemble and solve ---
    K = assemble_global_stiffness(nodes, elements, D)
    f = np.zeros(2 * n_nodes)
    K_mod, f_mod = apply_dirichlet(K, f, bc_dofs, bc_vals)
    u = solve_linear(K_mod, f_mod, verbose=False)

    # --- Check interior displacements ---
    interior_nodes = [i for i in range(n_nodes) if i not in bc_node_set]
    for ni in interior_nodes:
        ux_exact, uy_exact = exact_disp(nodes[ni, 0], nodes[ni, 1])
        # Regression tolerance: 1e-10.
        np.testing.assert_allclose(u[2 * ni], ux_exact, atol=1e-10,
                                   err_msg=f"Node {ni} u_x mismatch")
        np.testing.assert_allclose(u[2 * ni + 1], uy_exact, atol=1e-10,
                                   err_msg=f"Node {ni} u_y mismatch")

    # --- Check element strains are uniform ---
    expected_strain = np.array([a, d, b + c])
    for e in range(len(elements)):
        n0, n1, n2 = elements[e]
        coords_e = nodes[[n0, n1, n2]]
        u_e = np.array([u[2 * n0], u[2 * n0 + 1],
                        u[2 * n1], u[2 * n1 + 1],
                        u[2 * n2], u[2 * n2 + 1]])
        strain, _ = t3_stress(coords_e, D, u_e)
        # Regression tolerance: 1e-10.
        np.testing.assert_allclose(strain, expected_strain, atol=1e-10,
                                   err_msg=f"Element {e} strain mismatch")


def test_patch_pure_shear():
    """Patch test with pure shear: γxy = 0.001, εxx = εyy = 0."""
    E, nu = 1000.0, 0.25
    D = plane_stress_D(E, nu)
    nx, ny = 3, 3
    nodes, elements = rectangle_mesh(Lx=1.0, Ly=1.0, nx=nx, ny=ny)
    n_nodes = len(nodes)

    # u = 0.0005 * y, v = 0.0005 * x  → γxy = 0.001
    gamma = 0.001

    def exact_disp(x, y):
        return 0.5 * gamma * y, 0.5 * gamma * x

    bc_node_set = (set(boundary_nodes(nx, ny, "left")) |
                   set(boundary_nodes(nx, ny, "right")) |
                   set(boundary_nodes(nx, ny, "bottom")) |
                   set(boundary_nodes(nx, ny, "top")))
    bc_dofs, bc_vals = [], []
    for ni in bc_node_set:
        ux, uy = exact_disp(nodes[ni, 0], nodes[ni, 1])
        bc_dofs.extend([2 * ni, 2 * ni + 1])
        bc_vals.extend([ux, uy])

    K = assemble_global_stiffness(nodes, elements, D)
    f = np.zeros(2 * n_nodes)
    K_mod, f_mod = apply_dirichlet(K, f, np.array(bc_dofs), np.array(bc_vals))
    u = solve_linear(K_mod, f_mod, verbose=False)

    # Check interior displacements.
    interior = [i for i in range(n_nodes) if i not in bc_node_set]
    for ni in interior:
        ux_exact, uy_exact = exact_disp(nodes[ni, 0], nodes[ni, 1])
        np.testing.assert_allclose(u[2 * ni], ux_exact, atol=1e-10)
        np.testing.assert_allclose(u[2 * ni + 1], uy_exact, atol=1e-10)

    # Check uniform strain.
    expected_strain = np.array([0.0, 0.0, gamma])
    for e in range(len(elements)):
        n0, n1, n2 = elements[e]
        coords_e = nodes[[n0, n1, n2]]
        u_e = np.array([u[2 * n0], u[2 * n0 + 1],
                        u[2 * n1], u[2 * n1 + 1],
                        u[2 * n2], u[2 * n2 + 1]])
        strain, _ = t3_stress(coords_e, D, u_e)
        # Regression tolerance: 1e-10.
        np.testing.assert_allclose(strain, expected_strain, atol=1e-10,
                                   err_msg=f"Element {e} strain mismatch")
