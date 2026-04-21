"""Cantilever beam benchmark for 2D linear static FEM.

A cantilever beam is fixed on the left edge and loaded with a
point force P at the tip (mid-height of the right edge).

Analytical reference (Euler-Bernoulli beam theory):
    δ_tip = P L^3 / (3 E I)
where I = t * h^3 / 12 for a rectangular cross-section.

For the FEM solution with T3 elements (constant strain), convergence
is first-order, so we expect ~5-10% error on coarse meshes and
convergence toward the analytical solution with refinement.

Regression tolerances are documented per test.
"""

import numpy as np

from femlab.core.assembly import assemble_global_stiffness
from femlab.core.boundary import apply_dirichlet
from femlab.core.material import plane_stress_D
from femlab.core.solver import solve_linear
from femlab.mesh.rectangle import rectangle_mesh, boundary_nodes


def _solve_cantilever(nx, ny, L, H, E, nu, P, thickness=1.0):
    """Set up and solve a cantilever beam problem.

    Returns (nodes, elements, u_full, tip_node_index).
    """
    D = plane_stress_D(E, nu)
    nodes, elements = rectangle_mesh(Lx=L, Ly=H, nx=nx, ny=ny)
    n_nodes = len(nodes)

    # Assemble.
    K = assemble_global_stiffness(nodes, elements, D, thickness)
    f = np.zeros(2 * n_nodes)

    # Apply tip load at mid-height of right edge.
    right = boundary_nodes(nx, ny, "right")
    # Find node closest to (L, H/2).
    right_coords = nodes[right]
    mid_idx = np.argmin(np.abs(right_coords[:, 1] - H / 2.0))
    tip_node = right[mid_idx]
    f[2 * tip_node + 1] = P  # vertical load

    # Fix left edge (all DOFs).
    left = boundary_nodes(nx, ny, "left")
    bc_dofs = []
    bc_vals = []
    for ni in left:
        bc_dofs.extend([2 * ni, 2 * ni + 1])
        bc_vals.extend([0.0, 0.0])

    K_mod, f_mod = apply_dirichlet(K, f, np.array(bc_dofs), np.array(bc_vals))
    u = solve_linear(K_mod, f_mod, verbose=False)

    return nodes, elements, u, tip_node


def test_cantilever_tip_deflection():
    """Cantilever tip deflection within 15% of Euler-Bernoulli on a refined mesh.

    T3 (constant-strain) elements are stiff in bending, so a finer mesh
    is needed for reasonable accuracy.

    Parameters: L=10, H=1, E=1000, ν=0.0, P=-1 (downward).
    Mesh: 40×8 elements.

    Analytical: δ = PL³/(3EI) = -1*1000/(3*1000*(1/12)) = -4.0
    Regression tolerance: 15% relative error on 40×8 mesh.
    """
    L, H = 10.0, 1.0
    E, nu = 1000.0, 0.0  # ν=0 for clean comparison with beam theory
    P = -1.0
    thickness = 1.0
    I = thickness * H ** 3 / 12.0
    delta_exact = P * L ** 3 / (3.0 * E * I)

    _, _, u, tip_node = _solve_cantilever(40, 8, L, H, E, nu, P, thickness)
    delta_fem = u[2 * tip_node + 1]

    rel_error = abs((delta_fem - delta_exact) / delta_exact)
    assert rel_error < 0.15, (
        f"Cantilever tip deflection: FEM={delta_fem:.6f}, "
        f"exact={delta_exact:.6f}, rel_error={rel_error:.4f}"
    )


def test_cantilever_convergence():
    """Mesh refinement must improve tip deflection accuracy.

    Compare 10×2 vs 40×8 meshes — the finer mesh must have smaller error.
    Regression tolerance: finer error < coarser error.
    """
    L, H = 10.0, 1.0
    E, nu = 1000.0, 0.0
    P = -1.0
    I = H ** 3 / 12.0
    delta_exact = P * L ** 3 / (3.0 * E * I)

    _, _, u_coarse, tip_c = _solve_cantilever(10, 2, L, H, E, nu, P)
    _, _, u_fine, tip_f = _solve_cantilever(40, 8, L, H, E, nu, P)

    err_coarse = abs(u_coarse[2 * tip_c + 1] - delta_exact)
    err_fine = abs(u_fine[2 * tip_f + 1] - delta_exact)

    assert err_fine < err_coarse, (
        f"Finer mesh should be more accurate: "
        f"err_coarse={err_coarse:.6e}, err_fine={err_fine:.6e}"
    )


def test_cantilever_fixed_end_zero():
    """All DOFs on the fixed (left) edge must be exactly zero.

    Regression tolerance: 1e-14.
    """
    L, H = 10.0, 1.0
    E, nu = 1000.0, 0.3
    P = -1.0
    nx, ny = 10, 2

    nodes, _, u, _ = _solve_cantilever(nx, ny, L, H, E, nu, P)
    left = boundary_nodes(nx, ny, "left")
    for ni in left:
        np.testing.assert_allclose(u[2 * ni], 0.0, atol=1e-14,
                                   err_msg=f"Left node {ni} u_x nonzero")
        np.testing.assert_allclose(u[2 * ni + 1], 0.0, atol=1e-14,
                                   err_msg=f"Left node {ni} u_y nonzero")
