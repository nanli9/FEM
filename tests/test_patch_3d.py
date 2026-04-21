"""3D patch test for Tet4 elements.

Applies a known uniform strain field as Dirichlet BCs on all boundary
faces of a box mesh, solves, and checks that interior nodes reproduce
the exact displacement and that all element strains are uniform.

Regression tolerance: 1e-9 for displacements, 1e-9 for strains.
"""

import numpy as np

from femlab.core.assembly import assemble_global_stiffness_tet4
from femlab.core.boundary import apply_dirichlet
from femlab.core.element import tet4_stress
from femlab.core.material import isotropic_3d_D
from femlab.core.solver import solve_linear
from femlab.mesh.box import box_mesh, box_boundary_nodes


def test_patch_3d_uniform_stretch():
    """Uniform uniaxial stretch: εxx = 0.001, all other strains zero."""
    E, nu = 1000.0, 0.3
    D = isotropic_3d_D(E, nu)
    nx, ny, nz = 3, 2, 2
    nodes, elements = box_mesh(Lx=2.0, Ly=1.0, Lz=1.0, nx=nx, ny=ny, nz=nz)
    n_nodes = len(nodes)

    eps_xx = 0.001

    def exact_disp(x, y, z):
        return eps_xx * x, 0.0, 0.0

    # All boundary faces.
    bc_node_set = set()
    for face in ["x0", "x1", "y0", "y1", "z0", "z1"]:
        bc_node_set |= set(box_boundary_nodes(nx, ny, nz, face))

    bc_dofs, bc_vals = [], []
    for ni in bc_node_set:
        ux, uy, uz = exact_disp(*nodes[ni])
        bc_dofs.extend([3 * ni, 3 * ni + 1, 3 * ni + 2])
        bc_vals.extend([ux, uy, uz])

    K = assemble_global_stiffness_tet4(nodes, elements, D)
    f = np.zeros(3 * n_nodes)
    K_mod, f_mod = apply_dirichlet(K, f, np.array(bc_dofs), np.array(bc_vals))
    u = solve_linear(K_mod, f_mod, verbose=False)

    # Check interior displacements.
    interior = [i for i in range(n_nodes) if i not in bc_node_set]
    for ni in interior:
        ux_e, uy_e, uz_e = exact_disp(*nodes[ni])
        np.testing.assert_allclose(u[3*ni], ux_e, atol=1e-9,
                                   err_msg=f"Node {ni} ux")
        np.testing.assert_allclose(u[3*ni+1], uy_e, atol=1e-9,
                                   err_msg=f"Node {ni} uy")
        np.testing.assert_allclose(u[3*ni+2], uz_e, atol=1e-9,
                                   err_msg=f"Node {ni} uz")

    # Check element strains.
    expected_strain = np.array([eps_xx, 0, 0, 0, 0, 0])
    for e in range(len(elements)):
        el = elements[e]
        coords_e = nodes[el]
        u_e = np.zeros(12)
        for i in range(4):
            u_e[3*i:3*i+3] = u[3*el[i]:3*el[i]+3]
        strain, _ = tet4_stress(coords_e, D, u_e)
        # Regression tolerance: 1e-9.
        np.testing.assert_allclose(strain, expected_strain, atol=1e-9,
                                   err_msg=f"Element {e} strain")


def test_patch_3d_pure_shear():
    """Pure shear: u = γ/2 * y, v = γ/2 * x, w = 0  →  γxy = γ."""
    E, nu = 1000.0, 0.25
    D = isotropic_3d_D(E, nu)
    nx, ny, nz = 2, 2, 2
    nodes, elements = box_mesh(Lx=1.0, Ly=1.0, Lz=1.0, nx=nx, ny=ny, nz=nz)
    n_nodes = len(nodes)

    gamma = 0.001

    def exact_disp(x, y, z):
        return 0.5 * gamma * y, 0.5 * gamma * x, 0.0

    bc_node_set = set()
    for face in ["x0", "x1", "y0", "y1", "z0", "z1"]:
        bc_node_set |= set(box_boundary_nodes(nx, ny, nz, face))

    bc_dofs, bc_vals = [], []
    for ni in bc_node_set:
        ux, uy, uz = exact_disp(*nodes[ni])
        bc_dofs.extend([3 * ni, 3 * ni + 1, 3 * ni + 2])
        bc_vals.extend([ux, uy, uz])

    K = assemble_global_stiffness_tet4(nodes, elements, D)
    f = np.zeros(3 * n_nodes)
    K_mod, f_mod = apply_dirichlet(K, f, np.array(bc_dofs), np.array(bc_vals))
    u = solve_linear(K_mod, f_mod, verbose=False)

    # Check interior displacements.
    interior = [i for i in range(n_nodes) if i not in bc_node_set]
    for ni in interior:
        ux_e, uy_e, uz_e = exact_disp(*nodes[ni])
        np.testing.assert_allclose(u[3*ni], ux_e, atol=1e-9)
        np.testing.assert_allclose(u[3*ni+1], uy_e, atol=1e-9)
        np.testing.assert_allclose(u[3*ni+2], uz_e, atol=1e-9)

    # Check uniform strain.
    expected_strain = np.array([0, 0, 0, gamma, 0, 0])
    for e in range(len(elements)):
        el = elements[e]
        coords_e = nodes[el]
        u_e = np.zeros(12)
        for i in range(4):
            u_e[3*i:3*i+3] = u[3*el[i]:3*el[i]+3]
        strain, _ = tet4_stress(coords_e, D, u_e)
        # Regression tolerance: 1e-9.
        np.testing.assert_allclose(strain, expected_strain, atol=1e-9,
                                   err_msg=f"Element {e} strain")
