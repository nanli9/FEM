"""Global sparse assembly of stiffness matrix and force vector.

Builds the global system K u = f from element contributions using
COO (coordinate) format, then converts to CSR for efficient solving.
"""

import numpy as np
from scipy import sparse

from femlab.core.element import t3_element_stiffness


def assemble_global_stiffness(
    nodes: np.ndarray,
    elements: np.ndarray,
    D: np.ndarray,
    thickness: float = 1.0,
) -> sparse.csr_matrix:
    """Assemble the global stiffness matrix from T3 elements.

    Args:
        nodes: (N, 2) array of node coordinates.
        elements: (M, 3) array of element connectivity (node indices).
        D: (3, 3) constitutive matrix.
        thickness: Out-of-plane thickness.

    Returns:
        (2N, 2N) sparse CSR global stiffness matrix.
    """
    n_nodes = len(nodes)
    n_dof = 2 * n_nodes
    n_elem = len(elements)

    # Pre-allocate COO arrays: each T3 contributes 6x6 = 36 entries.
    rows = np.zeros(n_elem * 36, dtype=np.int64)
    cols = np.zeros(n_elem * 36, dtype=np.int64)
    vals = np.zeros(n_elem * 36, dtype=np.float64)

    for e in range(n_elem):
        n0, n1, n2 = elements[e]
        coords_e = nodes[[n0, n1, n2]]  # (3, 2)
        ke = t3_element_stiffness(coords_e, D, thickness)  # (6, 6)

        # Global DOF indices for this element.
        dof_map = np.array([2 * n0, 2 * n0 + 1,
                            2 * n1, 2 * n1 + 1,
                            2 * n2, 2 * n2 + 1])

        # Scatter into COO arrays.
        offset = e * 36
        idx = 0
        for i in range(6):
            for j in range(6):
                rows[offset + idx] = dof_map[i]
                cols[offset + idx] = dof_map[j]
                vals[offset + idx] = ke[i, j]
                idx += 1

    K = sparse.coo_matrix((vals, (rows, cols)), shape=(n_dof, n_dof))
    return K.tocsr()


def assemble_global_force(
    nodes: np.ndarray,
    elements: np.ndarray,
    body_force: np.ndarray | None = None,
    thickness: float = 1.0,
) -> np.ndarray:
    """Assemble the global force vector from distributed body forces.

    For T3 elements under constant body force b = [bx, by], the
    consistent nodal force per element is: f_e = b * A_e * thickness / 3
    distributed equally to each node.

    Args:
        nodes: (N, 2) node coordinates.
        elements: (M, 3) element connectivity.
        body_force: (2,) body force vector [bx, by]. None → zero.
        thickness: Out-of-plane thickness.

    Returns:
        (2N,) global force vector.
    """
    n_dof = 2 * len(nodes)
    f = np.zeros(n_dof)

    if body_force is None:
        return f

    bx, by = body_force
    for e in range(len(elements)):
        n0, n1, n2 = elements[e]
        coords_e = nodes[[n0, n1, n2]]
        # Triangle area via cross product: A = 0.5 * |det(J)|
        dx1 = coords_e[1] - coords_e[0]
        dx2 = coords_e[2] - coords_e[0]
        area = 0.5 * abs(dx1[0] * dx2[1] - dx1[1] * dx2[0])

        # Equal distribution to 3 nodes.
        force_per_node = area * thickness / 3.0
        for ni in [n0, n1, n2]:
            f[2 * ni] += bx * force_per_node
            f[2 * ni + 1] += by * force_per_node

    return f
