"""Nonlinear global assembly for Tet4 elements.

Assembles internal force vectors and tangent stiffness matrices
from element contributions for finite-deformation analysis.
Uses the same COO -> CSR pattern as the linear assembly module.
"""

import numpy as np
from scipy import sparse

from femlab.core.element_nl import tet4_internal_force_nl, tet4_tangent_stiffness_nl


def _build_dof_map(el_nodes: np.ndarray) -> np.ndarray:
    """Build the 12-entry DOF map for a Tet4 element."""
    dof_map = np.empty(12, dtype=np.int64)
    for i in range(4):
        dof_map[3 * i] = 3 * el_nodes[i]
        dof_map[3 * i + 1] = 3 * el_nodes[i] + 1
        dof_map[3 * i + 2] = 3 * el_nodes[i] + 2
    return dof_map


def assemble_internal_force_tet4_nl(
    nodes: np.ndarray,
    elements: np.ndarray,
    u: np.ndarray,
    pk1_fn,
) -> np.ndarray:
    """Assemble the global internal force vector (nonlinear).

    Args:
        nodes: (N, 3) reference node coordinates.
        elements: (M, 4) element connectivity.
        u: (3N,) global displacement vector.
        pk1_fn: Callable F -> P (3x3).

    Returns:
        f_int: (3N,) global internal force vector.
    """
    n_dof = 3 * len(nodes)
    f_int = np.zeros(n_dof)

    for e in range(len(elements)):
        el_nodes = elements[e]
        X_ref = nodes[el_nodes]
        dof_map = _build_dof_map(el_nodes)
        u_e = u[dof_map]

        fe = tet4_internal_force_nl(X_ref, u_e, pk1_fn)
        for i in range(12):
            f_int[dof_map[i]] += fe[i]

    return f_int


def assemble_system_tet4_nl(
    nodes: np.ndarray,
    elements: np.ndarray,
    u: np.ndarray,
    tangent_fn,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Assemble global tangent stiffness and internal force in one pass.

    More efficient than separate assembly since F and G are computed once
    per element.

    Args:
        nodes: (N, 3) reference node coordinates.
        elements: (M, 4) element connectivity.
        u: (3N,) global displacement vector.
        tangent_fn: Callable F -> (P, A) where P is (3,3), A is (9,9).

    Returns:
        K_T: (3N, 3N) sparse CSR tangent stiffness matrix.
        f_int: (3N,) global internal force vector.
    """
    n_dof = 3 * len(nodes)
    n_elem = len(elements)

    # COO storage for stiffness.
    rows = np.zeros(n_elem * 144, dtype=np.int64)
    cols = np.zeros(n_elem * 144, dtype=np.int64)
    vals = np.zeros(n_elem * 144, dtype=np.float64)

    f_int = np.zeros(n_dof)

    for e in range(n_elem):
        el_nodes = elements[e]
        X_ref = nodes[el_nodes]
        dof_map = _build_dof_map(el_nodes)
        u_e = u[dof_map]

        ke, fe = tet4_tangent_stiffness_nl(X_ref, u_e, tangent_fn)

        # Scatter internal force.
        for i in range(12):
            f_int[dof_map[i]] += fe[i]

        # Scatter stiffness into COO.
        offset = e * 144
        idx = 0
        for i in range(12):
            for j in range(12):
                rows[offset + idx] = dof_map[i]
                cols[offset + idx] = dof_map[j]
                vals[offset + idx] = ke[i, j]
                idx += 1

    K_T = sparse.coo_matrix((vals, (rows, cols)), shape=(n_dof, n_dof))
    return K_T.tocsr(), f_int
