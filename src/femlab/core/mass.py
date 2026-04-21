"""Consistent and lumped mass matrices for T3 and Tet4 elements.

Consistent mass:  M_e = ρ ∫ N^T N dV  (full coupling between nodes)
Lumped mass:      diagonal, total element mass distributed equally to nodes.
"""

import numpy as np
from scipy import sparse


# ---------------------------------------------------------------------------
# T3 (2D, 2 DOFs per node)
# ---------------------------------------------------------------------------

def t3_element_mass_consistent(
    coords: np.ndarray,
    rho: float,
    thickness: float = 1.0,
) -> np.ndarray:
    """Consistent element mass matrix for a T3 element (6×6).

    For a linear triangle, ∫ Ni Nj dA = A * (1 + δij) / 12.

    Args:
        coords: (3, 2) element vertex coordinates.
        rho: Mass density.
        thickness: Out-of-plane thickness.

    Returns:
        (6, 6) symmetric element mass matrix.
    """
    dx1 = coords[1] - coords[0]
    dx2 = coords[2] - coords[0]
    area = 0.5 * abs(dx1[0] * dx2[1] - dx1[1] * dx2[0])

    # Scalar mass sub-matrix (3×3): m_ij = ρ*t*A * (1 + δij) / 12
    m_scalar = rho * thickness * area / 12.0 * (np.ones((3, 3)) + np.eye(3))

    # Expand to 6×6 (interleaved DOFs: u0,v0,u1,v1,u2,v2)
    me = np.zeros((6, 6))
    for i in range(3):
        for j in range(3):
            me[2 * i, 2 * j] = m_scalar[i, j]
            me[2 * i + 1, 2 * j + 1] = m_scalar[i, j]
    return me


def t3_element_mass_lumped(
    coords: np.ndarray,
    rho: float,
    thickness: float = 1.0,
) -> np.ndarray:
    """Lumped element mass matrix for a T3 element (6×6 diagonal).

    Total mass = ρ * t * A, distributed equally to 3 nodes.

    Args:
        coords: (3, 2) element vertex coordinates.
        rho: Mass density.
        thickness: Out-of-plane thickness.

    Returns:
        (6, 6) diagonal element mass matrix.
    """
    dx1 = coords[1] - coords[0]
    dx2 = coords[2] - coords[0]
    area = 0.5 * abs(dx1[0] * dx2[1] - dx1[1] * dx2[0])
    m_node = rho * thickness * area / 3.0
    return np.diag(np.full(6, m_node))


# ---------------------------------------------------------------------------
# Tet4 (3D, 3 DOFs per node)
# ---------------------------------------------------------------------------

def tet4_element_mass_consistent(
    coords: np.ndarray,
    rho: float,
) -> np.ndarray:
    """Consistent element mass matrix for a Tet4 element (12×12).

    For a linear tet, ∫ Ni Nj dV = V * (1 + δij) / 20.

    Args:
        coords: (4, 3) element vertex coordinates.
        rho: Mass density.

    Returns:
        (12, 12) symmetric element mass matrix.
    """
    d1 = coords[1] - coords[0]
    d2 = coords[2] - coords[0]
    d3 = coords[3] - coords[0]
    vol = abs(np.dot(d1, np.cross(d2, d3))) / 6.0

    # Scalar mass sub-matrix (4×4): m_ij = ρ*V * (1 + δij) / 20
    m_scalar = rho * vol / 20.0 * (np.ones((4, 4)) + np.eye(4))

    me = np.zeros((12, 12))
    for i in range(4):
        for j in range(4):
            for d in range(3):
                me[3 * i + d, 3 * j + d] = m_scalar[i, j]
    return me


def tet4_element_mass_lumped(
    coords: np.ndarray,
    rho: float,
) -> np.ndarray:
    """Lumped element mass matrix for a Tet4 element (12×12 diagonal).

    Total mass = ρ * V, distributed equally to 4 nodes.

    Args:
        coords: (4, 3) element vertex coordinates.
        rho: Mass density.

    Returns:
        (12, 12) diagonal element mass matrix.
    """
    d1 = coords[1] - coords[0]
    d2 = coords[2] - coords[0]
    d3 = coords[3] - coords[0]
    vol = abs(np.dot(d1, np.cross(d2, d3))) / 6.0
    m_node = rho * vol / 4.0
    return np.diag(np.full(12, m_node))


# ---------------------------------------------------------------------------
# Global mass assembly
# ---------------------------------------------------------------------------

def assemble_global_mass_t3(
    nodes: np.ndarray,
    elements: np.ndarray,
    rho: float,
    thickness: float = 1.0,
    lumped: bool = False,
) -> sparse.csr_matrix:
    """Assemble the global mass matrix from T3 elements.

    Args:
        nodes: (N, 2) node coordinates.
        elements: (M, 3) element connectivity.
        rho: Mass density.
        thickness: Out-of-plane thickness.
        lumped: If True, use lumped mass; otherwise consistent.

    Returns:
        (2N, 2N) sparse CSR mass matrix.
    """
    n_nodes = len(nodes)
    n_dof = 2 * n_nodes
    n_elem = len(elements)

    rows = np.zeros(n_elem * 36, dtype=np.int64)
    cols = np.zeros(n_elem * 36, dtype=np.int64)
    vals = np.zeros(n_elem * 36, dtype=np.float64)

    mass_fn = t3_element_mass_lumped if lumped else t3_element_mass_consistent

    for e in range(n_elem):
        n0, n1, n2 = elements[e]
        coords_e = nodes[[n0, n1, n2]]
        me = mass_fn(coords_e, rho, thickness)

        dof_map = np.array([2 * n0, 2 * n0 + 1,
                            2 * n1, 2 * n1 + 1,
                            2 * n2, 2 * n2 + 1])

        offset = e * 36
        idx = 0
        for i in range(6):
            for j in range(6):
                rows[offset + idx] = dof_map[i]
                cols[offset + idx] = dof_map[j]
                vals[offset + idx] = me[i, j]
                idx += 1

    M = sparse.coo_matrix((vals, (rows, cols)), shape=(n_dof, n_dof))
    return M.tocsr()


def assemble_global_mass_tet4(
    nodes: np.ndarray,
    elements: np.ndarray,
    rho: float,
    lumped: bool = False,
) -> sparse.csr_matrix:
    """Assemble the global mass matrix from Tet4 elements.

    Args:
        nodes: (N, 3) node coordinates.
        elements: (M, 4) element connectivity.
        rho: Mass density.
        lumped: If True, use lumped mass; otherwise consistent.

    Returns:
        (3N, 3N) sparse CSR mass matrix.
    """
    n_nodes = len(nodes)
    n_dof = 3 * n_nodes
    n_elem = len(elements)

    rows = np.zeros(n_elem * 144, dtype=np.int64)
    cols = np.zeros(n_elem * 144, dtype=np.int64)
    vals = np.zeros(n_elem * 144, dtype=np.float64)

    mass_fn = tet4_element_mass_lumped if lumped else tet4_element_mass_consistent

    for e in range(n_elem):
        el_nodes = elements[e]
        coords_e = nodes[el_nodes]
        me = mass_fn(coords_e, rho)

        dof_map = np.empty(12, dtype=np.int64)
        for i in range(4):
            dof_map[3 * i]     = 3 * el_nodes[i]
            dof_map[3 * i + 1] = 3 * el_nodes[i] + 1
            dof_map[3 * i + 2] = 3 * el_nodes[i] + 2

        offset = e * 144
        idx = 0
        for i in range(12):
            for j in range(12):
                rows[offset + idx] = dof_map[i]
                cols[offset + idx] = dof_map[j]
                vals[offset + idx] = me[i, j]
                idx += 1

    M = sparse.coo_matrix((vals, (rows, cols)), shape=(n_dof, n_dof))
    return M.tocsr()
