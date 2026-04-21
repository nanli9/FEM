"""Structured box mesh generator with Tet4 elements.

Generates a regular grid of (nx × ny × nz) hexahedra, each split into
6 tetrahedra via the Freudenthal (Kuhn) triangulation.
"""

import numpy as np


def box_mesh(
    Lx: float = 1.0,
    Ly: float = 1.0,
    Lz: float = 1.0,
    nx: int = 4,
    ny: int = 4,
    nz: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a structured Tet4 mesh on a box [0,Lx] × [0,Ly] × [0,Lz].

    Each hex cell is split into 6 tetrahedra (Freudenthal triangulation).

    Args:
        Lx, Ly, Lz: Domain dimensions.
        nx, ny, nz: Number of elements in each direction.

    Returns:
        nodes: ((nx+1)*(ny+1)*(nz+1), 3) array of node coordinates.
        elements: (6*nx*ny*nz, 4) array of element connectivity.
    """
    x = np.linspace(0.0, Lx, nx + 1)
    y = np.linspace(0.0, Ly, ny + 1)
    z = np.linspace(0.0, Lz, nz + 1)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    nodes = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    def node_id(i, j, k):
        return i * (ny + 1) * (nz + 1) + j * (nz + 1) + k

    # 6 tetrahedra per hex, all with positive Jacobian determinant.
    # Freudenthal decomposition along the main diagonal (n000 → n111).
    elements = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                n000 = node_id(i, j, k)
                n100 = node_id(i + 1, j, k)
                n010 = node_id(i, j + 1, k)
                n110 = node_id(i + 1, j + 1, k)
                n001 = node_id(i, j, k + 1)
                n101 = node_id(i + 1, j, k + 1)
                n011 = node_id(i, j + 1, k + 1)
                n111 = node_id(i + 1, j + 1, k + 1)

                # Even permutations (positive orientation).
                elements.append([n000, n100, n110, n111])  # (xyz)
                elements.append([n000, n010, n011, n111])  # (yzx)
                elements.append([n000, n001, n101, n111])  # (zxy)
                # Odd permutations (swap last two for positive orientation).
                elements.append([n000, n100, n111, n101])  # (xzy)
                elements.append([n000, n010, n111, n110])  # (yxz)
                elements.append([n000, n001, n111, n011])  # (zyx)

    return nodes, np.array(elements, dtype=np.int32)


def box_boundary_nodes(
    nx: int,
    ny: int,
    nz: int,
    face: str,
) -> np.ndarray:
    """Return node indices on a named face of the box mesh.

    Args:
        nx, ny, nz: Number of elements in each direction.
        face: One of 'x0', 'x1', 'y0', 'y1', 'z0', 'z1'.
              'x0' means the face at x=0, 'x1' at x=Lx, etc.

    Returns:
        1-D array of node indices on the requested face.
    """
    def node_id(i, j, k):
        return i * (ny + 1) * (nz + 1) + j * (nz + 1) + k

    ids = []
    if face == "x0":
        for j in range(ny + 1):
            for k in range(nz + 1):
                ids.append(node_id(0, j, k))
    elif face == "x1":
        for j in range(ny + 1):
            for k in range(nz + 1):
                ids.append(node_id(nx, j, k))
    elif face == "y0":
        for i in range(nx + 1):
            for k in range(nz + 1):
                ids.append(node_id(i, 0, k))
    elif face == "y1":
        for i in range(nx + 1):
            for k in range(nz + 1):
                ids.append(node_id(i, ny, k))
    elif face == "z0":
        for i in range(nx + 1):
            for j in range(ny + 1):
                ids.append(node_id(i, j, 0))
    elif face == "z1":
        for i in range(nx + 1):
            for j in range(ny + 1):
                ids.append(node_id(i, j, nz))
    else:
        raise ValueError(f"Unknown face '{face}'. Use x0/x1/y0/y1/z0/z1.")

    return np.array(ids, dtype=np.int64)
