"""Structured rectangular mesh generator with T3 triangulation.

Generates a regular grid of (nx × ny) quads, each split into 2 triangles.
Node and element numbering follow a row-major pattern.
"""

import numpy as np


def rectangle_mesh(
    Lx: float = 1.0,
    Ly: float = 1.0,
    nx: int = 4,
    ny: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a structured triangular mesh on a rectangle [0, Lx] × [0, Ly].

    Each quad cell is split into two triangles by its diagonal.

    Args:
        Lx: Domain length in x.
        Ly: Domain length in y.
        nx: Number of elements in x direction.
        ny: Number of elements in y direction.

    Returns:
        nodes: ((nx+1)*(ny+1), 2) array of node coordinates.
        elements: (2*nx*ny, 3) array of element connectivity (CCW orientation).
    """
    x = np.linspace(0.0, Lx, nx + 1)
    y = np.linspace(0.0, Ly, ny + 1)
    xx, yy = np.meshgrid(x, y)
    nodes = np.column_stack([xx.ravel(), yy.ravel()])

    elements = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            n1 = n0 + 1
            n2 = n0 + (nx + 1)
            n3 = n2 + 1
            # Lower-right triangle (CCW: n0 → n1 → n3).
            elements.append([n0, n1, n3])
            # Upper-left triangle (CCW: n0 → n3 → n2).
            elements.append([n0, n3, n2])

    return nodes, np.array(elements, dtype=np.int32)


def boundary_nodes(
    nx: int,
    ny: int,
    side: str,
) -> np.ndarray:
    """Return node indices on a named boundary of the rectangle mesh.

    Args:
        nx: Number of elements in x.
        ny: Number of elements in y.
        side: One of 'left', 'right', 'bottom', 'top'.

    Returns:
        1-D array of node indices on the requested side.
    """
    if side == "left":
        return np.arange(0, (nx + 1) * (ny + 1), nx + 1)
    elif side == "right":
        return np.arange(nx, (nx + 1) * (ny + 1), nx + 1)
    elif side == "bottom":
        return np.arange(0, nx + 1)
    elif side == "top":
        return np.arange(ny * (nx + 1), (ny + 1) * (nx + 1))
    else:
        raise ValueError(f"Unknown side '{side}'. Use left/right/bottom/top.")
