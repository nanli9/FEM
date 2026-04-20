"""PyVista-based mesh viewer for FEM results."""

import numpy as np
import pyvista as pv


_CELL_TYPE_MAP = {
    3: pv.CellType.TRIANGLE,
    4: pv.CellType.TETRA,
    8: pv.CellType.HEXAHEDRON,
}


def _make_unstructured_grid(
    points: np.ndarray,
    cells: np.ndarray,
    cell_type: int | None = None,
) -> pv.UnstructuredGrid:
    """Build a PyVista UnstructuredGrid from raw arrays.

    Args:
        points: (N, 3) vertex positions.
        cells: (M, K) cell connectivity — each row is one cell.
        cell_type: VTK cell type int.  Inferred from K if None.
    """
    if points.shape[1] == 2:
        points = np.column_stack([points, np.zeros(len(points))])

    k = cells.shape[1]
    if cell_type is None:
        cell_type = _CELL_TYPE_MAP[k]

    # VTK legacy format: each cell is [n_pts, p0, p1, ..., pn]
    n_cells = len(cells)
    vtk_cells = np.column_stack([np.full(n_cells, k, dtype=cells.dtype), cells]).ravel()
    celltypes = np.full(n_cells, cell_type, dtype=np.uint8)
    return pv.UnstructuredGrid(vtk_cells, celltypes, points)


def show_mesh(
    points: np.ndarray,
    cells: np.ndarray,
    title: str = "Mesh",
    **kwargs,
) -> pv.Plotter:
    """Display a bare mesh wireframe/surface."""
    grid = _make_unstructured_grid(points, cells)
    pl = pv.Plotter(off_screen=kwargs.get("off_screen", False))
    pl.add_mesh(grid, show_edges=True, color="lightblue")
    pl.add_title(title)
    if not kwargs.get("off_screen", False):
        pl.show()
    return pl


def show_scalar_field(
    points: np.ndarray,
    cells: np.ndarray,
    scalars: np.ndarray,
    scalar_name: str = "field",
    title: str = "Scalar Field",
    show_edges: bool = True,
    **kwargs,
) -> pv.Plotter:
    """Display mesh colored by a point-based scalar field."""
    grid = _make_unstructured_grid(points, cells)
    grid.point_data[scalar_name] = scalars

    pl = pv.Plotter(off_screen=kwargs.get("off_screen", False))
    pl.add_mesh(grid, scalars=scalar_name, show_edges=show_edges, cmap="viridis")
    pl.add_scalar_bar(scalar_name)
    pl.add_title(title)
    if not kwargs.get("off_screen", False):
        pl.show()
    return pl


def show_vector_field(
    points: np.ndarray,
    cells: np.ndarray,
    vectors: np.ndarray,
    vector_name: str = "vectors",
    scale: float = 1.0,
    title: str = "Vector Field",
    **kwargs,
) -> pv.Plotter:
    """Display mesh with vector arrows at each vertex."""
    grid = _make_unstructured_grid(points, cells)
    grid.point_data[vector_name] = vectors

    arrows = grid.glyph(orient=vector_name, scale=False, factor=scale)

    pl = pv.Plotter(off_screen=kwargs.get("off_screen", False))
    pl.add_mesh(grid, show_edges=True, color="lightblue", opacity=0.3)
    pl.add_mesh(arrows, color="red")
    pl.add_title(title)
    if not kwargs.get("off_screen", False):
        pl.show()
    return pl
