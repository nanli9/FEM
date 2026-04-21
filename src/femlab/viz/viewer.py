"""PyVista-based mesh viewer for FEM results.

Interactive keyboard controls (printed in each window):
    w — cycle display: solid → wireframe → solid+edges
    i — toggle all internal edges (every tet/tri edge, not just surface)
    r — reset camera
    Close the window via the X button to continue.
"""

import numpy as np
import pyvista as pv


_CELL_TYPE_MAP = {
    3: pv.CellType.TRIANGLE,
    4: pv.CellType.TETRA,
    8: pv.CellType.HEXAHEDRON,
}

_CONTROLS_TEXT = "w: cycle display | i: internal edges | r: reset camera"


def _make_unstructured_grid(
    points: np.ndarray,
    cells: np.ndarray,
    cell_type: int | None = None,
) -> pv.UnstructuredGrid:
    """Build a PyVista UnstructuredGrid from raw arrays.

    Args:
        points: (N, 2) or (N, 3) vertex positions.
        cells: (M, K) cell connectivity — each row is one cell.
        cell_type: VTK cell type int.  Inferred from K if None.
    """
    if points.shape[1] == 2:
        points = np.column_stack([points, np.zeros(len(points))])

    k = cells.shape[1]
    if cell_type is None:
        cell_type = _CELL_TYPE_MAP[k]

    n_cells = len(cells)
    vtk_cells = np.column_stack([np.full(n_cells, k, dtype=cells.dtype), cells]).ravel()
    celltypes = np.full(n_cells, cell_type, dtype=np.uint8)
    return pv.UnstructuredGrid(vtk_cells, celltypes, points)


def _install_key_controls(pl, grid, mesh_actor):
    """Override VTK's default key handling to provide proper toggles.

    Intercepts CharEvent at the VTK interactor-style level so that
    built-in keys (e=exit, w=wireframe-only) are replaced with our
    cycling logic, and 'i' toggles internal edge visibility.
    """
    # Pre-extract all edges (including internal) for the 'i' toggle.
    all_edges = grid.extract_all_edges()
    state = {"edges_actor": None, "edges_visible": False}

    def on_char(obj, event):
        vtk_iren = pl.iren.interactor
        key = vtk_iren.GetKeySym()

        if key in ("e", "q"):
            # Block VTK's default exit — user closes via the X button.
            return

        if key == "w":
            prop = mesh_actor.GetProperty()
            rep = prop.GetRepresentation()   # 1=wireframe, 2=surface
            edge_vis = prop.GetEdgeVisibility()
            if rep == 2 and not edge_vis:
                # solid → wireframe
                prop.SetRepresentationToWireframe()
            elif rep == 1:
                # wireframe → solid + edges
                prop.SetRepresentationToSurface()
                prop.EdgeVisibilityOn()
            else:
                # solid + edges → solid
                prop.SetRepresentationToSurface()
                prop.EdgeVisibilityOff()
            pl.render()
            return

        if key == "i":
            if state["edges_actor"] is None:
                state["edges_actor"] = pl.add_mesh(
                    all_edges, color="black", line_width=1,
                    opacity=0.3, name="_internal_edges",
                )
                state["edges_visible"] = True
            else:
                state["edges_visible"] = not state["edges_visible"]
                state["edges_actor"].SetVisibility(state["edges_visible"])
            pl.render()
            return

        # Let VTK handle everything else (r=reset, etc.).
        style = pl.iren.get_interactor_style()
        style.OnChar()

    style = pl.iren.get_interactor_style()
    style.AddObserver("CharEvent", on_char, 100.0)


def show_mesh(
    points: np.ndarray,
    cells: np.ndarray,
    title: str = "Mesh",
    **kwargs,
) -> pv.Plotter:
    """Display a bare mesh wireframe/surface."""
    grid = _make_unstructured_grid(points, cells)
    off = kwargs.get("off_screen", False)
    pl = pv.Plotter(off_screen=off)
    actor = pl.add_mesh(grid, show_edges=True, color="lightblue")
    pl.add_title(title)
    if not off:
        _install_key_controls(pl, grid, actor)
        pl.add_text(_CONTROLS_TEXT, position="lower_left", font_size=8, color="grey")
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

    off = kwargs.get("off_screen", False)
    pl = pv.Plotter(off_screen=off)
    actor = pl.add_mesh(
        grid, scalars=scalar_name, show_edges=show_edges, cmap="viridis",
    )
    pl.add_scalar_bar(scalar_name)
    pl.add_title(title)
    if not off:
        _install_key_controls(pl, grid, actor)
        pl.add_text(_CONTROLS_TEXT, position="lower_left", font_size=8, color="grey")
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

    off = kwargs.get("off_screen", False)
    pl = pv.Plotter(off_screen=off)
    actor = pl.add_mesh(grid, show_edges=True, color="lightblue", opacity=0.3)
    pl.add_mesh(arrows, color="red")
    pl.add_title(title)
    if not off:
        _install_key_controls(pl, grid, actor)
        pl.add_text(_CONTROLS_TEXT, position="lower_left", font_size=8, color="grey")
    pl.show()
    return pl
